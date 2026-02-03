"""
V&C 보스 AI v4 - 동적 β/γ
==========================

v4 추가:
- β (안전 민감도): HP 기반 광폭화
  HP 높음 → 안전 플레이
  HP 낮음 → 이판사판 돌진
  폭딜 맞으면 패닉 (일시 회피)

- γ (예측 의존도): 적중률 피드백
  공격 적중 → 과감하게
  공격 빗나감 → 보수적으로

구조:
1. 포지셔닝 예측 (자유에너지 원리)
2. 길찾기 (A*)
3. c₀ 계산 (MMR 기반)
4. 전투 AI (동적 β/γ)
"""

import numpy as np
import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

# ============================================================
# 상수 정의
# ============================================================

class Role(Enum):
    TANK = "tank"
    OFFTANK = "offtank"
    MELEE_DPS = "melee_dps"
    RANGED_DPS = "ranged_dps"
    HEALER = "healer"

# 직업별 스탯
CLASS_STATS = {
    Role.TANK:       {"hp": 160, "dps": 12,  "optimal_dist": 2,  "aggro_weight": 0.5},
    Role.OFFTANK:    {"hp": 130, "dps": 16,  "optimal_dist": 3,  "aggro_weight": 0.7},
    Role.MELEE_DPS:  {"hp": 85,  "dps": 35,  "optimal_dist": 3,  "aggro_weight": 1.0},
    Role.RANGED_DPS: {"hp": 75,  "dps": 30,  "optimal_dist": 8,  "aggro_weight": 0.8},
    Role.HEALER:     {"hp": 80,  "dps": 6,   "optimal_dist": 6,  "aggro_weight": 1.5},
}

# 타겟 우선순위 (높을수록 먼저 맞음)
TARGET_PRIORITY = {
    Role.HEALER: 4,
    Role.MELEE_DPS: 3,
    Role.RANGED_DPS: 3,
    Role.OFFTANK: 2,
    Role.TANK: 1,
}

HEAL_PER_TURN = 22
GRID_SIZE = 30
BOSS_HP = 1925  # 기본값 (티어 1 기준)


# ============================================================
# 추가: 티어별 스케일링
# ============================================================

TIER_SCALING = {
    1:  {"hp_mult": 0.4,  "dmg_mult": 0.4, "fake_rate": 0.05, "patterns": 5,  "heal_mult": 0.4},
    2:  {"hp_mult": 0.5,  "dmg_mult": 0.5, "fake_rate": 0.08, "patterns": 8,  "heal_mult": 0.5},
    3:  {"hp_mult": 0.65, "dmg_mult": 0.65, "fake_rate": 0.12, "patterns": 12,  "heal_mult": 0.65},
    4:  {"hp_mult": 0.8,  "dmg_mult": 0.8, "fake_rate": 0.18, "patterns": 16,  "heal_mult": 0.8},
    5:  {"hp_mult": 1.0,  "dmg_mult": 1.0, "fake_rate": 0.25, "patterns": 20, "heal_mult": 1.0},
    6:  {"hp_mult": 1.3,  "dmg_mult": 1.25, "fake_rate": 0.32, "patterns": 25, "heal_mult": 1.25},
    7:  {"hp_mult": 1.7,  "dmg_mult": 1.55, "fake_rate": 0.40, "patterns": 32, "heal_mult": 1.55},
    8:  {"hp_mult": 2.2,  "dmg_mult": 1.9, "fake_rate": 0.50, "patterns": 40, "heal_mult": 1.9},
    9:  {"hp_mult": 2.9,  "dmg_mult": 2.4, "fake_rate": 0.58, "patterns": 50, "heal_mult": 2.4},
    10: {"hp_mult": 3.8,  "dmg_mult": 3.0, "fake_rate": 0.65, "patterns": 65, "heal_mult": 3.0},
}

def get_tier_settings(tier: int) -> dict:
    """티어별 설정 반환"""
    tier = np.clip(tier, 1, 10)
    return TIER_SCALING[tier]


# ============================================================
# 추가: 휴먼 에러 시스템
# ============================================================

def get_human_error_rate(mmr: int) -> float:
    """MMR 기반 실수율: 고수 5%, 초보 15%"""
    rate = 0.15 - 0.10 * (mmr - 800) / (1900 - 800)
    return np.clip(rate, 0.05, 0.20)

def check_heal_timing(mmr: int) -> bool:
    """힐 타이밍 성공 여부"""
    return np.random.random() > get_human_error_rate(mmr)


# ============================================================
# 추가: 직업별 방어 메커니즘
# ============================================================

# (방어타입, 기본성공률, 데미지감소율)
# 패링: 막아서 데미지 감소
# 회피: 피해서 데미지 0
# None: 방어 불가
DEFENSE_MECHANISM = {
    Role.TANK:       ("parry", 0.7, 0.5),   # 70% 패링, 50% 감소
    Role.OFFTANK:    ("parry", 0.5, 0.4),   # 50% 패링, 40% 감소
    Role.MELEE_DPS:  ("dodge", 0.4, 1.0),   # 40% 회피, 100% 회피
    Role.RANGED_DPS: (None, 0, 0),          # 방어 없음
    Role.HEALER:     (None, 0, 0),          # 방어 없음
}

def check_defense(role: Role, mmr: int) -> Tuple[str, float]:
    """
    방어 체크
    
    반환: (결과, 데미지 배율)
    - ("parry", 0.5): 패링 성공, 50% 데미지
    - ("dodge", 0.0): 회피 성공, 0% 데미지
    - ("hit", 1.0): 맞음, 100% 데미지
    - ("crit", 1.2): 피격 실패, 120% 데미지
    """
    defense_type, base_rate, reduction = DEFENSE_MECHANISM[role]
    
    if defense_type is None:
        # 방어 수단 없음 → 무조건 맞음
        return ("hit", 1.0)
    
    # 실수율 적용 (고수일수록 방어 잘함)
    error_rate = get_human_error_rate(mmr)
    success_rate = base_rate * (1 - error_rate)
    
    if np.random.random() < success_rate:
        # 방어 성공
        if defense_type == "parry":
            return ("parry", 1 - reduction)  # 데미지 감소
        else:  # dodge
            return ("dodge", 0.0)  # 완전 회피
    else:
        # 방어 실패 → 휴먼 에러면 추가 데미지
        if np.random.random() < error_rate:
            return ("crit", 1.2)  # 뼈아픈 실수
        return ("hit", 1.0)


# ============================================================
# 추가: 스탯 배율
# ============================================================

def get_stat_multiplier(mmr: int, tier: int = 5) -> float:
    """
    MMR + 티어 기반 스탯 배율
    
    낮은 티어: 고수도 장비 아낌 → 스탯 낮음
    높은 티어: 풀셋 → 스탯 높음
    
    고수 1.4, 초보 0.6 (차이 2.33배)
    """
    base = 0.6 + 0.8 * (mmr - 800) / (1900 - 800)
    
    # 티어별 장비 수준 (1~10 → 0.6~1.0)
    gear_level = 0.6 + 0.4 * (tier - 1) / 9
    
    return np.clip(base * gear_level, 0.4, 1.4)


# ============================================================
# 추가: 가중치 타겟팅
# ============================================================

TARGET_WEIGHTS = {
    Role.HEALER: 4.0,
    Role.RANGED_DPS: 2.0,
    Role.MELEE_DPS: 2.0,
    Role.OFFTANK: 1.0,
    Role.TANK: 0.5,
}

def select_targets_weighted(candidates, n: int):
    """가중치 기반 타겟 선택"""
    if not candidates:
        return []
    
    alive = [c for c in candidates if c.alive]
    if not alive:
        return []
    
    selected = []
    remaining = alive.copy()
    
    for _ in range(min(n, len(remaining))):
        if not remaining:
            break
        weights = [TARGET_WEIGHTS.get(c.role, 1.0) for c in remaining]
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = np.random.choice(len(remaining), p=probs)
        selected.append(remaining[idx])
        remaining.pop(idx)
    
    return selected


# ============================================================
# 1. 포지셔닝 예측 (자유에너지 원리)
# ============================================================

class PositionPredictor:
    """
    자유에너지 원리 기반 포지셔닝 예측
    
    p(s) ∝ exp(-β·V_boss) × H_user^γ
         = (안전확률) × (유저성향)^γ
    
    v4: 동적 β/γ
    - β: HP 기반 (딸피 → 공격적)
    - γ: 적중률 피드백 (맞추면 과감, 빗나가면 보수적)
    - γ 낮으면 현재 판 데이터 신뢰
    """
    
    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.beta = 0.3
        self.gamma = 1.2
        self.current_session = {}  # {user_id: [pos1, pos2, ...]}
        
    def update_beta(self, current_hp: float, max_hp: float, recent_damage: float = 0):
        hp_ratio = current_hp / max_hp
        base_beta = 0.1 + 0.9 * (hp_ratio ** 2)
        panic_factor = 2.0 if recent_damage > max_hp * 0.1 else 1.0
        self.beta = np.clip(base_beta * panic_factor, 0.05, 2.0)
        return self.beta
    
    def update_gamma(self, was_hit: bool):
        lr = 0.1
        if was_hit:
            self.gamma = min(2.0, self.gamma + lr)
        else:
            self.gamma = max(0.5, self.gamma - lr * 2)
        return self.gamma
    
    def record_session_pos(self, user_id: str, pos: Tuple[int, int]):
        """현재 판 위치 기록"""
        if user_id not in self.current_session:
            self.current_session[user_id] = []
        self.current_session[user_id].append(pos)
    
    def get_session_H(self, user_id: str) -> np.ndarray:
        """현재 판 데이터 기반 H"""
        H = np.zeros((self.grid_size, self.grid_size))
        if user_id not in self.current_session:
            return H
        
        positions = self.current_session[user_id]
        if not positions:
            return H
        
        # 최근 위치일수록 가중치 높음
        for i, pos in enumerate(positions):
            weight = (i + 1) / len(positions)
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                y, x = np.ogrid[:self.grid_size, :self.grid_size]
                H += weight * np.exp(-0.5 * ((x - pos[0])**2 + (y - pos[1])**2) / 2.0)
        
        return H / (H.sum() + 1e-10)
    
    def reset_session(self):
        """판 끝나면 리셋"""
        self.current_session = {}
        self.gamma = 1.2
        
    def compute_boss_potential(self, boss_pos: Tuple[int, int]) -> np.ndarray:
        """보스로부터의 위험도 (거리 기반)"""
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        dist = np.sqrt((x - boss_pos[0])**2 + (y - boss_pos[1])**2)
        dist = np.maximum(dist, 0.1)
        
        # 가까울수록 위험
        V = 1.0 / dist
        return V
    
    def compute_user_preference(self, 
                                boss_pos: Tuple[int, int],
                                role: Role,
                                obstacles: np.ndarray = None) -> np.ndarray:
        """
        유저 성향 히스토그램
        - 직업별 최적 거리 선호
        - 후방 선호 (보스 뒤)
        """
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        dist = np.sqrt((x - boss_pos[0])**2 + (y - boss_pos[1])**2)
        
        # 직업별 최적 거리
        optimal_dist = CLASS_STATS[role]["optimal_dist"]
        dist_preference = np.exp(-0.5 * ((dist - optimal_dist) / 2.0)**2)
        
        # 후방 선호 (보스 기준 아래쪽 = 후방)
        behind_bonus = np.where(y > boss_pos[1], 1.3, 1.0)
        
        H = dist_preference * behind_bonus
        
        # 장애물 처리
        if obstacles is not None:
            H = H * (1 - obstacles)
        
        return H / (H.sum() + 1e-10)
    
    def predict_position(self,
                        boss_pos: Tuple[int, int],
                        role: Role,
                        obstacles: np.ndarray = None) -> np.ndarray:
        """
        자유에너지 원리로 위치 확률 예측
        
        p(s) ∝ exp(-β·V_boss) × H_user^γ
        """
        V = self.compute_boss_potential(boss_pos)
        H = self.compute_user_preference(boss_pos, role, obstacles)
        
        # 자유에너지 원리
        safety = np.exp(-self.beta * V)
        preference = np.power(H + 1e-10, self.gamma)
        
        p = safety * preference
        p = p / (p.sum() + 1e-10)
        
        return p
    
    def get_optimal_position(self,
                            boss_pos: Tuple[int, int],
                            role: Role,
                            obstacles: np.ndarray = None) -> Tuple[int, int]:
        """최적 위치 반환"""
        p = self.predict_position(boss_pos, role, obstacles)
        idx = np.unravel_index(np.argmax(p), p.shape)
        return (idx[1], idx[0])  # (x, y)


# ============================================================
# 2. 길찾기 (A*)
# ============================================================

class Pathfinder:
    """A* 길찾기 알고리즘"""
    
    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """맨해튼 거리"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int], obstacles: np.ndarray = None) -> List[Tuple[int, int]]:
        """이웃 노드 반환 (8방향)"""
        x, y = pos
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                
                # 범위 체크
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                
                # 장애물 체크
                if obstacles is not None and obstacles[ny, nx] > 0.5:
                    continue
                    
                neighbors.append((nx, ny))
        
        return neighbors
    
    def find_path(self,
                  start: Tuple[int, int],
                  goal: Tuple[int, int],
                  obstacles: np.ndarray = None) -> List[Tuple[int, int]]:
        """A* 경로 탐색"""
        
        if start == goal:
            return [start]
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # 경로 재구성
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for neighbor in self.get_neighbors(current, obstacles):
                # 대각선 이동은 비용 √2
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.414 if dx + dy == 2 else 1.0
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 경로 없음 → 직선 이동
        return [start, goal]


# ============================================================
# 3. c₀ 계산
# ============================================================

class C0Calculator:
    """
    c₀ = 포지셔닝 품질 점수 (0~1)
    
    두 가지 모드:
    1. 실시간: 현재 위치 기반 계산
    2. MMR 기반: 어뷰징 방지
    """
    
    def __init__(self):
        self.predictor = PositionPredictor()
    
    def calculate_realtime(self,
                          player_pos: Tuple[int, int],
                          boss_pos: Tuple[int, int],
                          role: Role) -> float:
        """
        실시간 c₀ 계산
        = 현재 위치가 최적 위치에 얼마나 가까운지
        """
        optimal = self.predictor.get_optimal_position(boss_pos, role)
        
        dist = np.sqrt((player_pos[0] - optimal[0])**2 + 
                       (player_pos[1] - optimal[1])**2)
        
        c0 = np.exp(-dist / 3.0)
        return np.clip(c0, 0.1, 1.0)
    
    def calculate_from_mmr(self, mmr: float) -> float:
        """
        MMR 기반 c₀ (어뷰징 방지)
        
        고수가 일부러 못하는 척 해도:
        → MMR은 여전히 높음
        → c₀ 안 내려감
        → 보스 여전히 진심
        
        시그모이드 함수:
        MMR 800  → c₀ 0.27
        MMR 1200 → c₀ 0.50
        MMR 1900 → c₀ 0.83
        """
        c0 = 0.1 + 0.8 / (1 + np.exp(-(mmr - 1200) / 300))
        return np.clip(c0, 0.1, 0.95)
    
    def calculate_party_c0(self, mmr_list: List[float]) -> float:
        """파티 평균 c₀"""
        return np.mean([self.calculate_from_mmr(mmr) for mmr in mmr_list])


# ============================================================
# 4. 전투 AI
# ============================================================

@dataclass
class PartyMember:
    name: str
    role: Role
    hp: int
    max_hp: int
    pos: Tuple[int, int] = (0, 0)
    mmr: int = 1200
    
    @property
    def alive(self) -> bool:
        return self.hp > 0
    
    @property
    def dps(self) -> int:
        return CLASS_STATS[self.role]["dps"]


class BossAI:
    """
    보스 AI
    
    핵심 메커니즘:
    1. c₀ 기반 회복/데미지
    2. 파티 HP 비례 회복 → 막판 역전 가능
    3. 페이즈별 패턴
    4. 티어별 스케일링
    5. v4: 동적 β/γ
    """
    
    def __init__(self, hp: int = BOSS_HP, tier: int = 5):
        self.tier = tier
        self.tier_settings = get_tier_settings(tier)
        
        self.max_hp = int(hp * self.tier_settings["hp_mult"])
        self.hp = self.max_hp
        self.pos = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.pathfinder = Pathfinder()
        self.c0_calc = C0Calculator()
        self.predictor = PositionPredictor()
        self.recent_damage = 0
        
    def get_heal_amount(self, c0: float, party_hp_ratio: float, k: int = 55) -> int:
        """
        보스 회복 = c₀ × 파티HP비율 × k × 티어배율
        
        파티 건강할 때: 많이 회복 → DPS 레이스 힘듦
        파티 빈사일 때: 적게 회복 → 막판 역전 가능!
        """
        heal_mult = self.tier_settings["heal_mult"]
        return int(c0 * party_hp_ratio * k * heal_mult)
        return int(c0 * party_hp_ratio * k)
    
    def get_damage(self, c0: float, base_dmg: int) -> int:
        """
        보스 데미지
        
        c₀ 낮으면 (초보): 데미지 증가 (페널티)
        랜덤 ±15% 변동
        티어별 데미지 배율 적용
        """
        variance = np.random.uniform(0.85, 1.15)
        dmg_mult = self.tier_settings["dmg_mult"]
        
        if c0 > 0.4:
            return int(base_dmg * variance * dmg_mult)
        
        # 초보 페널티
        penalty = 1.0 + (0.4 - c0) * 2.0
        return int(base_dmg * penalty * variance * dmg_mult)
    
    def is_fake_attack(self) -> bool:
        """페이크 공격 여부 (티어별 확률)"""
        return np.random.random() < self.tier_settings["fake_rate"]
    
    def select_targets(self, party: List[PartyMember], n_targets: int) -> List[PartyMember]:
        """타겟 선정 (가중치 기반 확률적)"""
        return select_targets_weighted(party, n_targets)
    
    def get_phase(self) -> Tuple[int, int, int]:
        """
        페이즈별 패턴
        
        반환: (n_targets, base_damage, phase_num)
        티어별 데미지 증가
        """
        hp_ratio = self.hp / self.max_hp
        
        # 기본 데미지 (티어 배율은 get_damage에서 적용)
        if hp_ratio > 0.6:
            return (1, 28, 1)   # Phase 1: 단일 타겟
        elif hp_ratio > 0.3:
            return (2, 33, 2)   # Phase 2: 2타겟
        else:
            return (3, 40, 3)   # Phase 3: 광폭화
    
    def move_towards(self, target_pos: Tuple[int, int], obstacles: np.ndarray = None):
        """타겟을 향해 이동 (길찾기)"""
        path = self.pathfinder.find_path(self.pos, target_pos, obstacles)
        
        if len(path) > 1:
            # 한 칸 이동
            self.pos = path[1]


# ============================================================
# 5. 전투 시뮬레이션
# ============================================================

class BattleSimulator:
    """전투 시뮬레이터"""
    
    def __init__(self, boss_hp: int = BOSS_HP, tier: int = 5):
        self.tier = tier
        self.boss = BossAI(boss_hp, tier)
        self.c0_calc = C0Calculator()
        self.predictor = PositionPredictor()
        self.pathfinder = Pathfinder()
        self.adaptive_ai = AdaptiveBossAI()  # 개인화 + 메타 학습
        
    def create_party(self, mmr_list: List[int] = None) -> List[PartyMember]:
        """파티 생성"""
        if mmr_list is None:
            mmr_list = [1200] * 8
            
        roles = [
            (Role.TANK, "탱커"),
            (Role.OFFTANK, "서브탱"),
            (Role.MELEE_DPS, "암살자1"),
            (Role.MELEE_DPS, "암살자2"),
            (Role.RANGED_DPS, "궁수"),
            (Role.RANGED_DPS, "마법사"),
            (Role.HEALER, "힐러1"),
            (Role.HEALER, "힐러2"),
        ]
        
        party = []
        for i, (role, name) in enumerate(roles):
            stats = CLASS_STATS[role]
            mmr = mmr_list[i] if i < len(mmr_list) else 1200
            
            # 스탯 배율 적용 (티어 반영)
            stat_mult = get_stat_multiplier(mmr, self.tier)
            hp = int(stats["hp"] * stat_mult)
            
            # 초기 위치: 최적 위치로
            optimal_pos = self.predictor.get_optimal_position(self.boss.pos, role)
            
            party.append(PartyMember(
                name=name,
                role=role,
                hp=hp,
                max_hp=hp,
                pos=optimal_pos,
                mmr=mmr
            ))
        
        return party
    
    def run_battle(self, 
                   mmr_list: List[int] = None,
                   verbose: bool = False) -> Tuple[bool, dict]:
        """
        전투 실행
        
        반환: (클리어 여부, 상세 로그)
        """
        # 초기화
        self.boss = BossAI(BOSS_HP, self.tier)
        party = self.create_party(mmr_list)
        
        avg_c0 = self.c0_calc.calculate_party_c0([m.mmr for m in party])
        max_party_hp = sum(m.max_hp for m in party)
        
        log = {
            "turns": [],
            "result": None,
            "avg_c0": avg_c0,
            "boss_hp": BOSS_HP,
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"전투 시작! 보스 HP: {BOSS_HP}, 파티 c₀: {avg_c0:.2f}")
            print(f"{'='*60}")
        
        for turn in range(25):
            alive = [m for m in party if m.alive]
            if not alive:
                log["result"] = "전멸"
                if verbose:
                    print(f"\n☠️ 전멸 (턴 {turn})")
                return False, log
            
            turn_log = {"turn": turn + 1, "events": []}
            
            # 파티 HP 비율
            party_hp = sum(max(0, m.hp) for m in party)
            party_hp_ratio = party_hp / max_party_hp
            
            # ---- 유저 이동 (포지셔닝 + 길찾기) ----
            for m in alive:
                optimal = self.predictor.get_optimal_position(self.boss.pos, m.role)
                path = self.pathfinder.find_path(m.pos, optimal)
                if len(path) > 1:
                    m.pos = path[1]
                
                # 위치 기록 → 보스가 학습
                self.adaptive_ai.record_position(m.name, m.role, m.pos, m.mmr)
                # v4: 현재 판 데이터도 기록
                self.boss.predictor.record_session_pos(m.name, m.pos)
            
            # ---- 딜링 ----
            dps = sum(m.dps for m in alive)
            old_boss_hp = self.boss.hp
            self.boss.hp -= dps
            self.boss.recent_damage = old_boss_hp - self.boss.hp
            
            # v4: 동적 β 업데이트 (HP + 피해량 기반)
            self.boss.predictor.update_beta(self.boss.hp, self.boss.max_hp, self.boss.recent_damage)
            
            turn_log["events"].append(f"DPS {dps} → 보스 {self.boss.hp}")
            
            if verbose:
                print(f"\n[턴 {turn+1}] 파티HP: {party_hp_ratio*100:.0f}%")
                print(f"  ⚔️ DPS: {dps} → 보스: {self.boss.hp}")
            
            if self.boss.hp <= 0:
                log["result"] = "클리어"
                log["turns"].append(turn_log)
                if verbose:
                    print(f"\n🎉 클리어! (턴 {turn+1})")
                return True, log
            
            # ---- 힐링 (휴먼 에러 적용) ----
            for healer in [m for m in alive if m.role == Role.HEALER]:
                # 힐 타이밍 체크
                if not check_heal_timing(healer.mmr):
                    if verbose:
                        print(f"  💔 {healer.name} 힐 타이밍 놓침!")
                    continue
                    
                injured = [m for m in alive if m.hp < m.max_hp]
                if injured:
                    target = min(injured, key=lambda x: x.hp / x.max_hp)
                    old_hp = target.hp
                    target.hp = min(target.max_hp, target.hp + HEAL_PER_TURN)
                    
                    if verbose and target.hp > old_hp:
                        print(f"  💚 {healer.name} → {target.name}: +{target.hp - old_hp}")
            
            # ---- 보스 회복 ----
            heal = self.boss.get_heal_amount(avg_c0, party_hp_ratio)
            self.boss.hp = min(BOSS_HP, self.boss.hp + heal)
            
            turn_log["events"].append(f"보스 회복 +{heal}")
            
            if verbose and heal > 0:
                print(f"  💜 보스 회복: +{heal}")
            
            # ---- 보스 이동 (타겟 향해) ----
            target_candidates = self.boss.select_targets(party, 1)
            if target_candidates:
                self.boss.move_towards(target_candidates[0].pos)
            
            # ---- 보스 공격 ----
            alive = [m for m in party if m.alive]
            if not alive:
                log["result"] = "전멸"
                return False, log
            
            n_targets, base_dmg, phase = self.boss.get_phase()
            targets = self.boss.select_targets(party, n_targets)
            
            # 페이크 공격 체크
            if self.boss.is_fake_attack():
                if verbose:
                    print(f"  💨 Phase {phase}: 페이크! (공격 취소)")
                log["turns"].append(turn_log)
                continue
            
            if verbose:
                print(f"  🔥 Phase {phase}: {[t.name for t in targets]}")
            
            for target in targets:
                base_dmg_val = self.boss.get_damage(avg_c0, base_dmg)
                
                # 직업별 방어 체크
                defense_result, dmg_mult = check_defense(target.role, target.mmr)
                dmg = int(base_dmg_val * dmg_mult)
                
                # 결과 문자열
                if defense_result == "parry":
                    defense_str = " (패링!)"
                elif defense_result == "dodge":
                    defense_str = " (회피!)"
                elif defense_result == "crit":
                    defense_str = " (피격!)"
                else:
                    defense_str = ""
                
                old_hp = target.hp
                target.hp -= dmg
                
                # v4: 동적 γ 업데이트 (적중 여부)
                was_hit = (dmg > 0 and defense_result != "dodge")
                self.boss.predictor.update_gamma(was_hit)
                
                status = " ☠️" if target.hp <= 0 else ""
                turn_log["events"].append(f"{target.name}: {old_hp} → {max(0, target.hp)}{status}")
                
                if verbose:
                    print(f"    {target.name}: -{dmg}{defense_str} → {max(0, target.hp)}{status}")
            
            log["turns"].append(turn_log)
        
        log["result"] = "시간초과"
        if verbose:
            print(f"\n⏰ 시간 초과")
        return False, log
    
    def run_test(self, mmr_list: List[int], n_iterations: int = 1000) -> float:
        """n회 시뮬레이션 후 클리어율 반환"""
        clears = 0
        for i in range(n_iterations):
            np.random.seed(i)
            # 매 게임마다 adaptive AI 리셋 (새 파티)
            self.adaptive_ai = AdaptiveBossAI()
            cleared, _ = self.run_battle(mmr_list)
            if cleared:
                clears += 1
        return clears / n_iterations


# ============================================================
# 6. 시각화 (텍스트 기반)
# ============================================================

def visualize_positioning(boss_pos: Tuple[int, int], 
                         party: List[PartyMember],
                         grid_size: int = 15):
    """파티 포지셔닝 시각화"""
    
    grid = [['·' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # 보스 위치 (중앙 기준으로 조정)
    cx, cy = grid_size // 2, grid_size // 2
    grid[cy][cx] = '👹'
    
    # 파티원 위치
    symbols = {
        Role.TANK: '🛡️',
        Role.OFFTANK: '⚔️',
        Role.MELEE_DPS: '🗡️',
        Role.RANGED_DPS: '🏹',
        Role.HEALER: '💚',
    }
    
    for m in party:
        # 보스 기준 상대 위치
        rx = m.pos[0] - boss_pos[0] + cx
        ry = m.pos[1] - boss_pos[1] + cy
        
        if 0 <= rx < grid_size and 0 <= ry < grid_size:
            grid[ry][rx] = symbols.get(m.role, '?')
    
    print("\n포지셔닝:")
    for row in grid:
        print(' '.join(row))


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("V&C 보스 AI 찐완성본")
    print("=" * 65)
    
    print("""
[구조]
1. 포지셔닝 예측 (자유에너지 원리)
   p(s) ∝ exp(-β·V_boss) × H_user^γ
   
2. 길찾기 (A*)
   
3. c₀ 계산 (MMR 기반 → 어뷰징 방지)
   c₀ = 0.1 + 0.8 / (1 + exp(-(MMR - 1200) / 300))
   
4. 전투 AI
   보스 회복 = c₀ × 파티HP비율 × k
""")
    
    # 시뮬레이터 생성
    sim = BattleSimulator()
    c0_calc = C0Calculator()
    
    # 파티 구성
    pro_mmr = [1900] * 8
    mix_mmr = [1900, 1900, 1900, 1900, 800, 800, 800, 800]
    noob_mmr = [800] * 8
    
    pro_c0 = c0_calc.calculate_party_c0(pro_mmr)
    mix_c0 = c0_calc.calculate_party_c0(mix_mmr)
    noob_c0 = c0_calc.calculate_party_c0(noob_mmr)
    
    print(f"[파티 c₀]")
    print(f"  고수 8명 (MMR 1900): c₀ = {pro_c0:.2f}")
    print(f"  랜덤 매칭 (혼합):    c₀ = {mix_c0:.2f}")
    print(f"  초보 8명 (MMR 800):  c₀ = {noob_c0:.2f}")
    
    # 보스 회복량 비교
    print(f"\n[보스 회복량]")
    print(f"  {'파티':<10} {'HP100%':<10} {'HP50%':<10} {'HP20%':<10}")
    print(f"  {'-'*42}")
    
    boss = BossAI()
    for c0, name in [(pro_c0, "고수"), (mix_c0, "혼합"), (noob_c0, "초보")]:
        h100 = boss.get_heal_amount(c0, 1.0)
        h50 = boss.get_heal_amount(c0, 0.5)
        h20 = boss.get_heal_amount(c0, 0.2)
        print(f"  {name:<10} {h100:<10} {h50:<10} {h20:<10}")
    
    # 클리어율 테스트
    print(f"\n[클리어율 - 1000회 시뮬레이션]")
    print(f"  {'-'*42}")
    
    for mmr_list, name in [(pro_mmr, "고수 8명"), (mix_mmr, "랜덤 매칭"), (noob_mmr, "초보 8명")]:
        rate = sim.run_test(mmr_list, 1000)
        bar = "█" * int(rate * 20)
        print(f"  {name:<12}: {rate*100:>5.1f}% {bar}")
    
    print(f"""
[설계 의도]
✓ 랜덤 매칭 ~50% → 평균 2번 트라이로 클리어
✓ 고수끼리 ~8% → 오히려 어려움
✓ 초보도 랜덤 매칭으로 클리어 가능
✓ 확정 드랍 + 랜덤 매칭 = 파티 구하기 스트레스 없음!
""")
    
    # 포지셔닝 시각화
    print("=" * 65)
    print("포지셔닝 예측 테스트")
    print("=" * 65)
    
    predictor = PositionPredictor()
    boss_pos = (15, 15)
    
    print(f"\n보스 위치: {boss_pos}")
    print(f"\n직업별 최적 위치:")
    
    for role in Role:
        optimal = predictor.get_optimal_position(boss_pos, role)
        dist = np.sqrt((optimal[0] - boss_pos[0])**2 + (optimal[1] - boss_pos[1])**2)
        expected = CLASS_STATS[role]["optimal_dist"]
        print(f"  {role.value:<12}: {optimal}, 거리 {dist:.1f}m (설정: {expected}m)")
    
    # 상세 전투 시뮬레이션
    print("\n" + "=" * 65)
    print("상세 전투: 랜덤 매칭 (혼합 파티)")
    print("=" * 65)
    
    np.random.seed(42)
    sim.run_battle(mix_mmr, verbose=True)
    
    # 길찾기 테스트
    print("\n" + "=" * 65)
    print("길찾기 테스트")
    print("=" * 65)
    
    pathfinder = Pathfinder(grid_size=10)
    
    # 장애물 생성
    obstacles = np.zeros((10, 10))
    obstacles[3:7, 5] = 1  # 세로 벽
    
    start = (1, 5)
    goal = (8, 5)
    
    path = pathfinder.find_path(start, goal, obstacles)
    
    print(f"\n시작: {start}, 목표: {goal}")
    print(f"장애물: (5, 3~6)")
    print(f"경로: {path}")
    
    # 시각화
    grid = [['·' for _ in range(10)] for _ in range(10)]
    for y in range(10):
        for x in range(10):
            if obstacles[y, x] > 0:
                grid[y][x] = '█'
    
    for i, (x, y) in enumerate(path):
        if i == 0:
            grid[y][x] = 'S'
        elif i == len(path) - 1:
            grid[y][x] = 'G'
        else:
            grid[y][x] = '○'
    
    print("\n맵:")
    for row in grid:
        print(' '.join(row))
    
    print(f"\n경로 길이: {len(path)} 칸")


# ============================================================
# 8. 개인화 + 메타 학습 AI
# ============================================================

class AdaptiveBossAI:
    """
    H_user 분리 학습:
    1. H_individual: 유저별 개인화 (습관 파악)
    2. H_global: 전체 메타 학습 (MMR 가중치)
    
    최종: H = α × H_individual + (1-α) × H_global
    
    보스가 유저 습관을 학습해서 예측 저격
    → 고수도 박살남
    """
    
    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.predictor = PositionPredictor(grid_size)
        
        # Cold 데이터 (직업별 기본)
        self.H_cold = {}
        for role in Role:
            self.H_cold[role] = self.predictor.compute_user_preference(
                (grid_size//2, grid_size//2), role
            )
        
        # 개인화: {user_id: {role: H}}
        self.H_individual = {}
        self.individual_counts = {}  # 판 수 기록
        
        # 메타: {role: H}
        self.H_global = {role: self.H_cold[role].copy() for role in Role}
        self.global_counts = {role: 1 for role in Role}
    
    def record_position(self, user_id: str, role: Role, pos: Tuple[int, int], mmr: int):
        """
        유저 위치 기록 → H 업데이트
        
        고수 데이터 = 메타가 됨 (MMR 가중치)
        """
        # 위치를 히트맵으로 변환
        H_new = np.zeros((self.grid_size, self.grid_size))
        if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
            # 가우시안 블러로 부드럽게
            y, x = np.ogrid[:self.grid_size, :self.grid_size]
            H_new = np.exp(-0.5 * ((x - pos[0])**2 + (y - pos[1])**2) / 4.0)
            H_new /= H_new.sum() + 1e-10
        
        # === 개인화 업데이트 ===
        if user_id not in self.H_individual:
            self.H_individual[user_id] = {r: self.H_cold[r].copy() for r in Role}
            self.individual_counts[user_id] = {r: 1 for r in Role}
        
        n = self.individual_counts[user_id][role]
        self.H_individual[user_id][role] = (
            n * self.H_individual[user_id][role] + H_new
        ) / (n + 1)
        self.individual_counts[user_id][role] = n + 1
        
        # === 메타 업데이트 (MMR 가중치) ===
        # MMR 높을수록 가중치 높음 → 고수가 메타 결정
        w_mmr = 0.1 + 0.9 / (1 + np.exp(-(mmr - 1200) / 300))
        
        n_global = self.global_counts[role]
        self.H_global[role] = (
            self.H_global[role] + w_mmr * (H_new - self.H_global[role]) / n_global
        )
        self.global_counts[role] = n_global + 1
    
    def get_H_user(self, user_id: str, role: Role, boss_type: str = "normal") -> np.ndarray:
        """
        최종 H_user 계산
        
        γ 높음 → 과거 데이터 (H_individual, H_global)
        γ 낮음 → 현재 판 데이터 (H_session)
        """
        alpha_map = {
            "normal": 0.3,
            "named": 0.7,
            "raid": 0.1,
        }
        alpha = alpha_map.get(boss_type, 0.3)
        
        # 개인 H (과거)
        if user_id in self.H_individual:
            H_ind = self.H_individual[user_id][role]
        else:
            H_ind = self.H_cold[role]
        
        # 메타 H (과거)
        H_glob = self.H_global[role]
        
        # 과거 데이터 조합
        H_past = alpha * H_ind + (1 - alpha) * H_glob
        
        # 현재 판 데이터
        H_session = self.predictor.get_session_H(user_id)
        
        # γ 기반 과거 vs 현재 가중치
        # γ 높음(2.0) → 과거 100%
        # γ 낮음(0.5) → 현재 75%, 과거 25%
        gamma = self.predictor.gamma
        past_weight = (gamma - 0.5) / 1.5  # 0.5~2.0 → 0~1
        past_weight = np.clip(past_weight, 0, 1)
        
        if H_session.sum() > 0:
            H = past_weight * H_past + (1 - past_weight) * H_session
        else:
            H = H_past
        
        return H / (H.sum() + 1e-10)
    
    def predict_position(self, user_id: str, role: Role, 
                         boss_pos: Tuple[int, int], 
                         boss_type: str = "normal") -> Tuple[int, int]:
        """
        개인화된 위치 예측 → 보스가 여기로 장판 깔음
        """
        H_user = self.get_H_user(user_id, role, boss_type)
        V = self.predictor.compute_boss_potential(boss_pos)
        
        # 자유에너지 원리
        safety = np.exp(-self.predictor.beta * V)
        preference = np.power(H_user + 1e-10, self.predictor.gamma)
        
        p = safety * preference
        p = p / (p.sum() + 1e-10)
        
        idx = np.unravel_index(np.argmax(p), p.shape)
        return (idx[1], idx[0])
