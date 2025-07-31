"""
EXP3 Algorithm-based Cell On/Off Scenario for KISS Network Simulator - FIXED VERSION

python run_kiss.py -c data/input/exp3_cell_on_off/exp3_training.json

시뮬레이션 + 자동 분석 : python run_kiss.py -c data/input/exp3_cell_on_off/exp3_training.json
시뮬레이션 only : python run_kiss.py -c data/input/exp3_cell_on_off/exp3_training.json --no-analysis
기존 결과만 분석 : python run_kiss.py -c data/input/exp3_cell_on_off/exp3_training.json --analysis-only
별도 분석 스크립트 사용 : python analyze_exp3_results.py --results_dir _/data/output --output_dir my_analysis

"""

import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import dataclasses
from AIMM_simulator import Scenario


class EXP3CellOnOff(Scenario):
    """
    EXP3 algorithm-based cell on/off scenario that dynamically selects which cells to turn off
    to optimize network energy efficiency (throughput/power).
    
    The algorithm maintains weights for each possible combination of cells to turn off (arms)
    and updates them based on the observed rewards (network efficiency).
    """
    
    def __init__(self, sim, k_cells: int, n_cells_off: int, 
                 interval: float = 1.0, delay: float = 0.0,
                 gamma: float = 0.1, warmup_episodes: int = 100,
                 enable_warmup: bool = True, ensure_min_selection: bool = True,
                 linear_power_model: bool = False, power_model_k: float = 0.05,
                 learning_log_file: str = None, final_model_file: str = None,
                 verbose: bool = True):
        """
        Initialize EXP3 Cell On/Off scenario.
        
        Parameters:
        -----------
        sim : Simv2
            The simulation object
        k_cells : int
            Total number of cells in the network
        n_cells_off : int
            Number of cells to turn off simultaneously
        interval : float
            Time interval between decisions (default: 1.0)
        delay : float
            Initial delay before starting the scenario (default: 0.0)
        gamma : float
            Exploration parameter for EXP3 (default: 0.1)
        warmup_episodes : int
            Number of initial episodes for random exploration (default: 100)
        enable_warmup : bool
            Whether to enable warm-up phase (default: True)
        ensure_min_selection : bool
            Whether to ensure each arm is selected at least once (default: True)
        linear_power_model : bool
            Whether to use linear power model based on UE count (default: False)
        power_model_k : float
            Coefficient for linear power model (default: 0.05)
        learning_log_file : str
            Path to save learning progress (default: None)
        final_model_file : str
            Path to save final model (default: None)
        verbose : bool
            Whether to print detailed progress (default: True)
        """
        self.sim = sim
        self.k_cells = k_cells
        self.n_cells_off = n_cells_off
        self.interval = interval
        self.delay = delay
        self.gamma = gamma
        self.warmup_episodes = warmup_episodes if enable_warmup else 0
        self.ensure_min_selection = ensure_min_selection
        self.linear_power_model = linear_power_model
        self.power_model_k = power_model_k
        self.verbose = verbose
       
        # Regret 추적을 위한 변수 추가
        self.instant_regret_history = []  # 매 스텝의 순간 regret
        self.cumulative_regret_history = []  # 누적 regret
        self.cumulative_regret = 0.0
        self.oracle_cumulative_reward = 0.0  # 최적 arm의 누적 보상
        self.actual_cumulative_reward = 0.0  # 실제 얻은 누적 보상    
        
        # 추가 메트릭 추적 변수
        self.energy_consumption_history = []  # 에너지 소비 이력
        self.throughput_history = []  # 셀 throughput 이력
        self.cell_states_history = []  # 셀 상태 이력
        self.switching_count = 0  # 전환 횟수
        self.last_selected_arm = None  # 마지막 선택된 arm
        self.baseline_energy_consumption = None  # 베이스라인 에너지 소비
        self.random_baseline_efficiency = None  # 랜덤 on/off 베이스라인

        # Create log directory if needed
        if learning_log_file:
            os.makedirs(os.path.dirname(learning_log_file), exist_ok=True)
        self.learning_log_file = learning_log_file
        self.final_model_file = final_model_file
        
        # Calculate number of arms (combinations of n cells from k cells)
        from math import comb
        self.n_arms = comb(k_cells, n_cells_off)
        
        # Generate all possible combinations of cells to turn off
        from itertools import combinations
        self.arms = list(combinations(range(k_cells), n_cells_off))
        
        # Initialize EXP3 weights
        self.weights = np.ones(self.n_arms)
        self.probabilities = np.ones(self.n_arms) / self.n_arms
        
        # Track statistics
        self.episode_count = 0
        self.arm_selection_count = np.zeros(self.n_arms)
        self.cumulative_rewards = np.zeros(self.n_arms)
        self.reward_history = []
        self.weight_history = []
        self.probability_history = []
        self.selected_arm_history = []
        
        # Store baseline performance (all cells on)
        self.baseline_efficiency = None
        self.baseline_throughput = None
        self.baseline_power = None
        
        # Store original cell powers
        self.original_cell_powers = {}
        
        # Cell energy models reference
        self.cell_energy_models = None
        
        # Reward statistics for dynamic normalization
        self.efficiency_history = []
        self.min_efficiency = float('inf')
        self.max_efficiency = 0.0
        
        if self.verbose:
            print(f"EXP3CellOnOff initialized: {k_cells} cells, turning off {n_cells_off} cells")
            print(f"Total arms (combinations): {self.n_arms}")
            print(f"Exploration parameter (gamma): {gamma}")
            print(f"Warm-up episodes: {self.warmup_episodes if enable_warmup else 'Disabled'}")
        
    def set_cell_energy_models(self, cell_energy_models: Dict):
        """Set reference to cell energy models dictionary"""
        self.cell_energy_models = cell_energy_models
        
    def get_network_metrics(self) -> Tuple[float, float, float]:
        """
        Calculate current network metrics with improved power calculation.
        
        Returns:
        --------
        Tuple[float, float, float]
            (total_throughput_Mbps, total_power_kW, efficiency_bps_per_J)
        """
        total_throughput = 0.0
        total_power = 0.0
        active_cells = 0
        turned_off_cells = 0
        
        # 개선된 전력 계산
        for cell in self.sim.cells:
            # 셀이 꺼져있는지 확인 (더 정확한 조건)
            is_cell_off = (cell.power_dBm == -np.inf or 
                          cell.power_dBm < -50 or 
                          not np.isfinite(cell.power_dBm))
            
            if is_cell_off:
                turned_off_cells += 1
                # 꺼진 셀도 최소한의 전력 소비 (sleep mode)
                total_power += 0.3  # 300W sleep power
                continue
                
            # 활성 셀의 처리량 계산
            try:
                cell_tp = cell.get_cell_throughput()
                if np.isfinite(cell_tp) and cell_tp >= 0:
                    total_throughput += cell_tp
                else:
                    if self.verbose:
                        print(f"Warning: Invalid throughput for cell {cell.i}: {cell_tp}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to get throughput for cell {cell.i}: {e}")
            
            # 활성 셀의 전력 계산
            cell_power_kW = self._calculate_cell_power(cell)
            if cell_power_kW > 0:
                total_power += cell_power_kW
                active_cells += 1
        
        # 전력 검증 및 fallback
        if total_power <= 0.5:  # 너무 낮은 전력값 방지
            # 기본값: 활성 셀당 2kW + 꺼진 셀당 0.1kW
            total_power = active_cells * 2.0 + turned_off_cells * 0.1
            if self.verbose:
                print(f"Warning: Using fallback power calculation. Active={active_cells}, Off={turned_off_cells}")
        
        # 효율성 계산
        if total_power > 0:
            efficiency = (total_throughput * 1e6) / (total_power * 1e3)  # bits/J
        else:
            efficiency = 0.0
            
        # 효율성 통계 업데이트
        if efficiency > 0:
            self.efficiency_history.append(efficiency)
            self.min_efficiency = min(self.min_efficiency, efficiency)
            self.max_efficiency = max(self.max_efficiency, efficiency)
            
        if self.verbose and (self.episode_count % 10 == 0 or total_power < 1.0):
            print(f"  Debug: Active={active_cells}, Off={turned_off_cells}, TP={total_throughput:.2f}, Power={total_power:.2f}")
            
        return total_throughput, total_power, efficiency
    
    
    def calculate_energy_metrics(self) -> Dict[str, float]:
        """
        에너지 관련 메트릭 계산
        
        Returns:
        --------
        dict
            에너지 소비, 절감율 등
        """
        current_power = 0.0
        active_cells = 0
        
        # 각 셀의 전력 소비 계산
        for cell_id, cell in enumerate(self.sim.cells):
            if hasattr(cell, 'tx_power_dBm') and cell.tx_power_dBm > 0:
                # 선형 또는 비선형 전력 모델 적용
                if self.linear_power_model:
                    # 선형 모델: P_total = P_tx + k * P_tx
                    tx_power_watts = 10 ** (cell.tx_power_dBm / 10) / 1000
                    cell_power = tx_power_watts * (1 + self.power_model_k)
                else:
                    # 비선형 모델 (더 현실적)
                    tx_power_watts = 10 ** (cell.tx_power_dBm / 10) / 1000
                    # 고정 오버헤드 + 동적 전력
                    cell_power = 0.5 + tx_power_watts * (1 + self.power_model_k)
                
                current_power += cell_power
                active_cells += 1
        print(f"[DEBUG] calculate_energy_metrics: current_power={current_power:.2f} kW, active_cells={active_cells}")

        baseline_for_comparison = self.baseline_power if self.baseline_power else current_power

        # 에너지 절감율 계산
        energy_saving_all_on = 0.0
        if baseline_for_comparison > 0:
            energy_saving_all_on = (1 - current_power / baseline_for_comparison) * 100
        
        # 랜덤 베이스라인 대비 절감율 (평균 50% 셀 off 가정)
        random_baseline = baseline_for_comparison * 0.75  # 평균적으로 75% 전력 사용
        energy_saving_random = 0.0
        if random_baseline > 0:
            energy_saving_random = (1 - current_power / random_baseline) * 100
        
        return {
            'current_power_kw': current_power,
            'active_cells': active_cells,
            'energy_saving_all_on': energy_saving_all_on,
            'energy_saving_random': energy_saving_random,
            'baseline_power_kw': baseline_for_comparison
        }
    
    def calculate_throughput_metrics(self) -> Dict[str, float]:
        """
        Throughput 관련 메트릭 계산
        
        Returns:
        --------
        dict
            평균 셀 throughput, 총 throughput 등
        """
        total_throughput = 0.0
        active_cells = 0
        cell_throughputs = []
        
        # 각 UE의 throughput 합산
        for ue in self.sim.UEs:
            if hasattr(ue, 'attached_cell') and ue.attached_cell is not None:
                # UE의 data rate 추정 (Shannon capacity 기반)
                if hasattr(ue, 'SINR_dB') and ue.SINR_dB is not None:
                    sinr_linear = 10 ** (ue.SINR_dB / 10)
                    # Shannon capacity: C = B * log2(1 + SINR)
                    # B = 20 MHz 가정
                    bandwidth_hz = 20e6
                    capacity_bps = bandwidth_hz * np.log2(1 + sinr_linear)
                    total_throughput += capacity_bps
        
        # 활성 셀 수 계산
        for cell in self.sim.cells:
            if hasattr(cell, 'tx_power_dBm') and cell.tx_power_dBm > 0:
                active_cells += 1
                # 셀별 throughput 추정 (연결된 UE 수 기반)
                cell_ue_count = sum(1 for ue in self.sim.UEs 
                                  if hasattr(ue, 'attached_cell') and ue.attached_cell == cell)
                if cell_ue_count > 0:
                    cell_throughput = total_throughput * (cell_ue_count / len(self.sim.UEs))
                    cell_throughputs.append(cell_throughput)
        
        avg_cell_throughput = np.mean(cell_throughputs) if cell_throughputs else 0
        
        return {
            'total_throughput_gbps': total_throughput / 1e9,
            'avg_cell_throughput_gbps': avg_cell_throughput / 1e9,
            'active_cells': active_cells,
            'throughput_per_active_cell': (total_throughput / active_cells / 1e9) if active_cells > 0 else 0
        }
    
    def update_switching_cost(self, selected_arm: int):
        """전환 비용 업데이트"""
        if self.last_selected_arm is not None and self.last_selected_arm != selected_arm:
            self.switching_count += 1
        self.last_selected_arm = selected_arm
    
    def is_converged(self, window_size: int = 50, threshold: float = 0.01) -> bool:
        """
        가중치 분포가 수렴했는지 확인
        
        Parameters:
        -----------
        window_size : int
            확인할 윈도우 크기
        threshold : float
            수렴 판단 임계값
            
        Returns:
        --------
        bool
            수렴 여부
        """
        if len(self.probability_history) < window_size:
            return False
        
        recent_probs = self.probability_history[-window_size:]
        prob_changes = []
        
        for i in range(1, len(recent_probs)):
            change = np.mean(np.abs(recent_probs[i] - recent_probs[i-1]))
            prob_changes.append(change)
        
        return np.mean(prob_changes) < threshold

    
    
        
    def _calculate_cell_power(self, cell) -> float:
        """
        Calculate power consumption for a cell using unified approach.
        
        Parameters:
        -----------
        cell : Cell
            Cell object
            
        Returns:
        --------
        float
            Power consumption in kW
        """
        try:
            # 에너지 모델이 있는 경우 - 통합된 방식 사용
            if self.cell_energy_models and cell.i in self.cell_energy_models:
                energy_model = self.cell_energy_models[cell.i]
                
                # CellEnergyModel의 get_cell_power_watts 직접 사용
                # 이미 선형/비선형 모드가 설정되어 있으므로 그냥 호출
                cell_power_watts = energy_model.get_cell_power_watts(self.sim.env.now)
                
                # 전력값 검증
                if np.isfinite(cell_power_watts) and cell_power_watts > 0:
                    return cell_power_watts / 1e3  # kW로 변환
            
            # Fallback은 그대로 유지 (에너지 모델이 없는 경우를 위해)
            if hasattr(cell, 'power_dBm') and np.isfinite(cell.power_dBm):
                rf_power_watts = 10 ** (cell.power_dBm / 10) / 1000
                system_power_watts = 1500 + rf_power_watts * 20
                total_power_watts = rf_power_watts + system_power_watts
                return total_power_watts / 1e3
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Power calculation failed for cell {cell.i}: {e}")
        
        # 최종 fallback
        return 2.0  # 기본 2kW
    
    def turn_cells_off(self, cell_indices: List[int]):
        """Turn off specified cells by setting their power to -inf"""
        for idx in cell_indices:
            if idx < len(self.sim.cells):
                cell = self.sim.cells[idx]
                # Store original power if not already stored
                if idx not in self.original_cell_powers:
                    self.original_cell_powers[idx] = cell.power_dBm
                # Turn off cell
                cell.set_power_dBm(-np.inf)
                
    def turn_cells_on(self, cell_indices: List[int]):
        """Turn on specified cells by restoring their original power"""
        for idx in cell_indices:
            if idx < len(self.sim.cells) and idx in self.original_cell_powers:
                cell = self.sim.cells[idx]
                cell.set_power_dBm(self.original_cell_powers[idx])
                
    def select_arm(self) -> int:
        """
        Select an arm (combination of cells to turn off) based on EXP3 algorithm.
        
        Returns:
        --------
        int
            Index of selected arm
        """
        # During warm-up, select randomly
        if self.episode_count < self.warmup_episodes:
            return self.sim.rng.choice(self.n_arms)
        
        # Ensure minimum selection if enabled
        if self.ensure_min_selection:
            # Find arms that haven't been selected yet
            unselected_arms = np.where(self.arm_selection_count == 0)[0]
            if len(unselected_arms) > 0:
                return self.sim.rng.choice(unselected_arms)
        
        # Update probabilities
        self.update_probabilities()
        
        # Select arm based on probabilities
        return self.sim.rng.choice(self.n_arms, p=self.probabilities)
    
    def update_probabilities(self):
        """Update arm selection probabilities based on weights"""
        # EXP3 probability calculation
        sum_weights = np.sum(self.weights)
        exploitation_probs = self.weights / sum_weights
        
        # Mix exploitation with exploration
        self.probabilities = (1 - self.gamma) * exploitation_probs + self.gamma / self.n_arms
        
    def calculate_reward(self, efficiency: float) -> float:
        """
        Calculate normalized reward with improved dynamic range.
        
        Parameters:
        -----------
        efficiency : float
            Network efficiency in bits/J
            
        Returns:
        --------
        float
            Normalized reward in [0, 1]
        """
        # Handle zero efficiency
        if efficiency <= 0:
            return 0.0
        
        # 동적 정규화 사용 (더 넓은 보상 범위)
        if self.baseline_efficiency is not None and self.baseline_efficiency > 0:
            # 베이스라인 대비 상대적 성능
            relative_efficiency = efficiency / self.baseline_efficiency
            
            # 더 부드러운 보상 함수: sigmoid 형태
            if relative_efficiency >= 1.0:
                # 베이스라인보다 좋을 때
                improvement = relative_efficiency - 1.0
                reward = 0.5 + 0.4 * (1 - np.exp(-improvement * 3))  # [0.5, 0.9]
            else:
                # 베이스라인보다 나쁠 때
                degradation = 1.0 - relative_efficiency
                reward = 0.5 * np.exp(-degradation * 2)  # [0, 0.5]
        else:
            # 베이스라인이 없을 때 절대적 정규화
            # 과거 효율성 데이터 기반 정규화
            if len(self.efficiency_history) > 10:
                # 동적 범위 사용
                eff_range = max(self.max_efficiency - self.min_efficiency, 1000)
                reward = (efficiency - self.min_efficiency) / eff_range
                reward = np.clip(reward, 0, 1)
            else:
                # 고정 범위 사용 (초기)
                typical_efficiency = 1e4  # 10,000 bits/J
                reward = np.clip(efficiency / (typical_efficiency * 2), 0, 1)
            
        return float(reward)
    
    def update_weights(self, selected_arm: int, reward: float):
        """
        Update EXP3 weights based on observed reward.
        
        Parameters:
        -----------
        selected_arm : int
            Index of the selected arm
        reward : float
            Observed reward (should be in [0, 1])
        """
        # Estimated reward for the selected arm
        estimated_reward = reward / self.probabilities[selected_arm]
        
        # 적응적 학습률
        if len(self.reward_history) > 50:
            # 최근 보상들의 분산을 기반으로 학습률 조정
            recent_rewards = self.reward_history[-50:]
            reward_variance = np.var(recent_rewards)
            learning_scale = 1.0 + min(reward_variance * 10, 5.0)  # [1.0, 6.0]
        else:
            learning_scale = 1.0
        
        # Update weight for selected arm
        self.weights[selected_arm] *= np.exp(self.gamma * estimated_reward * learning_scale / self.n_arms)
        
        # Prevent numerical overflow
        max_weight = np.max(self.weights)
        if max_weight > 1e10:
            self.weights /= max_weight
            
    def save_learning_progress(self):
        """Save current learning progress to file"""
        
        if self.learning_log_file:
            # 수렴 에피소드 찾기
            convergence_episode = -1
            for i in range(50, self.episode_count, 10):
                if self.is_converged(window_size=50, threshold=0.01):
                    convergence_episode = i
                    break
            
            # 에너지 및 throughput 통계
            energy_stats = {}
            throughput_stats = {}
            
            # power_history 추가 (get_network_metrics()에서 얻은 실제 전력값들)
            power_history = []
            if hasattr(self, 'power_measurements'):  # 이 리스트를 만들어야 함
                power_history = self.power_measurements[-100:]
            
            # energy_statistics 계산 시 실제 전력값 사용
            if power_history and self.baseline_power > 0:
                avg_power = np.mean(power_history)
                energy_saving_all_on = (1 - avg_power / self.baseline_power) * 100
                
                energy_stats = {
                    'avg_energy_saving_all_on': energy_saving_all_on,
                    'avg_energy_saving_random': (1 - avg_power / (self.baseline_power * 0.75)) * 100,
                    'std_energy_saving': np.std([(1 - p / self.baseline_power) * 100 for p in power_history]),
                    'max_energy_saving': max([(1 - p / self.baseline_power) * 100 for p in power_history]),
                    'current_power_kw': power_history[-1] if power_history else 0
                }
            else:
                energy_stats = {}
            
            if hasattr(self, 'throughput_measurements') and self.throughput_measurements:
                recent_throughput = self.throughput_measurements[-100:]
                throughput_stats = {
                    'avg_throughput_mbps': np.mean(recent_throughput),
                    'std_throughput_mbps': np.std(recent_throughput),
                    'min_throughput_mbps': np.min(recent_throughput),
                    'max_throughput_mbps': np.max(recent_throughput),
                    'final_throughput_mbps': recent_throughput[-1] if recent_throughput else 0
                }
                
            progress_data = {
                # ... 기존 필드들 ...
                'episode': int(self.episode_count),
                'timestamp': float(self.sim.env.now),
                'weights': self.weights.tolist(),
                'probabilities': self.probabilities.tolist(),
                'arm_selection_count': [int(x) for x in self.arm_selection_count],
                'cumulative_rewards': self.cumulative_rewards.tolist(),
                'reward_history': self.reward_history,
                'selected_arm_history': [int(x) for x in self.selected_arm_history],
                
                # 추가 메트릭
                'energy_statistics': energy_stats,
                'throughput_statistics': throughput_stats,
                'switching_count': self.switching_count,
                'switching_rate': self.switching_count / self.episode_count if self.episode_count > 0 else 0,
                'convergence_episode': convergence_episode,
                'is_converged': self.is_converged(),
                
                # Regret 정보
                'regret_statistics': self.get_regret_statistics(),
                'cumulative_regret_history': self.cumulative_regret_history[-100:],
                'instant_regret_history': self.instant_regret_history[-100:],
                
                # throughput 추가
                'throughput_measurements': self.throughput_measurements[-100:] if hasattr(self, 'throughput_measurements') else [],
                'throughput_statistics': throughput_stats,
                
                # 기존 메트릭
                'efficiency_history': self.efficiency_history[-100:],
                'min_efficiency': float(self.min_efficiency) if self.min_efficiency != float('inf') else None,
                'max_efficiency': float(self.max_efficiency),
                'baseline_efficiency': float(self.baseline_efficiency) if self.baseline_efficiency else None,
                'baseline_throughput': float(self.baseline_throughput) if self.baseline_throughput else None,
                'baseline_power': float(self.baseline_power) if self.baseline_power else None
            }
            
            # 파일 저장
            try:
                os.makedirs(os.path.dirname(self.learning_log_file), exist_ok=True)
                with open(self.learning_log_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to save progress: {e}") 
                
    def save_final_model(self):
        """확장된 최종 모델 저장"""
        if self.final_model_file:
            # 최종 통계 계산
            final_energy_stats = {}
            final_throughput_stats = {}
            
            if self.energy_consumption_history:
                all_energy_savings = [e['energy_saving_all_on'] for e in self.energy_consumption_history]
                final_energy_stats = {
                    'mean_energy_saving': np.mean(all_energy_savings),
                    'std_energy_saving': np.std(all_energy_savings),
                    'max_energy_saving': np.max(all_energy_savings),
                    'min_energy_saving': np.min(all_energy_savings),
                    'final_energy_saving': all_energy_savings[-1] if all_energy_savings else 0
                }
            
            if self.throughput_history:
                all_avg_throughputs = [t['avg_cell_throughput_gbps'] for t in self.throughput_history]
                final_throughput_stats = {
                    'mean_avg_cell_throughput': np.mean(all_avg_throughputs),
                    'std_avg_cell_throughput': np.std(all_avg_throughputs),
                    'max_avg_cell_throughput': np.max(all_avg_throughputs),
                    'min_avg_cell_throughput': np.min(all_avg_throughputs),
                    'final_avg_cell_throughput': all_avg_throughputs[-1] if all_avg_throughputs else 0
                }
            
            model_data = {
                # ... 기존 필드들 ...
                'k_cells': self.k_cells,
                'n_cells_off': self.n_cells_off,
                'gamma': self.gamma,
                'arms': [list(arm) for arm in self.arms],
                'weights': self.weights.tolist(),
                'probabilities': self.probabilities.tolist(),
                'arm_selection_count': self.arm_selection_count.tolist(),
                'cumulative_rewards': self.cumulative_rewards.tolist(),
                'total_episodes': self.episode_count,
                
                # 추가된 최종 통계
                'final_energy_statistics': final_energy_stats,
                'final_throughput_statistics': final_throughput_stats,
                'total_switching_count': self.switching_count,
                'final_switching_rate': self.switching_count / self.episode_count if self.episode_count > 0 else 0,
                'convergence_achieved': self.is_converged(),
                'convergence_episode': self._find_convergence_episode(),
                
                # Regret 정보
                'final_regret_statistics': self.get_regret_statistics(),
                'final_cumulative_regret': self.cumulative_regret,
                'final_average_regret': self.cumulative_regret / self.episode_count if self.episode_count > 0 else 0,
                
                # 기존 메트릭
                'baseline_efficiency': self.baseline_efficiency,
                'min_efficiency': float(self.min_efficiency) if self.min_efficiency != float('inf') else None,
                'max_efficiency': float(self.max_efficiency),
                'training_completed': datetime.now().isoformat()
            }
            
            # 파일 저장
            try:
                os.makedirs(os.path.dirname(self.final_model_file), exist_ok=True)
                with open(self.final_model_file, 'w') as f:
                    json.dump(model_data, f, indent=2)
                print(f"Final model saved to: {self.final_model_file}")
            except Exception as e:
                print(f"Error saving final model: {e}")
   
   
    def _find_convergence_episode(self) -> int:
        """수렴 에피소드 찾기"""
        window_size = 50
        threshold = 0.01
        
        for i in range(window_size, len(self.probability_history), 10):
            recent_probs = self.probability_history[i-window_size:i]
            prob_changes = []
            
            for j in range(1, len(recent_probs)):
                change = np.mean(np.abs(recent_probs[j] - recent_probs[j-1]))
                prob_changes.append(change)
            
            if np.mean(prob_changes) < threshold:
                return i * 10  # probability_history는 10 에피소드마다 저장
        
        return -1      
     
     
    def plot_learning_progress(self, save_path: str = None):
        """Generate plots showing learning progress"""
        if len(self.weight_history) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Weight evolution for top 10 arms
        ax = axes[0, 0]
        weight_array = np.array(self.weight_history)
        if weight_array.shape[0] > 0:
            # Show top 10 arms by final weight
            final_weights = self.weights
            top_arms = np.argsort(final_weights)[-10:]
            
            for arm_idx in top_arms:
                weights_over_time = weight_array[:, arm_idx]
                ax.plot(weights_over_time, label=f'Arm {arm_idx}')
            
            ax.set_xlabel('Learning Steps (×10 episodes)')
            ax.set_ylabel('Weight')
            ax.set_title('Weight Evolution (Top 10 Arms)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_yscale('log')
        
        # Plot 2: Reward history
        ax = axes[0, 1]
        if len(self.reward_history) > 0:
            ax.plot(self.reward_history)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Reward History')
            
            # Moving average
            if len(self.reward_history) > 10:
                window = min(20, len(self.reward_history) // 5)
                moving_avg = np.convolve(self.reward_history, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.reward_history)), moving_avg, 'r-', linewidth=2, label='Moving Average')
                ax.legend()
        
        # Plot 3: Efficiency evolution
        ax = axes[1, 0]
        if len(self.efficiency_history) > 0:
            ax.plot(self.efficiency_history)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Efficiency (bits/J)')
            ax.set_title('Network Efficiency Over Time')
            
            if self.baseline_efficiency:
                ax.axhline(y=self.baseline_efficiency, color='r', linestyle='--', 
                          label=f'Baseline: {self.baseline_efficiency:.0f}')
                ax.legend()
        
        # Plot 4: Arm selection frequency
        ax = axes[1, 1]
        if np.sum(self.arm_selection_count) > 0:
            sorted_counts = np.sort(self.arm_selection_count)[::-1][:20]  # Top 20
            ax.bar(range(len(sorted_counts)), sorted_counts)
            ax.set_xlabel('Arm Rank')
            ax.set_ylabel('Selection Count')
            ax.set_title('Arm Selection Frequency (Top 20)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning progress plot saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def loop(self):
        """Main loop of the EXP3 cell on/off scenario"""
        # Wait for initial delay
        if self.delay > 0:
            yield self.sim.env.timeout(self.delay)
            
        print(f"\n{'='*60}")
        print(f"EXP3CellOnOff scenario started at t={self.sim.env.now}")
        print(f"Total cells: {self.k_cells}, Turning off: {self.n_cells_off} cells")
        print(f"Total arms (combinations): {self.n_arms}")
        print(f"{'='*60}\n")
        
        # Get baseline performance (all cells on)
        print("Measuring baseline performance...")
        yield self.sim.env.timeout(self.interval)
        baseline_tp, baseline_pwr, baseline_eff = self.get_network_metrics()
        self.baseline_throughput = baseline_tp
        self.baseline_power = baseline_pwr
        self.baseline_efficiency = baseline_eff

        # baseline_energy_consumption도 함께 설정 (중요!)
        self.baseline_energy_consumption = baseline_pwr
        
        # Validate baseline
        if baseline_pwr <= 1.0 or np.isnan(baseline_pwr):
            print(f"Warning: Suspicious baseline power ({baseline_pwr:.2f} kW). Adjusting...")
            # 더 현실적인 베이스라인 추정
            estimated_power = len(self.sim.cells) * 2.0  # 셀당 2kW
            self.baseline_power = estimated_power
            if baseline_tp > 0:
                self.baseline_efficiency = (baseline_tp * 1e6) / (estimated_power * 1e3)
            else:
                self.baseline_efficiency = 1e4  # 기본값
        
        print(f"Baseline: Throughput={self.baseline_throughput:.2f} Mbps, Power={self.baseline_power:.2f} kW, Efficiency={self.baseline_efficiency:.2e} bits/J")
        print(f"{'='*60}\n")
        
        # Main learning loop
        while True:
            # Select arm (combination of cells to turn off)
            selected_arm = self.select_arm()
            cells_to_turn_off = self.arms[selected_arm]
            
            # Progress indicator every episode
            if self.episode_count < self.warmup_episodes:
                phase = "WARM-UP"
            else:
                phase = "LEARNING"
            
            print(f"\n[Episode {self.episode_count + 1}] {phase} - t={self.sim.env.now:.1f}s")
            print(f"  Selected arm: {selected_arm} -> Turning off cells: {list(cells_to_turn_off)}")
            
            # Turn off selected cells
            self.turn_cells_off(cells_to_turn_off)
            
            # Wait for system to stabilize and handovers to complete
            yield self.sim.env.timeout(self.interval * 2)
            
            # 추가 메트릭 계산
            energy_metrics = self.calculate_energy_metrics()
            throughput_metrics = self.calculate_throughput_metrics()
            
            # 이력에 추가
            self.energy_consumption_history.append(energy_metrics)
            self.throughput_history.append(throughput_metrics)
            
            # 전환 비용 업데이트
            self.update_switching_cost(selected_arm)
            
            # Measure network performance
            throughput, power, efficiency = self.get_network_metrics()
            
            if not hasattr(self, 'throughput_measurements'):
                self.throughput_measurements = []
            self.throughput_measurements.append(throughput)  # Mbps 단위
            
            if not hasattr(self, 'power_measurements'):
                self.power_measurements = []
            self.power_measurements.append(power)
            
            # Calculate reward
            reward = self.calculate_reward(efficiency)
            
            energy_metrics = self.calculate_energy_metrics()

            # calculate_throughput_metrics() 대신 측정된 값 직접 사용
            throughput_metrics = {
                'total_throughput_gbps': throughput / 1000.0,  # Mbps to Gbps
                'avg_cell_throughput_gbps': throughput / 1000.0 / (self.k_cells - len(cells_to_turn_off)),
                'std_cell_throughput': 0,  # 단일 측정값이므로 0
                'throughput_per_active_cell': throughput / (self.k_cells - len(cells_to_turn_off))  # Mbps per cell
            }

            # 이력에 추가
            self.energy_consumption_history.append(energy_metrics)
            self.throughput_history.append(throughput_metrics)
            
            # Update statistics
            self.episode_count += 1
            self.arm_selection_count[selected_arm] += 1
            self.cumulative_rewards[selected_arm] += reward
            self.reward_history.append(reward)
            self.selected_arm_history.append(selected_arm)
            self.update_regret_tracking(selected_arm, reward)

            # Update EXP3 weights (only after warm-up)
            if self.episode_count > self.warmup_episodes:
                self.update_weights(selected_arm, reward)
                
            # Display current results
            print(f"  Network state: TP={throughput:.2f} Mbps, Power={power:.2f} kW, Eff={efficiency:.2e} bits/J")
            print(f"  Reward: {reward:.4f} (Relative Eff: {efficiency/self.baseline_efficiency:.3f})")
            
            # Show top arms every 10 episodes
            if self.episode_count % 10 == 0:
                print(f"\n  Top 3 arms by weight:")
                top_arms = np.argsort(self.weights)[-3:][::-1]
                for rank, arm_idx in enumerate(top_arms):
                    print(f"    {rank+1}. Arm {arm_idx}: cells {list(self.arms[arm_idx])}, weight={self.weights[arm_idx]:.4f}")
                
            # Store history for visualization
            if self.episode_count % 10 == 0:  # Store every 10 episodes to limit memory
                self.weight_history.append(self.weights.copy())
                self.probability_history.append(self.probabilities.copy())
            
            # Regret 정보 출력 (매 10 에피소드)
            if self.episode_count % 10 == 0:
                regret_stats = self.get_regret_statistics()
                print(f"\n  Regret Statistics:")
                print(f"    Cumulative regret: {regret_stats['cumulative_regret']:.4f}")
                print(f"    Average regret: {regret_stats['average_regret']:.4f}")
                print(f"    Theoretical bound: {regret_stats['theoretical_bound']:.4f}")
                print(f"    Regret ratio: {regret_stats['regret_ratio']:.2%}")    
            
            # Log progress and save
            if self.episode_count % 50 == 0:
                avg_reward = np.mean(self.reward_history[-50:]) if len(self.reward_history) >= 50 else np.mean(self.reward_history)
                avg_efficiency = np.mean(self.efficiency_history[-50:]) if len(self.efficiency_history) >= 50 else np.mean(self.efficiency_history)
                print(f"\n{'='*60}")
                print(f"Progress Summary - Episode {self.episode_count}")
                print(f"Average Reward (last 50): {avg_reward:.4f}")
                print(f"Average Efficiency (last 50): {avg_efficiency:.2e} bits/J")
                print(f"Best arm so far: {np.argmax(self.weights)} -> cells {list(self.arms[np.argmax(self.weights)])}")
                print(f"Efficiency range: [{self.min_efficiency:.2e}, {self.max_efficiency:.2e}]")
                print(f"{'='*60}\n")
                
                try:
                    self.save_learning_progress()
                    print("Progress saved successfully.")
                except Exception as e:
                    print(f"Warning: Failed to save progress: {e}")
                
            # Turn cells back on
            self.turn_cells_on(cells_to_turn_off)
            
            # Wait before next decision
            yield self.sim.env.timeout(self.interval)
            
            # Check if simulation is ending
            if self.sim.env.now >= self.sim.until - self.interval * 3:
                # Save final model and generate plots
                print(f"\n{'='*60}")
                print(f"EXP3CellOnOff learning completed after {self.episode_count} episodes")
                print(f"Final best arm: {np.argmax(self.weights)} -> cells {list(self.arms[np.argmax(self.weights)])}")
                print(f"Final efficiency range: [{self.min_efficiency:.2e}, {self.max_efficiency:.2e}]")
                print(f"{'='*60}\n")
                
                try:
                    self.save_final_model()
                    plot_path = self.learning_log_file.replace('.json', '_plot.png') if self.learning_log_file else None
                    self.plot_learning_progress(plot_path)
                except Exception as e:
                    print(f"Warning: Failed to save final results: {e}")
                    
                break

    def calculate_instant_regret(self, selected_arm: int, reward: float):
        """
        순간 regret 계산
        
        Parameters:
        -----------
        selected_arm : int
            선택된 arm
        reward : float
            획득한 보상
            
        Returns:
        --------
        float
            순간 regret
        """
        # 현재까지의 각 arm의 평균 보상 계산
        avg_rewards = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.arm_selection_count[i] > 0:
                avg_rewards[i] = self.cumulative_rewards[i] / self.arm_selection_count[i]
        
        # 최고 평균 보상 (oracle)
        best_avg_reward = np.max(avg_rewards) if np.any(avg_rewards > 0) else reward
        
        # 순간 regret = 최적 보상 - 실제 보상
        instant_regret = best_avg_reward - reward
        
        return instant_regret
    
    def update_regret_tracking(self, selected_arm: int, reward: float):
        """
        Regret 관련 통계 업데이트
        
        Parameters:
        -----------
        selected_arm : int
            선택된 arm
        reward : float
            획득한 보상
        """
        # 순간 regret 계산
        instant_regret = self.calculate_instant_regret(selected_arm, reward)
        self.instant_regret_history.append(instant_regret)
        
        # 누적 regret 업데이트
        self.cumulative_regret += instant_regret
        self.cumulative_regret_history.append(self.cumulative_regret)
        
        # 실제 누적 보상 업데이트
        self.actual_cumulative_reward += reward
        
        # Oracle 누적 보상 업데이트 (사후 분석용)
        if self.episode_count > self.warmup_episodes:
            avg_rewards = np.zeros(self.n_arms)
            for i in range(self.n_arms):
                if self.arm_selection_count[i] > 0:
                    avg_rewards[i] = self.cumulative_rewards[i] / self.arm_selection_count[i]
            
            if np.any(avg_rewards > 0):
                best_avg_reward = np.max(avg_rewards)
                self.oracle_cumulative_reward += best_avg_reward
    
    def calculate_theoretical_regret_bound(self):
        """
        EXP3의 이론적 regret bound 계산
        
        Returns:
        --------
        float
            이론적 regret 상한
        """
        T = self.episode_count
        K = self.n_arms
        
        # EXP3의 이론적 regret bound: O(√(TK log K))
        theoretical_bound = 2 * np.sqrt(T * K * np.log(K))
        
        return theoretical_bound
    
    def get_regret_statistics(self):
        """
        Regret 관련 통계 반환
        
        Returns:
        --------
        dict
            Regret 통계
        """
        if self.episode_count == 0:
            return {}
            
        # 최종 arm별 평균 보상
        avg_rewards = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.arm_selection_count[i] > 0:
                avg_rewards[i] = self.cumulative_rewards[i] / self.arm_selection_count[i]
        
        best_arm_idx = np.argmax(avg_rewards)
        best_arm_avg_reward = avg_rewards[best_arm_idx]
        
        # 실제 평균 보상
        actual_avg_reward = self.actual_cumulative_reward / self.episode_count
        
        # 이론적 bound
        theoretical_bound = self.calculate_theoretical_regret_bound()
        
        return {
            'cumulative_regret': self.cumulative_regret,
            'average_regret': self.cumulative_regret / self.episode_count,
            'theoretical_bound': theoretical_bound,
            'regret_ratio': self.cumulative_regret / theoretical_bound if theoretical_bound > 0 else 0,
            'best_arm_idx': int(best_arm_idx),
            'best_arm_avg_reward': float(best_arm_avg_reward),
            'actual_avg_reward': float(actual_avg_reward),
            'regret_per_episode': self.cumulative_regret / self.episode_count
        }
        
    def plot_regret_analysis(self, save_path=None):
        """
        Regret 분석 플롯 생성
        
        Parameters:
        -----------
        save_path : str
            저장 경로 (None이면 화면에 표시)
        """
        if len(self.cumulative_regret_history) == 0:
            print("No regret data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('EXP3 Regret Analysis', fontsize=16)
        
        # 누적 regret
        ax = axes[0, 0]
        episodes = range(1, len(self.cumulative_regret_history) + 1)
        ax.plot(episodes, self.cumulative_regret_history, 'b-', label='Actual')
        
        # 이론적 bound 추가
        theoretical_bounds = [2 * np.sqrt(t * self.n_arms * np.log(self.n_arms)) 
                            for t in episodes]
        ax.plot(episodes, theoretical_bounds, 'r--', label='Theoretical Bound')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title('Cumulative Regret vs Theoretical Bound')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 순간 regret
        ax = axes[0, 1]
        ax.plot(self.instant_regret_history)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Instant Regret')
        ax.set_title('Instant Regret per Episode')
        ax.grid(True, alpha=0.3)
        
        # 평균 regret
        ax = axes[1, 0]
        avg_regret = [self.cumulative_regret_history[i] / (i + 1) 
                     for i in range(len(self.cumulative_regret_history))]
        ax.plot(avg_regret)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Regret')
        ax.set_title('Average Regret over Time')
        ax.grid(True, alpha=0.3)
        
        # Regret 비율
        ax = axes[1, 1]
        regret_ratios = [self.cumulative_regret_history[i] / theoretical_bounds[i] 
                        if theoretical_bounds[i] > 0 else 0 
                        for i in range(len(self.cumulative_regret_history))]
        ax.plot(regret_ratios)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Regret Ratio')
        ax.set_title('Actual / Theoretical Regret')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Regret analysis plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()    
