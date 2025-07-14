#!/usr/bin/env python3
"""
EXP3 Multi-seed Evaluation Script
온라인 러닝 평가를 위한 다중 시드 실험 관리 스크립트

python exp3_multi_seed_evaluation.py -c data/input/exp3_cell_on_off/exp3_training_fixed.json



"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Tuple
import scipy.stats as stats
import pandas as pd

class EXP3MultiSeedEvaluator:
    def __init__(self, base_config_path: str, n_seeds: int = 10, output_base_dir: str = "_/data/output"):
        """
        Parameters:
        -----------
        base_config_path : str
            기본 설정 파일 경로
        n_seeds : int
            실험할 시드 개수
        output_base_dir : str
            결과 저장 기본 디렉토리
        """
        self.base_config_path = base_config_path
        self.n_seeds = n_seeds
        self.output_base_dir = output_base_dir
        
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")   
        time_str = now.strftime("%H%M%S") 
        date_dir = os.path.join(output_base_dir, f"exp3_multi_seed_{date_str}")
        os.makedirs(date_dir, exist_ok=True)
        self.experiment_dir = os.path.join(date_dir, time_str)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 결과 저장용 변수
        self.all_results = []
        self.seed_results = {}
        
    def run_single_seed(self, seed: int) -> Dict:
        """단일 시드로 실험 실행"""
        print(f"\n{'='*60}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*60}")
        
        # 설정 파일 로드 및 수정
        with open(self.base_config_path, 'r') as f:
            config = json.load(f)
        
        # 시드 및 출력 경로 수정
        config['seed'] = seed
        config['exp3_learning_log'] = f"exp3_learning_seed_{seed}.json"
        config['exp3_final_model'] = f"exp3_model_seed_{seed}.json"
        
        # 임시 설정 파일 생성
        temp_config_path = os.path.join(self.experiment_dir, f"config_seed_{seed}.json")
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 시뮬레이션 실행
        start_time = time.time()
        process = subprocess.run(
            ["python", "run_kiss.py", "-c", temp_config_path],
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            print(f"Error running seed {seed}: {process.stderr}")
            return None
            
        execution_time = time.time() - start_time
        print(f"Seed {seed} completed in {execution_time:.2f} seconds")
        
        # 결과 파일 찾기 및 로드
        result_dir = self._find_latest_output_dir(config['experiment_description'])
        if not result_dir:
            print(f"Warning: Could not find output directory for seed {seed}")
            return None
            
        # 학습 로그 로드
        learning_log_path = os.path.join(result_dir, config['exp3_learning_log'])
        model_path = os.path.join(result_dir, config['exp3_final_model'])
        
        if not os.path.exists(learning_log_path):
            print(f"Warning: Learning log not found for seed {seed}")
            return None
            
        with open(learning_log_path, 'r') as f:
            learning_data = json.load(f)
            
        with open(model_path, 'r') as f:
            model_data = json.load(f)
            
        return {
            'seed': seed,
            'learning_data': learning_data,
            'model_data': model_data,
            'execution_time': execution_time,
            'result_dir': result_dir
        }
    
    def _find_latest_output_dir(self, experiment_description: str) -> str:
        """최신 출력 디렉토리 찾기"""
        base_path = Path(self.output_base_dir) / experiment_description
        if not base_path.exists():
            return None
            
        # 날짜 폴더들 찾기
        date_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        if not date_dirs:
            return None
            
        # 가장 최근 날짜 폴더 선택
        latest_date = sorted(date_dirs)[-1]
        
        # 시간 폴더가 있는지 확인
        time_dirs = [d for d in latest_date.iterdir() if d.is_dir()]
        if time_dirs:
            return str(sorted(time_dirs)[-1])
        else:
            return str(latest_date)
    
    def run_all_seeds(self):
        """모든 시드로 실험 실행"""
        print(f"\n{'='*80}")
        print(f"Starting multi-seed evaluation with {self.n_seeds} seeds")
        print(f"Output directory: {self.experiment_dir}")
        print(f"{'='*80}")
        
        for seed in range(1, self.n_seeds + 1):
            result = self.run_single_seed(seed)
            if result:
                self.all_results.append(result)
                self.seed_results[seed] = result
        
        print(f"\n{'='*80}")
        print(f"Completed {len(self.all_results)}/{self.n_seeds} experiments successfully")
        print(f"{'='*80}")
    
    def calculate_metrics(self):
        """평가 지표 계산"""
        if not self.all_results:
            print("No results to analyze")
            return None
            
        metrics = {
            'cumulative_rewards': [],
            'cumulative_regrets': [],
            'average_regrets': [],
            'final_efficiencies': [],
            'convergence_episodes': [],
            'energy_savings': [],
            'average_throughputs': [],
            'switching_costs': [],
            'best_arm_selections': []
        }
        
        for result in self.all_results:
            learning_data = result['learning_data']
            model_data = result['model_data']
            
            # 누적 보상
            rewards = learning_data.get('reward_history', [])
            cumulative_reward = np.cumsum(rewards)
            metrics['cumulative_rewards'].append(cumulative_reward)
            
            # 누적 후회
            regret_stats = learning_data.get('regret_statistics', {})
            cumulative_regret = learning_data.get('cumulative_regret', 0)
            average_regret = learning_data.get('average_regret', 0)
            
            metrics['cumulative_regrets'].append(cumulative_regret)
            metrics['average_regrets'].append(average_regret)
            
            # 최종 효율성
            final_efficiency = learning_data.get('max_efficiency', 0)
            metrics['final_efficiencies'].append(final_efficiency)
            
            # 수렴 에피소드 (안정화 시점)
            convergence_episode = self._find_convergence_episode(learning_data)
            metrics['convergence_episodes'].append(convergence_episode)
            
            # 에너지 절감율
            baseline_power = learning_data.get('baseline_power', 0)
            efficiency_history = learning_data.get('efficiency_history', [])
            if baseline_power > 0 and efficiency_history:
                # 평균 전력 소비 추정 (효율성 기반)
                avg_efficiency = np.mean(efficiency_history[-100:])
                energy_saving = self._calculate_energy_saving(
                    baseline_power, avg_efficiency, learning_data
                )
                metrics['energy_savings'].append(energy_saving)
            
            # 평균 throughput
            avg_throughput = self._calculate_average_throughput(learning_data)
            metrics['average_throughputs'].append(avg_throughput)
            
            # 전환 비용
            switching_cost = self._calculate_switching_cost(learning_data)
            metrics['switching_costs'].append(switching_cost)
            
            # 최적 arm 선택
            weights = np.array(model_data.get('weights', []))
            best_arm = np.argmax(weights) if len(weights) > 0 else -1
            metrics['best_arm_selections'].append(best_arm)
        
        return metrics
    
    def _find_convergence_episode(self, learning_data: Dict) -> int:
        """가중치 안정화 시점 찾기"""
        # 확률 변화량이 임계값 이하로 떨어지는 시점
        probability_history = learning_data.get('probability_history', [])
        if len(probability_history) < 2:
            return -1
            
        threshold = 0.01  # 1% 변화 임계값
        window_size = 10  # 10 에피소드 평균
        
        for i in range(window_size, len(probability_history)):
            recent_probs = probability_history[i-window_size:i]
            prob_changes = []
            
            for j in range(1, len(recent_probs)):
                change = np.mean(np.abs(recent_probs[j] - recent_probs[j-1]))
                prob_changes.append(change)
            
            if np.mean(prob_changes) < threshold:
                return i * 10  # probability_history는 10 에피소드마다 저장됨
        
        return -1
    
    def _calculate_energy_saving(self, baseline_power: float, avg_efficiency: float, 
                                learning_data: Dict) -> float:
        """에너지 절감율 계산"""
        # 베이스라인 대비 실제 소비 에너지 비율
        # 효율성이 높을수록 같은 throughput에 더 적은 에너지 사용
        baseline_efficiency = learning_data.get('baseline_efficiency', avg_efficiency)
        
        if baseline_efficiency > 0:
            # 에너지 절감율 = 1 - (baseline_efficiency / current_efficiency)
            # 효율성이 높아질수록 절감율이 증가
            energy_ratio = baseline_efficiency / avg_efficiency
            energy_saving = (1 - energy_ratio) * 100  # 퍼센트
            return max(0, energy_saving)  # 음수 방지
        
        return 0
    
    def _calculate_average_throughput(self, learning_data: Dict) -> float:
        """평균 셀 throughput 계산"""
        # 효율성과 전력에서 throughput 추정
        efficiency_history = learning_data.get('efficiency_history', [])
        baseline_power = learning_data.get('baseline_power', 75.72)  # kW
        
        if efficiency_history:
            avg_efficiency = np.mean(efficiency_history[-100:])
            # throughput = efficiency * power (간단한 추정)
            avg_throughput = avg_efficiency * baseline_power * 0.75  # 75% 활성 셀 가정
            return avg_throughput / 1e9  # Gbps 단위로 변환
        
        return 0
    
    def _calculate_switching_cost(self, learning_data: Dict) -> float:
        """전환 비용 계산 (arm 변경 빈도)"""
        selected_arms = learning_data.get('selected_arm_history', [])
        if len(selected_arms) < 2:
            return 0
            
        switches = 0
        for i in range(1, len(selected_arms)):
            if selected_arms[i] != selected_arms[i-1]:
                switches += 1
        
        return switches / len(selected_arms) * 100  # 퍼센트
    
    def generate_plots(self, metrics: Dict):
        """평가 결과 플롯 생성"""
        # 1. 누적 보상 플롯
        self._plot_cumulative_metric(
            metrics['cumulative_rewards'],
            'Cumulative Reward',
            'cumulative_reward.png'
        )
        
        # 2. 누적 후회 플롯
        self._plot_cumulative_regret(metrics)
        
        # 3. 선택 확률 분포
        self._plot_probability_distribution(metrics)
        
        # 4. 성능 지표 박스플롯
        self._plot_performance_boxplots(metrics)
        
        # 5. 수렴 분석
        self._plot_convergence_analysis(metrics)
        
        # 6. 에너지 절감 및 throughput
        self._plot_energy_throughput(metrics)
    
    def _plot_cumulative_metric(self, data_list: List, ylabel: str, filename: str):
        """누적 지표 플롯 (평균 ± 표준편차)"""
        if not data_list:
            return
            
        plt.figure(figsize=(10, 6))
        
        # 모든 데이터의 길이를 맞추기
        max_len = max(len(d) for d in data_list)
        aligned_data = []
        
        for d in data_list:
            if len(d) < max_len:
                # 마지막 값으로 패딩
                padded = np.pad(d, (0, max_len - len(d)), 'edge')
                aligned_data.append(padded)
            else:
                aligned_data.append(d[:max_len])
        
        # 평균과 표준편차 계산
        data_array = np.array(aligned_data)
        mean_values = np.mean(data_array, axis=0)
        std_values = np.std(data_array, axis=0)
        
        episodes = np.arange(1, max_len + 1)
        
        # 평균 곡선
        plt.plot(episodes, mean_values, 'b-', linewidth=2, label='Mean')
        
        # 표준편차 영역
        plt.fill_between(episodes, 
                        mean_values - std_values, 
                        mean_values + std_values,
                        alpha=0.3, color='blue', label='±1 std')
        
        # 95% 신뢰구간
        confidence_interval = 1.96 * std_values / np.sqrt(len(data_list))
        plt.fill_between(episodes,
                        mean_values - confidence_interval,
                        mean_values + confidence_interval,
                        alpha=0.5, color='lightblue', label='95% CI')
        
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} over Episodes (n={len(data_list)} seeds)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.experiment_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")
    
    def _plot_cumulative_regret(self, metrics: Dict):
        """누적 후회 플롯"""
        plt.figure(figsize=(12, 8))
        
        # 각 시드별 누적 후회 계산
        cumulative_regrets = []
        for result in self.all_results:
            learning_data = result['learning_data']
            regret_history = learning_data.get('cumulative_regret_history', [])
            if regret_history:
                cumulative_regrets.append(regret_history)
        
        if cumulative_regrets:
            # 평균 및 신뢰구간 계산
            max_len = max(len(r) for r in cumulative_regrets)
            aligned_regrets = []
            
            for r in cumulative_regrets:
                if len(r) < max_len:
                    padded = np.pad(r, (0, max_len - len(r)), 'edge')
                    aligned_regrets.append(padded)
                else:
                    aligned_regrets.append(r[:max_len])
            
            regret_array = np.array(aligned_regrets)
            mean_regret = np.mean(regret_array, axis=0)
            std_regret = np.std(regret_array, axis=0)
            
            episodes = np.arange(1, max_len + 1)
            
            # 평균 누적 후회
            plt.plot(episodes, mean_regret, 'r-', linewidth=2, label='Mean Cumulative Regret')
            
            # 표준편차
            plt.fill_between(episodes,
                           mean_regret - std_regret,
                           mean_regret + std_regret,
                           alpha=0.3, color='red')
            
            # 이론적 상한
            n_arms = 969  # C(19,3)
            theoretical_bound = 2 * np.sqrt(episodes * n_arms * np.log(n_arms))
            plt.plot(episodes, theoretical_bound, 'k--', label='Theoretical Bound')
            
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Regret')
            plt.title('Cumulative Regret Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.experiment_dir, 'cumulative_regret.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")
    
    def _plot_probability_distribution(self, metrics: Dict):
        """선택 확률 분포 히트맵"""
        plt.figure(figsize=(14, 8))
        
        # 최종 확률 분포들 수집
        final_probabilities = []
        for result in self.all_results:
            model_data = result['model_data']
            probs = model_data.get('probabilities', [])
            if probs:
                final_probabilities.append(probs)
        
        if final_probabilities:
            # 상위 20개 arm의 평균 확률
            avg_probs = np.mean(final_probabilities, axis=0)
            top_20_indices = np.argsort(avg_probs)[-20:][::-1]
            
            # 히트맵 데이터 준비
            heatmap_data = []
            for i, result in enumerate(self.all_results):
                probs = result['model_data'].get('probabilities', [])
                if probs:
                    heatmap_data.append([probs[idx] for idx in top_20_indices])
            
            if heatmap_data:
                plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
                plt.colorbar(label='Selection Probability')
                plt.xlabel('Top 20 Arms')
                plt.ylabel('Seed')
                plt.title('Selection Probability Distribution across Seeds')
                
                # x축 레이블
                arms_info = []
                for idx in top_20_indices:
                    if idx < len(self.all_results[0]['model_data']['arms']):
                        arm = self.all_results[0]['model_data']['arms'][idx]
                        arms_info.append(f"{idx}\n{arm}")
                plt.xticks(range(20), arms_info, rotation=90, fontsize=8)
        
        save_path = os.path.join(self.experiment_dir, 'probability_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")
    
    def _plot_performance_boxplots(self, metrics: Dict):
        """성능 지표 박스플롯"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 평균 후회
        ax = axes[0, 0]
        if metrics['average_regrets']:
            ax.boxplot(metrics['average_regrets'])
            ax.set_ylabel('Average Regret')
            ax.set_title('Average Regret Distribution')
            ax.grid(True, alpha=0.3)
        
        # 2. 최종 효율성
        ax = axes[0, 1]
        if metrics['final_efficiencies']:
            ax.boxplot(metrics['final_efficiencies'])
            ax.set_ylabel('Final Efficiency (bits/J)')
            ax.set_title('Final Efficiency Distribution')
            ax.grid(True, alpha=0.3)
        
        # 3. 에너지 절감율
        ax = axes[1, 0]
        if metrics['energy_savings']:
            ax.boxplot(metrics['energy_savings'])
            ax.set_ylabel('Energy Saving (%)')
            ax.set_title('Energy Saving Distribution')
            ax.grid(True, alpha=0.3)
        
        # 4. 전환 비용
        ax = axes[1, 1]
        if metrics['switching_costs']:
            ax.boxplot(metrics['switching_costs'])
            ax.set_ylabel('Switching Cost (%)')
            ax.set_title('Switching Cost Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.experiment_dir, 'performance_boxplots.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")
    
    def _plot_convergence_analysis(self, metrics: Dict):
        """수렴 분석 플롯"""
        plt.figure(figsize=(10, 6))
        
        convergence_episodes = [e for e in metrics['convergence_episodes'] if e > 0]
        
        if convergence_episodes:
            plt.hist(convergence_episodes, bins=20, alpha=0.7, color='green', edgecolor='black')
            
            # 통계 정보 추가
            mean_conv = np.mean(convergence_episodes)
            std_conv = np.std(convergence_episodes)
            
            plt.axvline(mean_conv, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_conv:.0f} episodes')
            plt.axvline(mean_conv - std_conv, color='orange', linestyle=':', linewidth=1)
            plt.axvline(mean_conv + std_conv, color='orange', linestyle=':', linewidth=1,
                       label=f'±1 std: {std_conv:.0f}')
            
            plt.xlabel('Convergence Episode')
            plt.ylabel('Frequency')
            plt.title('Weight Distribution Convergence Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.experiment_dir, 'convergence_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")
    
    def _plot_energy_throughput(self, metrics: Dict):
        """에너지 및 throughput 분석"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 에너지 절감율 vs Throughput
        if metrics['energy_savings'] and metrics['average_throughputs']:
            ax1.scatter(metrics['energy_savings'], metrics['average_throughputs'])
            ax1.set_xlabel('Energy Saving (%)')
            ax1.set_ylabel('Average Throughput (Gbps)')
            ax1.set_title('Energy Saving vs Throughput Trade-off')
            ax1.grid(True, alpha=0.3)
            
            # 추세선
            if len(metrics['energy_savings']) > 3:
                z = np.polyfit(metrics['energy_savings'], metrics['average_throughputs'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(metrics['energy_savings']), 
                                    max(metrics['energy_savings']), 100)
                ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        # 시드별 성능 비교
        seeds = list(range(1, len(metrics['energy_savings']) + 1))
        
        ax2_twin = ax2.twinx()
        
        if metrics['energy_savings']:
            ax2.bar(seeds, metrics['energy_savings'], alpha=0.7, color='green', 
                   label='Energy Saving')
            ax2.set_ylabel('Energy Saving (%)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
        
        if metrics['average_throughputs']:
            ax2_twin.plot(seeds, metrics['average_throughputs'], 'ro-', 
                         label='Avg Throughput')
            ax2_twin.set_ylabel('Average Throughput (Gbps)', color='red')
            ax2_twin.tick_params(axis='y', labelcolor='red')
        
        ax2.set_xlabel('Seed')
        ax2.set_title('Performance by Seed')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.experiment_dir, 'energy_throughput_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")
    
    def save_summary_report(self, metrics: Dict):
        """종합 보고서 저장"""
        report = {
            'experiment_info': {
                'timestamp': self.timestamp,
                'n_seeds': self.n_seeds,
                'successful_runs': len(self.all_results),
                'base_config': self.base_config_path
            },
            'aggregate_statistics': {
                'cumulative_regret': {
                    'mean': np.mean(metrics['cumulative_regrets']),
                    'std': np.std(metrics['cumulative_regrets']),
                    'min': np.min(metrics['cumulative_regrets']),
                    'max': np.max(metrics['cumulative_regrets'])
                },
                'average_regret': {
                    'mean': np.mean(metrics['average_regrets']),
                    'std': np.std(metrics['average_regrets']),
                    'min': np.min(metrics['average_regrets']),
                    'max': np.max(metrics['average_regrets'])
                },
                'energy_saving': {
                    'mean': np.mean(metrics['energy_savings']),
                    'std': np.std(metrics['energy_savings']),
                    'min': np.min(metrics['energy_savings']),
                    'max': np.max(metrics['energy_savings'])
                },
                'average_throughput': {
                    'mean': np.mean(metrics['average_throughputs']),
                    'std': np.std(metrics['average_throughputs']),
                    'min': np.min(metrics['average_throughputs']),
                    'max': np.max(metrics['average_throughputs'])
                },
                'convergence_episode': {
                    'mean': np.mean([e for e in metrics['convergence_episodes'] if e > 0]),
                    'std': np.std([e for e in metrics['convergence_episodes'] if e > 0]),
                    'converged_seeds': sum(1 for e in metrics['convergence_episodes'] if e > 0)
                },
                'switching_cost': {
                    'mean': np.mean(metrics['switching_costs']),
                    'std': np.std(metrics['switching_costs']),
                    'min': np.min(metrics['switching_costs']),
                    'max': np.max(metrics['switching_costs'])
                }
            },
            'best_arm_consensus': self._analyze_best_arms(metrics),
            'raw_metrics': metrics
        }
        
        # JSON 파일로 저장
        report_path = os.path.join(self.experiment_dir, 'summary_report.json')
        with open(report_path, 'w') as f:
            # numpy 타입을 기본 Python 타입으로 변환
            json.dump(self._convert_numpy_types(report), f, indent=2)
        print(f"Saved summary report: {report_path}")
        
        # CSV 형식으로도 저장
        self._save_metrics_csv(metrics)
    
    def _analyze_best_arms(self, metrics: Dict) -> Dict:
        """최적 arm 합의 분석"""
        best_arms = metrics['best_arm_selections']
        unique_arms, counts = np.unique(best_arms, return_counts=True)
        
        consensus = {}
        for arm, count in zip(unique_arms, counts):
            consensus[int(arm)] = {
                'count': int(count),
                'percentage': float(count / len(best_arms) * 100)
            }
        
        # 가장 많이 선택된 arm
        most_common_idx = np.argmax(counts)
        most_common_arm = unique_arms[most_common_idx]
        
        return {
            'arm_distribution': consensus,
            'most_common_arm': int(most_common_arm),
            'consensus_rate': float(counts[most_common_idx] / len(best_arms) * 100)
        }
    
    def _convert_numpy_types(self, obj):
        """NumPy 타입을 JSON 호환 타입으로 변환"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        return obj
    
    def _save_metrics_csv(self, metrics: Dict):
        """메트릭을 CSV 파일로 저장"""
        # 각 시드별 결과를 행으로 정리
        rows = []
        for i, result in enumerate(self.all_results):
            row = {
                'seed': result['seed'],
                'execution_time': result['execution_time'],
                'cumulative_regret': metrics['cumulative_regrets'][i] if i < len(metrics['cumulative_regrets']) else None,
                'average_regret': metrics['average_regrets'][i] if i < len(metrics['average_regrets']) else None,
                'final_efficiency': metrics['final_efficiencies'][i] if i < len(metrics['final_efficiencies']) else None,
                'energy_saving': metrics['energy_savings'][i] if i < len(metrics['energy_savings']) else None,
                'average_throughput': metrics['average_throughputs'][i] if i < len(metrics['average_throughputs']) else None,
                'convergence_episode': metrics['convergence_episodes'][i] if i < len(metrics['convergence_episodes']) else None,
                'switching_cost': metrics['switching_costs'][i] if i < len(metrics['switching_costs']) else None,
                'best_arm': metrics['best_arm_selections'][i] if i < len(metrics['best_arm_selections']) else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.experiment_dir, 'metrics_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics CSV: {csv_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='EXP3 Multi-seed Evaluation')
    parser.add_argument('-c', '--config', type=str, required=True,
                      help='Base configuration file path')
    parser.add_argument('-n', '--n-seeds', type=int, default=10,
                      help='Number of seeds to run (default: 10)')
    parser.add_argument('-o', '--output-dir', type=str, default='_/data/output',
                      help='Output directory (default: _/data/output)')
    
    args = parser.parse_args()
    
    # 평가기 생성 및 실행
    evaluator = EXP3MultiSeedEvaluator(
        base_config_path=args.config,
        n_seeds=args.n_seeds,
        output_base_dir=args.output_dir
    )
    
    # 모든 시드 실행
    evaluator.run_all_seeds()
    
    # 메트릭 계산
    metrics = evaluator.calculate_metrics()
    
    if metrics:
        # 플롯 생성
        evaluator.generate_plots(metrics)
        
        # 보고서 저장
        evaluator.save_summary_report(metrics)
        
        print(f"\n{'='*80}")
        print(f"Evaluation complete! Results saved in: {evaluator.experiment_dir}")
        print(f"{'='*80}")
    else:
        print("No metrics to analyze. Please check the experiment results.")


if __name__ == "__main__":
    main()
