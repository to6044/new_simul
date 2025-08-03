#!/usr/bin/env python3
"""
EXP3 다중 시드 실험 결과 분석 스크립트

사용법:
python exp3_analysis.py --results_dir _/data/output --config exp3_training.json

디렉토리 구조:
data/output/
└── exp3_cell_optimization_training_fixed/
    └── 2025_07_17/
        └── 120233/
            ├── ecotf_s0_p43_0/  # 시드별 폴더
            │   ├── exp3_learning_progress_fixed.json
            │   └── exp3_trained_model_fixed.json
            ├── ecotf_s1_p43_0/
            └── analysis_results/  # 분석 결과
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pandas as pd
from datetime import datetime
from scipy import stats
import os
import sys
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 한글 폰트 설정
import platform
import matplotlib as mpl
if platform.system() == 'Linux':
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

# 전역 설정
SEED_DISPLAY_COUNT = 10  # 모든 그래프에서 표시할 시드 수 통일
MOVING_AVG_WINDOW = 50  # 이동평균 창 크기


class EXP3MultiSeedAnalyzer:
    def __init__(self, results_dir, config_file=None, output_dir=None):
        self.results_dir = Path(results_dir)
        self.config_file = config_file
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / 'analysis_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 결과 저장용 딕셔너리
        self.all_results = defaultdict(list)
        self.metrics = {}
        self.seed_data = {}  # 시드별 데이터 저장
        
    def find_seed_directories(self):
        """새로운 디렉토리 구조에서 시드별 폴더 찾기"""
        seed_dirs = []
        
        print(f"🔍 결과 디렉토리 탐색: {self.results_dir}")
        
        if not self.results_dir.exists():
            print(f"❌ 디렉토리가 존재하지 않습니다: {self.results_dir}")
            return seed_dirs
        
        # 시드 폴더 패턴 (예: ecotf_s0_p43_0, ecotf_s1_p43_0)
        seed_pattern = re.compile(r'.*_s(\d+)_p.*')
        
        # 직접 하위 디렉토리들 확인
        for item in self.results_dir.iterdir():
            if item.is_dir() and seed_pattern.match(item.name):
                # analysis_results 폴더는 제외
                if item.name != 'analysis_results':
                    seed_dirs.append(item)
        
        # 시드 번호로 정렬
        seed_dirs.sort(key=lambda x: int(seed_pattern.match(x.name).group(1)))
        
        print(f"✅ 발견된 시드 폴더: {len(seed_dirs)}개")
        for seed_dir in seed_dirs[:5]:  # 처음 5개만 출력
            print(f"   - {seed_dir.name}")
        if len(seed_dirs) > 5:
            print(f"   ... 그리고 {len(seed_dirs) - 5}개 더")
        
        return seed_dirs
    
    def load_csv_data(self, csv_file):
        """CSV 파일에서 에피소드 데이터 로드"""
        try:
            df = pd.read_csv(csv_file)
            
            # 리스트 형태로 변환 (기존 코드와 호환성 유지)
            episode_data = {
                'reward_history': df['reward'].tolist(),
                'selected_arm_history': df['selected_arm'].tolist(),
                'efficiency_history': df['efficiency'].tolist(),
                'throughput_history': df['throughput_mbps'].tolist(),
                'power_history': df['power_kw'].tolist(),
                'energy_saving_history': df['energy_saving_pct'].tolist(),
                'cumulative_regret_history': df['cumulative_regret'].tolist(),
                'timestamps': df['timestamp'].tolist(),
                'episodes': df['episode'].tolist()
            }
            
            # 통계 계산
            episode_data['statistics'] = {
                'reward_mean': df['reward'].mean(),
                'reward_std': df['reward'].std(),
                'efficiency_mean': df['efficiency'].mean(),
                'efficiency_std': df['efficiency'].std(),
                'throughput_mean': df['throughput_mbps'].mean(),
                'throughput_std': df['throughput_mbps'].std(),
                'energy_saving_mean': df['energy_saving_pct'].mean(),
                'energy_saving_std': df['energy_saving_pct'].std(),
                'final_cumulative_regret': df['cumulative_regret'].iloc[-1] if len(df) > 0 else 0
            }
            
            return episode_data
            
        except Exception as e:
            print(f"❌ CSV 파일 로드 오류: {e}")
            return None
        
    def load_seed_results(self, seed_dir):
        """시드 디렉토리에서 결과 파일 로드 (JSON + CSV)"""
        progress_file = None
        model_file = None
        csv_file = None
        
        # 해당 시드 폴더에서 파일 찾기
        for file in seed_dir.iterdir():
            if 'learning_progress' in file.name and file.suffix == '.json':
                progress_file = file
            elif 'trained_model' in file.name and file.suffix == '.json':
                model_file = file
            elif 'episodes.csv' in file.name:
                csv_file = file
        
        if not progress_file:
            print(f"⚠️ {seed_dir.name}에서 progress JSON 파일을 찾을 수 없습니다.")
            return None, None
        
        # JSON 파일 로드
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # model 파일이 없으면 progress에서 추출
            if model_file and model_file.exists():
                with open(model_file, 'r') as f:
                    model_data = json.load(f)
            else:
                # progress_data에서 model 정보 추출
                model_data = {
                    'weights': progress_data.get('model_state', {}).get('weights', []),
                    'arms': self.extract_arms_from_progress(progress_data),
                    'total_episodes': progress_data.get('episode', 0)
                }
            
            # CSV 파일 로드 및 병합
            if csv_file and csv_file.exists():
                episode_data = self.load_csv_data(csv_file)
                if episode_data:
                    # progress_data에 CSV 데이터 병합
                    progress_data.update(episode_data)
                    print(f"✅ {seed_dir.name}: CSV 데이터 병합 완료")
            else:
                # CSV 파일이 없으면 episode_data_csv 경로에서 찾기
                csv_path = progress_data.get('episode_data_csv')
                if csv_path and Path(csv_path).exists():
                    episode_data = self.load_csv_data(csv_path)
                    if episode_data:
                        progress_data.update(episode_data)
                        print(f"✅ {seed_dir.name}: CSV 데이터 병합 완료 (경로: {csv_path})")
                else:
                    print(f"⚠️ {seed_dir.name}: CSV 파일을 찾을 수 없습니다")
            
            return progress_data, model_data
            
        except Exception as e:
            print(f"❌ {seed_dir.name} 파일 로드 오류: {e}")
            return None, None
        
        
    def load_all_seeds(self):
        """모든 시드의 데이터 로드"""
        seed_dirs = self.find_seed_directories()
        
        if not seed_dirs:
            print("❌ 분석할 시드 폴더를 찾을 수 없습니다.")
            return False
        
        successful_loads = 0
        
        for seed_dir in seed_dirs:
            progress_data, model_data = self.load_seed_results(seed_dir)
            
            if progress_data and model_data:
                # 시드 번호 추출
                seed_match = re.search(r'_s(\d+)_', seed_dir.name)
                seed_num = int(seed_match.group(1)) if seed_match else len(self.seed_data)
                
                self.seed_data[seed_num] = {
                    'progress': progress_data,
                    'model': model_data,
                    'dir': seed_dir
                }
                successful_loads += 1
        
        print(f"\n✅ {successful_loads}/{len(seed_dirs)}개 시드 데이터 로드 완료")
        return successful_loads > 0
    
    def extract_arms_from_progress(self, progress_data):
        """progress_data에서 arms 정보 추출"""
        # best_arm 정보에서 추출 시도
        best_arm_info = progress_data.get('best_arm', {})
        if 'cells_to_turn_off' in best_arm_info:
            # 전체 arms 리스트를 재구성하기 어려우므로 None 반환
            return None
        return None    
    
    
    def calculate_seed_regret(self, progress_data, model_data):
        """수정된 후회(regret) 계산 - CSV 데이터 활용"""
        
        # CSV에서 로드된 데이터 사용
        cumulative_regret_history = progress_data.get('cumulative_regret_history', [])
        reward_history = progress_data.get('reward_history', [])
        selected_arm_history = progress_data.get('selected_arm_history', [])
        
        if cumulative_regret_history:
            # 이미 계산된 누적 regret 사용
            # 순간 regret 계산
            instant_regret = [0] + [cumulative_regret_history[i] - cumulative_regret_history[i-1] 
                                   for i in range(1, len(cumulative_regret_history))]
            
            # 최적 arm 정보는 JSON의 regret_statistics에서
            regret_stats = progress_data.get('performance_summary', {}).get('regret_statistics', {})
            best_arm_idx = regret_stats.get('best_arm_idx', -1)
            best_reward = regret_stats.get('best_arm_avg_reward', 0)
            
            return cumulative_regret_history, instant_regret, best_arm_idx, best_reward
        
        # cumulative_regret이 없으면 reward_history에서 계산
        if not reward_history or not selected_arm_history:
            return [], [], None, 0
        
        # 각 arm의 평균 보상 계산
        arm_rewards = defaultdict(list)
        for arm, reward in zip(selected_arm_history, reward_history):
            arm_rewards[arm].append(reward)
        
        # 최적 arm 찾기
        arm_avg_rewards = {}
        for arm, rewards in arm_rewards.items():
            arm_avg_rewards[arm] = np.mean(rewards)
        
        if not arm_avg_rewards:
            return [], [], None, 0
            
        best_arm = max(arm_avg_rewards, key=arm_avg_rewards.get)
        best_reward = arm_avg_rewards[best_arm]
        
        # 누적 후회 계산
        instant_regret = []
        for t, (chosen_arm, reward) in enumerate(zip(selected_arm_history, reward_history)):
            regret = best_reward - reward
            instant_regret.append(max(0, regret))
        
        cumulative_regret = np.cumsum(instant_regret).tolist()
        
        return cumulative_regret, instant_regret, best_arm, best_reward
    
    
    def calculate_energy_savings(self, progress_data):
        """에너지 절감율 계산 - CSV 데이터 활용"""
        
        # CSV에서 직접 로드된 에너지 절감 데이터 사용
        energy_saving_history = progress_data.get('energy_saving_history', [])
        
        if energy_saving_history:
            return {
                'mean': np.mean(energy_saving_history),
                'std': np.std(energy_saving_history),
                'final': energy_saving_history[-1] if energy_saving_history else 0
            }
        
        # 대체 방법: JSON의 performance_summary 사용
        energy_stats = progress_data.get('performance_summary', {}).get('energy_statistics', {})
        if energy_stats:
            return {
                'mean': energy_stats.get('avg_energy_saving_all_on', 0),
                'std': energy_stats.get('std_energy_saving', 0),
                'final': energy_stats.get('current_energy_saving', 0)
            }
        
        return {'mean': 0, 'std': 0, 'final': 0}
    
    def analyze_performance(self):
        """성능 분석 - CSV와 JSON 데이터 통합"""
        print("\n📊 성능 지표 분석 중...")
        
        for seed_num, data in self.seed_data.items():
            progress = data['progress']
            model = data['model']
            
            # 에너지 절감율 (CSV 데이터 활용)
            energy_savings = self.calculate_energy_savings(progress)
            self.all_results['energy_savings'].append(energy_savings['mean'])
            
            # 평균 보상 (CSV 데이터)
            reward_history = progress.get('reward_history', [])
            if reward_history:
                avg_reward = np.mean(reward_history[-100:])  # 마지막 100개
            else:
                avg_reward = progress.get('performance_summary', {}).get('recent_avg_reward', 0)
            self.all_results['avg_rewards'].append(avg_reward)
            
            # Throughput (CSV 데이터)
            throughput_history = progress.get('throughput_history', [])
            if throughput_history:
                avg_throughput = np.mean(throughput_history[-100:])
            else:
                throughput_stats = progress.get('performance_summary', {}).get('throughput_statistics', {})
                avg_throughput = throughput_stats.get('avg_throughput_mbps', 0)
            self.all_results['avg_throughputs'].append(avg_throughput)
            
            # 최종 가중치 (JSON 데이터)
            weights = model.get('weights', [])
            if not weights:
                weights = progress.get('model_state', {}).get('weights', [])
            self.all_results['final_weights'].append(weights)
            
            # 수렴 정보 (JSON 데이터)
            convergence_info = progress.get('learning_status', {})
            self.all_results['convergence_episodes'].append(
                convergence_info.get('convergence_episode', -1)
            )
            self.all_results['is_converged'].append(
                convergence_info.get('is_converged', False)
            )
        
        # 전체 통계 계산
        self.metrics['energy_savings_mean'] = np.mean(self.all_results['energy_savings'])
        self.metrics['energy_savings_std'] = np.std(self.all_results['energy_savings'])
        self.metrics['reward_mean'] = np.mean(self.all_results['avg_rewards'])
        self.metrics['reward_std'] = np.std(self.all_results['avg_rewards'])
        self.metrics['throughput_mean'] = np.mean(self.all_results['avg_throughputs'])
        self.metrics['throughput_std'] = np.std(self.all_results['avg_throughputs'])
        
        print(f"✅ 평균 에너지 절감율: {self.metrics['energy_savings_mean']:.1f}% ± {self.metrics['energy_savings_std']:.1f}%")
        print(f"✅ 평균 보상: {self.metrics['reward_mean']:.4f} ± {self.metrics['reward_std']:.4f}")
        print(f"✅ 평균 Throughput: {self.metrics['throughput_mean']:.1f} ± {self.metrics['throughput_std']:.1f} Mbps")
    
            
            
    def calculate_statistics(self):
        """통계 지표 계산"""
        # 에너지 절감율 통계
        if self.all_results['energy_savings']:
            self.metrics['energy_savings_mean'] = np.mean(self.all_results['energy_savings'])
            self.metrics['energy_savings_std'] = np.std(self.all_results['energy_savings'])
        else:
            self.metrics['energy_savings_mean'] = 0
            self.metrics['energy_savings_std'] = 0
        
        # 평균 보상
        if self.all_results['rewards']:
            all_rewards_flat = [r for rewards in self.all_results['rewards'] for r in rewards]
            if all_rewards_flat:
                self.metrics['reward_mean'] = np.mean(all_rewards_flat)
                self.metrics['reward_std'] = np.std(all_rewards_flat)
            else:
                self.metrics['reward_mean'] = 0
                self.metrics['reward_std'] = 0
        else:
            self.metrics['reward_mean'] = 0
            self.metrics['reward_std'] = 0
        
        # 처리량 통계
        if self.all_results['avg_throughput']:
            self.metrics['throughput_mean'] = np.mean(self.all_results['avg_throughput'])
            self.metrics['throughput_std'] = np.std(self.all_results['avg_throughput'])
        else:
            self.metrics['throughput_mean'] = 0
            self.metrics['throughput_std'] = 0
        
        print(f"\n📈 주요 지표:")
        print(f"  - 평균 에너지 절감율: {self.metrics['energy_savings_mean']:.1f}% ± {self.metrics['energy_savings_std']:.1f}%")
        print(f"  - 평균 보상: {self.metrics.get('reward_mean', 0):.4f} ± {self.metrics.get('reward_std', 0):.4f}")
        if self.metrics.get('throughput_mean', 0) > 0:
            print(f"  - 평균 처리량: {self.metrics['throughput_mean']:.2f} Mbps ± {self.metrics['throughput_std']:.2f}")
        
    def calculate_overall_regret(self):
        """메트릭 계산 - 수정된 버전"""
        if not self.all_results['cumulative_regret']:
            print("⚠️ cumulative_regret 데이터가 없습니다. 메트릭 계산을 건너뜁니다.")
            return
        
        # 각 시드의 regret 데이터 길이 확인
        regret_lengths = [len(regret) for regret in self.all_results['cumulative_regret']]
        print(f"각 시드의 regret 데이터 길이: {regret_lengths}")
        
        if all(length == 0 for length in regret_lengths):
            print("❌ 모든 시드의 regret 데이터가 비어있습니다.")
            return
        
        # 모든 시드의 최대 에피소드 수 찾기
        max_episodes = max(regret_lengths)
        
        # 평균 누적 후회 계산
        aligned_regrets = []
        for regret in self.all_results['cumulative_regret']:
            if len(regret) == 0:  # 빈 리스트는 건너뛰기
                continue
            # 짧은 시드는 마지막 값으로 패딩
            if len(regret) < max_episodes:
                padded = regret + [regret[-1]] * (max_episodes - len(regret))
            else:
                padded = regret[:max_episodes]
            aligned_regrets.append(padded)
        
        if not aligned_regrets:
            print("❌ 유효한 regret 데이터가 없습니다.")
            return
        
        self.metrics['avg_cumulative_regret'] = np.mean(aligned_regrets, axis=0)
        self.metrics['std_cumulative_regret'] = np.std(aligned_regrets, axis=0)
        
        # 평균 후회 계산
        self.metrics['avg_regret'] = self.metrics['avg_cumulative_regret'] / np.arange(1, max_episodes + 1)
        
        print(f"✅ 메트릭 계산 완료: {max_episodes} 에피소드")
    

    
    def save_summary(self):
        """분석 요약 저장"""
        summary_file = self.output_dir / 'analysis_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("EXP3 Multi-Seed Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {self.results_dir}\n")
            f.write(f"Number of Seeds Analyzed: {len(self.seed_data)}\n\n")
            
            f.write("Key Performance Indicators:\n")
            f.write("-" * 30 + "\n")
            f.write(f"1. Energy Savings: {self.metrics['energy_savings_mean']:.1f}% ± {self.metrics['energy_savings_std']:.1f}%\n")
            f.write(f"2. Average Reward: {self.metrics.get('reward_mean', 0):.4f} ± {self.metrics.get('reward_std', 0):.4f}\n")
            if 'throughput_mean' in self.metrics:
                f.write(f"3. Average Throughput: {self.metrics['throughput_mean']:.2e} Mbps\n")
            if 'avg_regret' in self.metrics and len(self.metrics['avg_regret']) > 0:
                f.write(f"4. Final Average Regret: {self.metrics['avg_regret'][-1]:.4f}\n")
            
            f.write("\nSeed-wise Results:\n")
            f.write("-" * 30 + "\n")
            for seed_num in sorted(self.seed_data.keys())[:10]:  # 처음 10개만
                if seed_num < len(self.all_results['energy_savings']):
                    f.write(f"Seed {seed_num}: Energy Savings = {self.all_results['energy_savings'][seed_num]:.1f}%\n")
        
        # JSON 형식으로도 저장
        metrics_file = self.output_dir / 'analysis_metrics.json'
        save_metrics = {
            'summary': {
                'total_seeds': len(self.seed_data),
                'energy_savings_mean': self.metrics.get('energy_savings_mean', 0),
                'energy_savings_std': self.metrics.get('energy_savings_std', 0),
                'reward_mean': self.metrics.get('reward_mean', 0),
                'reward_std': self.metrics.get('reward_std', 0),
                'throughput_mean': self.metrics.get('throughput_mean', 0),
                'throughput_std': self.metrics.get('throughput_std', 0),
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(save_metrics, f, indent=2)
        
        print(f"\n✅ 분석 요약이 저장되었습니다:")
        print(f"   - {summary_file}")
        print(f"   - {metrics_file}")


    def plot_learning_curves(self, save_dir):
        """학습 곡선 그리기 - 개선된 버전"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 보상 히스토리 with EMA
        ax = axes[0, 0]
        for seed_num, data in list(self.seed_data.items())[:SEED_DISPLAY_COUNT]:
            rewards = data['progress'].get('reward_history', [])
            if rewards:
                # 이동평균
                window = MOVING_AVG_WINDOW
                if len(rewards) > window:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    ax.plot(moving_avg, label=f'Seed {seed_num}', alpha=0.7)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Reward (MA window={MOVING_AVG_WINDOW})')
        ax.set_title('Reward Learning Curves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        
        # 2. 에너지 절감율 with EMA
        ax = axes[0, 1]
        all_energy_data = []
        
        for seed_num, data in list(self.seed_data.items())[:SEED_DISPLAY_COUNT]:
            energy_savings = data['progress'].get('energy_saving_history', [])
            if energy_savings:
                # 원본 데이터 (투명하게)
                ax.plot(energy_savings, alpha=0.3, color='gray')
                
                # EMA (Exponential Moving Average)
                ema_alpha = 2 / (MOVING_AVG_WINDOW + 1)
                ema = pd.Series(energy_savings).ewm(alpha=ema_alpha, adjust=False).mean()
                ax.plot(ema, label=f'Seed {seed_num}', linewidth=2)
                all_energy_data.append(energy_savings)
        
        # 목표 밴드 추가 (9-10% 예시)
        ax.axhspan(9, 10, alpha=0.2, color='green', label='Target Band')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy Saving (%)')
        ax.set_title('Energy Saving Over Time (with EMA)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        
        # 3. 누적 후회 with 95% CI
        ax = axes[1, 0]
        all_cumulative_regrets = []
        
        for seed_num, data in list(self.seed_data.items())[:SEED_DISPLAY_COUNT]:
            cumulative_regret = data['progress'].get('cumulative_regret_history', [])
            if cumulative_regret:
                episodes = range(1, len(cumulative_regret) + 1)
                ax.plot(episodes, cumulative_regret, alpha=0.3, color='blue')
                all_cumulative_regrets.append(cumulative_regret)
        
        if all_cumulative_regrets:
            # 평균과 95% CI 계산
            min_length = min(len(cr) for cr in all_cumulative_regrets)
            truncated_regrets = np.array([cr[:min_length] for cr in all_cumulative_regrets])
            mean_regret = np.mean(truncated_regrets, axis=0)
            std_regret = np.std(truncated_regrets, axis=0)
            ci_95 = 1.96 * std_regret / np.sqrt(len(all_cumulative_regrets))
            
            episodes = range(1, len(mean_regret) + 1)
            ax.plot(episodes, mean_regret, 'k-', linewidth=2, label='Mean')
            ax.fill_between(episodes, mean_regret - ci_95, mean_regret + ci_95, 
                            alpha=0.3, color='gray', label='95% CI')
            
            # 이론적 O(√T) 가이드라인
            n_arms = 969  # EXP3의 arm 수
            theoretical_bound = 2 * np.sqrt(np.arange(1, len(mean_regret) + 1) * n_arms * np.log(n_arms))
            ax.plot(episodes, theoretical_bound, 'r--', label=r'$O(\sqrt{T})$ bound', alpha=0.7)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title('Cumulative Regret with 95% CI')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Throughput with SLA
        ax = axes[1, 1]
        for seed_num, data in list(self.seed_data.items())[:SEED_DISPLAY_COUNT]:
            throughput = data['progress'].get('throughput_history', [])
            if throughput:
                window = MOVING_AVG_WINDOW
                if len(throughput) > window:
                    moving_avg = np.convolve(throughput, np.ones(window)/window, mode='valid')
                    ax.plot(moving_avg, label=f'Seed {seed_num}', alpha=0.7)
        
        # SLA 기준선 (예: 300 Mbps)
        ax.axhline(y=300, color='red', linestyle='--', label='SLA (300 Mbps)', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Throughput (Mbps, MA window={MOVING_AVG_WINDOW})')
        ax.set_title('Network Throughput')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'learning_curves_improved.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_regret_analysis(self, save_dir):
        """후회(regret) 분석 플롯 - 개선된 버전"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('EXP3 Regret Analysis Across Seeds', fontsize=16)
        
        # 1. 누적 regret 비교
        ax = axes[0, 0]
        for seed_num, data in list(self.seed_data.items())[:SEED_DISPLAY_COUNT]:
            cumulative_regret = data['progress'].get('cumulative_regret_history', [])
            if cumulative_regret:
                episodes = range(1, len(cumulative_regret) + 1)
                ax.plot(episodes, cumulative_regret, alpha=0.6)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title('Cumulative Regret Comparison')
        ax.grid(True, alpha=0.3)
        
        # 2. 평균 regret (선형 스케일)
        ax = axes[0, 1]
        all_avg_regrets = []
        for seed_num, data in list(self.seed_data.items())[:SEED_DISPLAY_COUNT]:
            cumulative_regret = data['progress'].get('cumulative_regret_history', [])
            if cumulative_regret:
                avg_regret = [cumulative_regret[i] / (i + 1) for i in range(len(cumulative_regret))]
                ax.plot(avg_regret, alpha=0.6)
                all_avg_regrets.append(avg_regret)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Regret')
        ax.set_title('Average Regret (Linear Scale)')
        ax.grid(True, alpha=0.3)
        
        # 3. 평균 regret (로그 스케일)
        ax = axes[0, 2]
        for avg_regret in all_avg_regrets:
            ax.semilogy(avg_regret, alpha=0.6)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Regret (log scale)')
        ax.set_title('Average Regret (Log Scale)')
        ax.grid(True, alpha=0.3)
        
        # 4. 최종 regret 분포 (개선된 bin)
        ax = axes[1, 0]
        final_regrets = []
        for data in self.seed_data.values():
            cum_regret = data['progress'].get('cumulative_regret_history', [])
            if cum_regret:
                final_regrets.append(cum_regret[-1])
        
        if final_regrets:
            # Freedman-Diaconis rule for bin width
            q75, q25 = np.percentile(final_regrets, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(final_regrets) ** (1/3))
            n_bins = int((max(final_regrets) - min(final_regrets)) / bin_width)
            n_bins = max(n_bins, 10)  # 최소 10개 bin
            
            ax.hist(final_regrets, bins=n_bins, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(final_regrets), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(final_regrets):.2f}')
            ax.axvline(np.median(final_regrets), color='green', linestyle='--', 
                    label=f'Median: {np.median(final_regrets):.2f}')
        
        ax.set_xlabel('Final Cumulative Regret')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Final Regret')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Regret 비율 (모든 시드)
        ax = axes[1, 1]
        regret_ratios = []
        seeds = []
        
        for seed_num, data in self.seed_data.items():
            regret_stats = data['progress'].get('performance_summary', {}).get('regret_statistics', {})
            regret_ratio = regret_stats.get('regret_ratio', 0)
            if regret_ratio > 0:
                regret_ratios.append(regret_ratio)
                seeds.append(seed_num)
        
        if regret_ratios:
            ax.scatter(seeds, regret_ratios, s=50, alpha=0.6)
            
            # 평균과 표준편차
            mean_ratio = np.mean(regret_ratios)
            std_ratio = np.std(regret_ratios)
            ax.axhline(y=mean_ratio, color='blue', linestyle='-', alpha=0.5, 
                    label=f'Mean: {mean_ratio:.4f}')
            ax.fill_between(seeds, mean_ratio - std_ratio, mean_ratio + std_ratio, 
                            alpha=0.2, color='blue', label=f'±1 STD')
        
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Theoretical Limit')
        ax.set_xlabel('Seed Number')
        ax.set_ylabel('Regret Ratio (Actual/Theoretical)')
        ax.set_title('Regret Ratio - All Seeds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 빈 플롯 처리
        ax = axes[1, 2]
        ax.text(0.5, 0.5, 'Additional Analysis\n(Reserved)', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'regret_analysis_improved.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_energy_throughput_comparison(self, save_dir):
        """에너지-처리량 비교 플롯 - 개선된 버전"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 에너지 절감 vs Throughput 산점도 with 회귀분석
        ax = axes[0]
        energy_savings_final = []
        throughput_final = []
        
        for seed_num, data in self.seed_data.items():
            energy_history = data['progress'].get('energy_saving_history', [])
            throughput_history = data['progress'].get('throughput_history', [])
            
            if energy_history and throughput_history:
                final_energy = np.mean(energy_history[-100:])
                final_throughput = np.mean(throughput_history[-100:])
                
                energy_savings_final.append(final_energy)
                throughput_final.append(final_throughput)
                
                ax.scatter(final_energy, final_throughput, s=50, alpha=0.6)
        
        if energy_savings_final and throughput_final:
            # 평균점 표시
            ax.scatter(np.mean(energy_savings_final), np.mean(throughput_final), 
                    s=200, c='red', marker='*', label='Average', zorder=5)
            
            # 회귀선과 상관계수
            slope, intercept, r_value, p_value, std_err = stats.linregress(energy_savings_final, throughput_final)
            line_x = np.array([min(energy_savings_final), max(energy_savings_final)])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', alpha=0.8)
            
            # 상관계수와 p값 표시
            ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Energy Saving (%)')
        ax.set_ylabel('Throughput (Mbps)')
        ax.set_title('Energy Saving vs Throughput Trade-off')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. 시간에 따른 효율성 변화 with 평형 구간 강조
        ax = axes[1]
        for seed_num, data in list(self.seed_data.items())[:SEED_DISPLAY_COUNT]:
            efficiency_history = data['progress'].get('efficiency_history', [])
            if efficiency_history:
                # EMA 적용
                ema_alpha = 2 / (MOVING_AVG_WINDOW + 1)
                ema = pd.Series(efficiency_history).ewm(alpha=ema_alpha, adjust=False).mean()
                ax.plot(ema, label=f'Seed {seed_num}', alpha=0.7)
                
                # 마지막 20% 구간 강조
                last_20_percent = int(len(efficiency_history) * 0.8)
                ax.axvspan(last_20_percent, len(efficiency_history), 
                        alpha=0.1, color='gray', label='Equilibrium Region' if seed_num == 0 else "")
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Efficiency (bits/J, EMA α={ema_alpha:.3f})')
        ax.set_title('Network Efficiency Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'energy_throughput_comparison_improved.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_convergence_analysis(self, save_dir):
        """수렴 분석 그래프 - 개선된 버전"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 가중치 상대 엔트로피 (KL divergence to uniform)
        ax = axes[0, 0]
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        
        weight_kl_divergences = []
        seed_numbers = []
        
        for seed_num, data in self.seed_data.items():
            weights = data['model'].get('weights', [])
            if not weights:
                weights = data['progress'].get('model_state', {}).get('weights', [])
            
            if weights:
                weights_array = np.array(weights)
                # 정규화
                weights_norm = weights_array / (weights_array.sum() + 1e-10)
                # 균등분포
                uniform_dist = np.ones(len(weights)) / len(weights)
                # KL divergence
                kl_div = np.sum(weights_norm * np.log(weights_norm / uniform_dist + 1e-10))
                weight_kl_divergences.append(kl_div)
                seed_numbers.append(seed_num)
        
        if weight_kl_divergences:
            ax.scatter(seed_numbers, weight_kl_divergences, s=50)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Uniform (KL=0)')
        
        ax.set_xlabel('Seed Number')
        ax.set_ylabel('KL Divergence from Uniform')
        ax.set_title('Weight Distribution Concentration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 수렴 에피소드 분포 또는 N/A
        ax = axes[0, 1]
        converged_episodes = []
        
        for data in self.seed_data.values():
            conv_ep = data['progress'].get('learning_status', {}).get('convergence_episode', -1)
            if conv_ep > 0:
                converged_episodes.append(conv_ep)
        
        if converged_episodes:
            ax.hist(converged_episodes, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(converged_episodes), color='red', linestyle='--',
                    label=f'Mean: {np.mean(converged_episodes):.0f}')
            ax.set_xlabel('Convergence Episode')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Convergence Detected\n(Criterion may be too strict)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        ax.set_title(f'Convergence Episodes (τ={0.01}, M={50})')
        ax.grid(True, alpha=0.3)
        
        # 3. 최적 arm 선택 빈도 - 개선된 시각화
        ax = axes[1, 0]
        best_arms = []
        for data in self.seed_data.values():
            best_arm_info = data['progress'].get('best_arm', {})
            best_arm_idx = best_arm_info.get('index', -1)
            if best_arm_idx >= 0:
                best_arms.append(best_arm_idx)
        
        if best_arms:
            unique_arms, counts = np.unique(best_arms, return_counts=True)
            # 가장 많이 선택된 arm
            most_common_idx = np.argmax(counts)
            most_common_arm = unique_arms[most_common_idx]
            most_common_count = counts[most_common_idx]
            
            # 빈도별 색상
            colors = ['red' if arm == most_common_arm else 'blue' for arm in unique_arms]
            ax.bar(range(len(unique_arms)), counts, color=colors, alpha=0.7)
            
            ax.set_xlabel('Best Arm Index')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Best Arm Distribution (Most common: Arm {most_common_arm}, {most_common_count}/{len(self.seed_data)} seeds)')
            ax.grid(True, alpha=0.3)
        
        # 4. 최종 성능 분포 with baseline
        ax = axes[1, 1]
        final_rewards = []
        for data in self.seed_data.values():
            reward_history = data['progress'].get('reward_history', [])
            if reward_history:
                final_avg_reward = np.mean(reward_history[-100:])
                final_rewards.append(final_avg_reward)
        
        if final_rewards:
            bp = ax.boxplot(final_rewards, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            
            # 무작위 선택 baseline (예시값)
            random_baseline = 0.3
            ax.axhline(y=random_baseline, color='red', linestyle='--', 
                    label=f'Random baseline ({random_baseline:.3f})')
            
            ax.set_ylabel('Final Average Reward')
            ax.set_title('Final Performance Distribution')
            ax.set_ylim(bottom=0.25, top=max(final_rewards) * 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'convergence_analysis_improved.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_distribution(self, save_dir):
        """성능 분포 분석 - 개선된 버전"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 모든 패널에 대해 동일한 처리
        metrics = [
            ('energy_saving_history', 'Energy Saving (%)', axes[0, 0]),
            ('throughput_history', 'Throughput (Mbps)', axes[0, 1]),
            ('reward_history', 'Reward', axes[1, 0]),
            ('efficiency_history', 'Efficiency (bits/J)', axes[1, 1])
        ]
        
        for metric_name, ylabel, ax in metrics:
            all_values = []
            ema_values = []  # EMA 기반 값
            
            for data in self.seed_data.values():
                history = data['progress'].get(metric_name, [])
                if history:
                    # 마지막 20% 데이터
                    final_portion = history[int(len(history)*0.8):]
                    all_values.extend(final_portion)
                    
                    # EMA 기반 최종값
                    ema_alpha = 0.1
                    ema = pd.Series(history).ewm(alpha=ema_alpha, adjust=False).mean()
                    ema_values.append(ema.iloc[-1])
            
            if all_values:
                # 히스토그램
                ax.hist(all_values, bins=50, density=True, alpha=0.5, edgecolor='black', 
                        label='Raw (last 20%)')
                
                # EMA 분포
                if ema_values:
                    ax.hist(ema_values, bins=20, density=True, alpha=0.7, 
                            edgecolor='red', color='red', label='EMA final')
                
                # 통계값
                mean_val = np.mean(all_values)
                median_val = np.median(all_values)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2)
                
                # 짧은 텍스트 표기
                ax.text(0.02, 0.98, f'μ={mean_val:.1f}\nM={median_val:.1f}', 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(ylabel)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {ylabel.split(" ")[0]}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_distribution_improved.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_arm_selection_frequency(self, save_dir):
        """Arm 선택 빈도 분석 - 개선된 버전"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 상위 arm 선택 빈도 - 가로 막대 그래프
        ax = axes[0]
        
        total_arm_counts = defaultdict(int)
        for data in self.seed_data.values():
            arm_counts = data['progress'].get('model_state', {}).get('arm_selection_count', [])
            if arm_counts:
                for i, count in enumerate(arm_counts):
                    total_arm_counts[i] += count
        
        if total_arm_counts:
            # 상위 20개 arm
            sorted_arms = sorted(total_arm_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            arms, counts = zip(*sorted_arms)
            
            y_pos = np.arange(len(arms))
            bars = ax.barh(y_pos, counts, alpha=0.7, color='steelblue')
            
            # 값 라벨 표시
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{count}', ha='left', va='center')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f'Arm {arm}' for arm in arms])
            ax.set_xlabel('Total Selection Count')
            ax.set_title('Top 20 Arms by Selection Frequency')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 2. 시드별 최적 arm의 일관성 - jitter 적용
        ax = axes[1]
        
        seed_best_arms = []
        for seed_num, data in self.seed_data.items():
            best_arm = data['progress'].get('best_arm', {}).get('index', -1)
            if best_arm >= 0:
                seed_best_arms.append((seed_num, best_arm))
        
        if seed_best_arms:
            seeds, best_arms = zip(*seed_best_arms)
            
            # Jitter 추가
            jitter = np.random.normal(0, 0.1, len(seeds))
            ax.scatter(np.array(seeds) + jitter, best_arms, s=50, alpha=0.6)
            
            # 가장 많이 선택된 best arm
            unique_best, counts = np.unique(best_arms, return_counts=True)
            most_common_arm = unique_best[np.argmax(counts)]
            most_common_count = counts[np.argmax(counts)]
            
            ax.axhline(y=most_common_arm, color='red', linestyle='--', 
                    label=f'Most Common: Arm {most_common_arm}')
            
            ax.set_xlabel('Seed Number')
            ax.set_ylabel('Best Arm Index')
            ax.set_title(f'Best Arm Selection Across Seeds\n(Most common: Arm {most_common_arm}, {most_common_count}/{len(self.seed_data)} seeds)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'arm_selection_frequency_improved.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_final_performance_summary(self, save_dir):
        """최종 성능 요약 플롯 - 개선된 버전"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. 주요 지표 박스플롯 (단위 통일)
        ax1 = fig.add_subplot(gs[0, :2])
        
        metrics_data = []
        metric_labels = []
        
        # 에너지 절감률
        energy_savings = []
        for data in self.seed_data.values():
            energy_history = data['progress'].get('energy_saving_history', [])
            if energy_history:
                energy_savings.append(np.mean(energy_history[-100:]))
        if energy_savings:
            metrics_data.append(energy_savings)
            metric_labels.append('Energy\nSaving (%)')
        
        # 평균 보상
        avg_rewards = []
        for data in self.seed_data.values():
            reward_history = data['progress'].get('reward_history', [])
            if reward_history:
                avg_rewards.append(np.mean(reward_history[-100:]))
        if avg_rewards:
            metrics_data.append(avg_rewards)
            metric_labels.append('Average\nReward')
        
        # Throughput (Mbps로 통일)
        avg_throughputs = []
        for data in self.seed_data.values():
            throughput_history = data['progress'].get('throughput_history', [])
            if throughput_history:
                avg_throughputs.append(np.mean(throughput_history[-100:]))
        if avg_throughputs:
            metrics_data.append(avg_throughputs)
            metric_labels.append('Throughput\n(Mbps)')
        
        if metrics_data:
            bp = ax1.boxplot(metrics_data, labels=metric_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax1.set_title('Performance Metrics Distribution', fontsize=14)
            ax1.grid(True, alpha=0.3)
        
        # 2. 수렴 통계 with 기준 표시
        ax2 = fig.add_subplot(gs[0, 2])
        
        convergence_stats = {'Converged': 0, 'Not Converged': 0}
        for data in self.seed_data.values():
            is_converged = data['progress'].get('learning_status', {}).get('is_converged', False)
            if is_converged:
                convergence_stats['Converged'] += 1
            else:
                convergence_stats['Not Converged'] += 1
        
        if sum(convergence_stats.values()) > 0:
            colors = ['green', 'red']
            ax2.pie(convergence_stats.values(), labels=convergence_stats.keys(), 
                    autopct='%1.1f%%', startangle=90, colors=colors)
            ax2.set_title(f'Convergence Status\n(τ=0.01, M=50 steps)', fontsize=14)
        
        # 3. 성능 vs 에너지 절감 산점도 with r, p
        ax3 = fig.add_subplot(gs[1, 0])
        
        if energy_savings and avg_rewards:
            ax3.scatter(energy_savings, avg_rewards, s=100, alpha=0.6, 
                        c=range(len(energy_savings)), cmap='viridis')
            ax3.set_xlabel('Energy Saving (%)')
            ax3.set_ylabel('Average Reward')
            ax3.set_title('Energy-Performance Trade-off', fontsize=14)
            ax3.grid(True, alpha=0.3)
            
            # 추세선과 통계
            if len(energy_savings) > 3:
                slope, intercept, r_value, p_value, _ = stats.linregress(energy_savings, avg_rewards)
                line_x = np.array([min(energy_savings), max(energy_savings)])
                line_y = slope * line_x + intercept
                ax3.plot(line_x, line_y, "r--", alpha=0.8)
                
                # r, p 값 표시
                ax3.text(0.95, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                        transform=ax3.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. 최종 Regret 분포
        ax4 = fig.add_subplot(gs[1, 1])
        
        final_regrets = []
        for data in self.seed_data.values():
            regret_history = data['progress'].get('cumulative_regret_history', [])
            if regret_history:
                final_regrets.append(regret_history[-1])
        
        if final_regrets:
            ax4.hist(final_regrets, bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax4.axvline(np.mean(final_regrets), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(final_regrets):.1f}')
            ax4.set_xlabel('Final Cumulative Regret')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Final Regret Distribution', fontsize=14)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. 통계 요약 테이블 (형식 통일)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        summary_stats = []
        summary_stats.append(['Metric', 'Mean ± Std'])
        summary_stats.append(['', ''])
        
        if energy_savings:
            summary_stats.append(['Energy Saving (%)', f'{np.mean(energy_savings):.1f} ± {np.std(energy_savings):.1f}'])
        if avg_rewards:
            summary_stats.append(['Avg Reward', f'{np.mean(avg_rewards):.4f} ± {np.std(avg_rewards):.4f}'])
        if avg_throughputs:
            summary_stats.append(['Throughput (Mbps)', f'{np.mean(avg_throughputs):.1f} ± {np.std(avg_throughputs):.1f}'])
        if final_regrets:
            summary_stats.append(['Final Regret', f'{np.mean(final_regrets):.1f} ± {np.std(final_regrets):.1f}'])
        
        summary_stats.append(['', ''])
        summary_stats.append(['Total Seeds (N)', str(len(self.seed_data))])
        summary_stats.append(['Converged', f'{convergence_stats["Converged"]}/{len(self.seed_data)}'])
        summary_stats.append(['Episodes (T)', str(data['progress'].get('episode', 0))])
        summary_stats.append(['MA Window', str(MOVING_AVG_WINDOW)])
        
        # 테이블 생성
        table = ax5.table(cellText=summary_stats, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 헤더 스타일
        for i in range(len(summary_stats[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax5.set_title('Summary Statistics', fontsize=14, pad=20)
        
        # 전체 제목
        fig.suptitle('EXP3 Multi-Seed Analysis Summary', fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'final_performance_summary_improved.png', dpi=300, bbox_inches='tight')
        plt.close()
    def plot_all_results(self):
        """모든 결과 플롯 생성"""
        print("\n📊 결과 시각화 중...")
        
        # 분석 결과 디렉토리 생성
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 각 플롯 생성
        self.plot_learning_curves(plots_dir)
        self.plot_regret_analysis(plots_dir)
        self.plot_energy_throughput_comparison(plots_dir)
        self.plot_convergence_analysis(plots_dir)
        self.plot_performance_distribution(plots_dir)
        self.plot_arm_selection_frequency(plots_dir)
        self.plot_final_performance_summary(plots_dir)   
        
    def run_analysis(self):
        """전체 분석 실행"""
        print("\n🚀 EXP3 다중 시드 분석 시작...")
        print("=" * 60)
        
        # 1. 모든 시드 데이터 로드
        if not self.load_all_seeds():
            print("❌ 데이터 로드 실패")
            return
        
        # 2. 성능 분석
        self.analyze_performance()
        
        # 3. 후회 분석
        self.calculate_overall_regret()
        
        # 4. 결과 시각화
        self.plot_all_results()
        
        # 5. 요약 저장
        self.save_summary()
        
        print("\n" + "=" * 60)
        print("✅ 분석 완료!")


def run_analysis_integrated(config_file, output_base_dir="data/output", current_run_timestamp=None):
    """
    시뮬레이션 완료 후 자동으로 실행되는 분석 함수
    
    Args:
        config_file (str): 사용된 설정 파일 경로
        output_base_dir (str): 출력 기본 디렉토리
        current_run_timestamp (str): 현재 실행의 타임스탬프 (YYYY_MM_DD/HHMMSS 형식)
    """
    print("\n" + "="*60)
    print("🔬 EXP3 실험 결과 자동 분석 시작...")
    print("="*60)
    
    # 설정 파일에서 실험 이름 추출
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    experiment_name = config.get('experiment_description', 'exp3_experiment')
    
    # 현재 실행의 결과 디렉토리 지정
    if current_run_timestamp:
        # 타임스탬프 파싱
        date_part, time_part = current_run_timestamp.split('/')
        results_dir = Path(output_base_dir) / experiment_name / date_part / time_part
    else:
        # 최신 실행 찾기
        exp_dir = Path(output_base_dir) / experiment_name
        if not exp_dir.exists():
            print(f"❌ 실험 디렉토리를 찾을 수 없습니다: {exp_dir}")
            return
        
        # 최신 날짜/시간 찾기
        latest_date = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)[0]
        latest_time = sorted([t for t in latest_date.iterdir() if t.is_dir()], reverse=True)[0]
        results_dir = latest_time
    
    print(f"📁 분석 대상 디렉토리: {results_dir}")
    
    # 분석기 실행
    analyzer = EXP3MultiSeedAnalyzer(
        results_dir=results_dir,
        config_file=config_file
    )
    
    try:
        analyzer.run_analysis()
    except Exception as e:
        print(f"\n❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def main():
    """독립 실행용 main 함수"""
    parser = argparse.ArgumentParser(
        description="EXP3 다중 시드 실험 결과 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                사용 예시:
                # 특정 실행 결과 분석
                python exp3_analysis.py --results_dir data/output/exp3_cell_optimization_training_fixed/2025_07_17/120233
                
                # 최신 결과 자동 찾기
                python exp3_analysis.py --experiment_name exp3_cell_optimization_training_fixed
                        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        help='결과가 있는 정확한 디렉토리 경로'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='실험 이름 (최신 결과 자동 찾기)'
    )
    
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='data/output',
        help='출력 기본 디렉토리 (기본값: data/output)'
    )
    
    args = parser.parse_args()
    
    if args.results_dir:
        # 직접 지정된 디렉토리 분석
        analyzer = EXP3MultiSeedAnalyzer(results_dir=args.results_dir)
        analyzer.run_analysis()
    elif args.experiment_name:
        # 최신 결과 찾아서 분석
        run_analysis_integrated(
            config_file=None,  # 독립 실행시에는 config 파일 없어도 됨
            output_base_dir=args.output_base_dir,
            current_run_timestamp=None  # 최신 찾기
        )
    else:
        parser.print_help()
        print("\n❌ --results_dir 또는 --experiment_name 중 하나를 지정해야 합니다.")


if __name__ == "__main__":
    main()