#!/usr/bin/env python3
"""
EXP3 다중 시드 실험 결과 분석 스크립트

사용법:
python analyze_exp3_results.py --results_dir _/data/output --config exp3_training_fixed.json

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
import scipy.stats as stats
import os
import sys
from collections import defaultdict
import re

# 한글 폰트 설정
import platform
import matplotlib as mpl
if platform.system() == 'Linux':
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False


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
    
    def load_seed_results(self, seed_dir):
        """시드 디렉토리에서 결과 파일 로드"""
        progress_file = None
        model_file = None
        
        # 해당 시드 폴더에서 파일 찾기
        for file in seed_dir.iterdir():
            if 'learning_progress' in file.name and file.suffix == '.json':
                progress_file = file
            elif 'trained_model' in file.name and file.suffix == '.json':
                model_file = file
        
        if not progress_file or not model_file:
            print(f"⚠️ {seed_dir.name}에서 필요한 파일을 찾을 수 없습니다.")
            return None, None
        
        # JSON 파일 로드
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            with open(model_file, 'r') as f:
                model_data = json.load(f)
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
    
    def calculate_seed_regret(self, progress_data, model_data):
        """후회(regret) 계산"""
        reward_history = progress_data.get('reward_history', [])
        arm_history = progress_data.get('arm_history', [])
        
        if not reward_history:
            return [], [], None, 0
        
        # 각 arm의 평균 보상 계산
        arm_rewards = defaultdict(list)
        for arm, reward in zip(arm_history, reward_history):
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
        for t, (chosen_arm, reward) in enumerate(zip(arm_history, reward_history)):
            regret = best_reward - reward
            instant_regret.append(max(0, regret))  # 음수 후회는 0으로
        
        cumulative_regret = np.cumsum(instant_regret).tolist()
        
        return cumulative_regret, instant_regret, best_arm, best_reward
    
    def calculate_energy_savings(self, progress_data):
        """에너지 절감율 계산"""
        baseline_power = progress_data.get('baseline_power', 0)
        power_history = progress_data.get('power_history', [])
        
        if not power_history or baseline_power == 0:
            return 0, baseline_power, 0
        
        avg_power = np.mean(power_history)
        savings = (baseline_power - avg_power) / baseline_power * 100
        
        return savings, baseline_power, avg_power
    
    def analyze_performance(self):
        """전체 성능 분석"""
        if not self.seed_data:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        print("\n📊 성능 분석 중...")
        
        # 각 시드별 분석
        for seed_num, data in sorted(self.seed_data.items()):
            progress_data = data['progress']
            model_data = data['model']
            
            # 1. 후회 계산
            cumulative_regret, instant_regret, best_arm, best_reward = self.calculate_seed_regret(
                progress_data, model_data
            )
            self.all_results['cumulative_regret'].append(cumulative_regret)
            self.all_results['instant_regret'].append(instant_regret)
            self.all_results['best_reward'].append(best_reward)
            
            # 2. 에너지 절감율
            savings_rate, baseline_power, avg_power = self.calculate_energy_savings(progress_data)
            self.all_results['energy_savings'].append(savings_rate)
            self.all_results['baseline_power'].append(baseline_power)
            self.all_results['avg_power'].append(avg_power)
            
            # 3. 보상 이력
            rewards = progress_data.get('reward_history', [])
            self.all_results['rewards'].append(rewards)
            
            # 4. 처리량 정보
            throughput_history = progress_data.get('throughput_history', [])
            if throughput_history:
                avg_throughput = np.mean(throughput_history)
                self.all_results['avg_throughput'].append(avg_throughput)
            
            # 5. 최종 가중치
            weights = model_data.get('weights', [])
            self.all_results['final_weights'].append(weights)
        
        # 통계 계산
        self.calculate_statistics()
    
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
            self.metrics['reward_mean'] = np.mean(all_rewards_flat)
            self.metrics['reward_std'] = np.std(all_rewards_flat)
        
        # 처리량 통계
        if self.all_results['avg_throughput']:
            self.metrics['throughput_mean'] = np.mean(self.all_results['avg_throughput'])
            self.metrics['throughput_std'] = np.std(self.all_results['avg_throughput'])
        
        print(f"\n📈 주요 지표:")
        print(f"  - 평균 에너지 절감율: {self.metrics['energy_savings_mean']:.1f}% ± {self.metrics['energy_savings_std']:.1f}%")
        print(f"  - 평균 보상: {self.metrics.get('reward_mean', 0):.4f} ± {self.metrics.get('reward_std', 0):.4f}")
        if 'throughput_mean' in self.metrics:
            print(f"  - 평균 처리량: {self.metrics['throughput_mean']:.2e} bits/s")
    
    def calculate_overall_regret(self):
        """전체 후회 분석"""
        if not self.all_results['cumulative_regret']:
            return
        
        # 모든 시드의 최대 에피소드 수 찾기
        max_episodes = max(len(regret) for regret in self.all_results['cumulative_regret'])
        
        # 평균 누적 후회 계산
        aligned_regrets = []
        for regret in self.all_results['cumulative_regret']:
            # 짧은 시드는 마지막 값으로 패딩
            if len(regret) < max_episodes:
                padded = regret + [regret[-1]] * (max_episodes - len(regret))
            else:
                padded = regret[:max_episodes]
            aligned_regrets.append(padded)
        
        self.metrics['avg_cumulative_regret'] = np.mean(aligned_regrets, axis=0)
        self.metrics['std_cumulative_regret'] = np.std(aligned_regrets, axis=0)
        
        # 평균 후회 계산
        self.metrics['avg_regret'] = self.metrics['avg_cumulative_regret'] / np.arange(1, max_episodes + 1)
    
    def plot_all_results(self, save_dir=None):
        """모든 결과 시각화"""
        save_dir = save_dir or self.output_dir
        
        print("\n📈 그래프 생성 중...")
        
        # 1. 누적 보상
        self.plot_cumulative_rewards(save_dir)
        
        # 2. 누적 후회
        if 'avg_cumulative_regret' in self.metrics:
            self.plot_cumulative_regret(save_dir)
        
        # 3. 평균 후회
        if 'avg_regret' in self.metrics:
            self.plot_average_regret(save_dir)
        
        # 4. Arm 선택 분포
        self.plot_selection_distribution(save_dir)
        
        # 5. 에너지-처리량 비교
        self.plot_energy_throughput_comparison(save_dir)
        
        # 6. 수렴 분석
        self.plot_convergence_analysis(save_dir)
        
        print(f"✅ 모든 그래프가 {save_dir}에 저장되었습니다.")
    
    def plot_cumulative_rewards(self, save_dir):
        """누적 보상 그래프"""
        plt.figure(figsize=(10, 6))
        
        # 각 시드별 누적 보상
        for seed_num, rewards in enumerate(self.all_results['rewards']):
            if rewards:
                cumulative_rewards = np.cumsum(rewards)
                plt.plot(cumulative_rewards, alpha=0.3, label=f'Seed {seed_num}' if seed_num < 5 else None)
        
        # 평균 누적 보상
        if self.all_results['rewards']:
            max_len = max(len(r) for r in self.all_results['rewards'])
            aligned_rewards = []
            for rewards in self.all_results['rewards']:
                if len(rewards) < max_len:
                    padded = rewards + [rewards[-1]] * (max_len - len(rewards))
                else:
                    padded = rewards[:max_len]
                aligned_rewards.append(np.cumsum(padded))
            
            mean_cumulative = np.mean(aligned_rewards, axis=0)
            plt.plot(mean_cumulative, 'r-', linewidth=2, label='Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards over Episodes')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'cumulative_rewards.png', dpi=300)
        plt.close()
    
    def plot_cumulative_regret(self, save_dir):
        """누적 후회 그래프"""
        plt.figure(figsize=(10, 6))
        
        episodes = np.arange(len(self.metrics['avg_cumulative_regret']))
        mean_regret = self.metrics['avg_cumulative_regret']
        std_regret = self.metrics['std_cumulative_regret']
        
        plt.plot(episodes, mean_regret, 'b-', linewidth=2, label='Mean')
        plt.fill_between(episodes, 
                        mean_regret - std_regret,
                        mean_regret + std_regret,
                        alpha=0.3, color='blue', label='±1 STD')
        
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Regret')
        plt.title('Average Cumulative Regret over Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'cumulative_regret.png', dpi=300)
        plt.close()
    
    def plot_average_regret(self, save_dir):
        """평균 후회 그래프"""
        plt.figure(figsize=(10, 6))
        
        episodes = np.arange(1, len(self.metrics['avg_regret']) + 1)
        avg_regret = self.metrics['avg_regret']
        
        plt.plot(episodes, avg_regret, 'g-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Average Regret')
        plt.title('Average Regret per Episode')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'average_regret.png', dpi=300)
        plt.close()
    
    def plot_selection_distribution(self, save_dir):
        """Arm 선택 분포"""
        plt.figure(figsize=(12, 6))
        
        # 모든 시드의 최종 가중치 평균
        if self.all_results['final_weights']:
            all_weights = np.array(self.all_results['final_weights'])
            mean_weights = np.mean(all_weights, axis=0)
            
            # 상위 10개 arms
            top_indices = np.argsort(mean_weights)[-10:][::-1]
            
            plt.bar(range(len(top_indices)), mean_weights[top_indices])
            plt.xlabel('Top 10 Arms')
            plt.ylabel('Average Weight')
            plt.title('Distribution of Final Weights (Top 10 Arms)')
            plt.xticks(range(len(top_indices)), [f'Arm {i}' for i in top_indices], rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(save_dir / 'selection_distribution.png', dpi=300)
            plt.close()
    
    def plot_energy_throughput_comparison(self, save_dir):
        """에너지-처리량 비교"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 에너지 절감율
        if self.all_results['energy_savings']:
            seeds = range(len(self.all_results['energy_savings']))
            ax1.bar(seeds, self.all_results['energy_savings'])
            ax1.axhline(y=self.metrics['energy_savings_mean'], color='r', linestyle='--',
                       label=f'Mean: {self.metrics["energy_savings_mean"]:.1f}%')
            ax1.set_xlabel('Seed')
            ax1.set_ylabel('Energy Savings (%)')
            ax1.set_title('Energy Savings by Seed')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 평균 처리량
        if self.all_results['avg_throughput']:
            seeds = range(len(self.all_results['avg_throughput']))
            ax2.bar(seeds, self.all_results['avg_throughput'])
            if 'throughput_mean' in self.metrics:
                ax2.axhline(y=self.metrics['throughput_mean'], color='r', linestyle='--',
                           label=f'Mean: {self.metrics["throughput_mean"]:.2e}')
            ax2.set_xlabel('Seed')
            ax2.set_ylabel('Average Throughput (bits/s)')
            ax2.set_title('Average Throughput by Seed')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'energy_throughput_comparison.png', dpi=300)
        plt.close()
    
    def plot_convergence_analysis(self, save_dir):
        """수렴 분석 그래프"""
        if not self.all_results['final_weights']:
            return
            
        plt.figure(figsize=(10, 6))
        
        # 각 시드의 가중치 엔트로피 변화
        for seed_num, weights in enumerate(self.all_results['final_weights'][:10]):  # 처음 10개만
            weights_array = np.array(weights)
            # 정규화
            weights_norm = weights_array / weights_array.sum()
            # 엔트로피 계산
            entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-10))
            plt.scatter(seed_num, entropy, label=f'Seed {seed_num}')
        
        plt.xlabel('Seed')
        plt.ylabel('Weight Entropy')
        plt.title('Weight Distribution Entropy by Seed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'convergence_analysis.png', dpi=300)
        plt.close()
    
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
                f.write(f"3. Average Throughput: {self.metrics['throughput_mean']:.2e} bits/s\n")
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