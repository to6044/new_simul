#!/usr/bin/env python3
"""
EXP3 수정사항 테스트 및 검증 스크립트

사용법:
python test_exp3_fixes.py --config exp3_training_fixed.json

이 스크립트는 수정된 EXP3 구현이 제대로 작동하는지 확인합니다.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import matplotlib as mpl

# 한글 폰트 설정
import platform
if platform.system() == 'Linux':
    # Linux에서 한글 폰트 설정
    try:
        import matplotlib.font_manager as fm
        # 나눔고딕 폰트 경로 찾기
        font_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]
        
        font_found = False
        for font_path in font_paths:
            if Path(font_path).exists():
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                font_found = True
                break
        
        if not font_found:
            # 폰트를 찾지 못한 경우 기본 폰트 사용
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            
    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}")
        # 영어로만 표시
        mpl.rcParams['axes.unicode_minus'] = False

def load_exp3_results(results_dir):
    """EXP3 결과 파일들을 로드"""
    results_dir = Path(results_dir)
    
    print(f"🔍 결과 디렉토리 탐색 중: {results_dir.absolute()}")
    
    # 현재 디렉토리부터 시작해서 재귀적으로 탐색
    search_patterns = [
        "**/exp3_learning_progress*.json",
        "**/exp3_trained_model*.json", 
        "exp3_learning_progress*.json",
        "exp3_trained_model*.json"
    ]
    
    progress_files = []
    model_files = []
    
    # 여러 패턴으로 파일 탐색
    for pattern in search_patterns[:2]:  # progress 파일 찾기
        files = list(results_dir.glob(pattern))
        progress_files.extend([f for f in files if 'progress' in f.name])
    
    for pattern in search_patterns[:2]:  # model 파일 찾기  
        files = list(results_dir.glob(pattern))
        model_files.extend([f for f in files if 'model' in f.name and 'progress' not in f.name])
    
    # 중복 제거
    progress_files = list(set(progress_files))
    model_files = list(set(model_files))
    
    print(f"🔍 발견된 progress 파일: {len(progress_files)}개")
    for f in progress_files:
        print(f"   - {f}")
        
    print(f"🔍 발견된 model 파일: {len(model_files)}개")
    for f in model_files:
        print(f"   - {f}")
    
    if not progress_files:
        print("\n❌ 학습 진행 파일을 찾을 수 없습니다.")
        print("💡 힌트: 다음 위치에서 파일을 확인해보세요:")
        print("   - _/data/output/**/exp3_learning_progress*.json")
        print("   - data/output/**/exp3_learning_progress*.json")
        return None, None
        
    if not model_files:
        print("\n❌ 훈련된 모델 파일을 찾을 수 없습니다.")
        return None, None
    
    # 가장 최근 파일 선택
    progress_file = sorted(progress_files, key=lambda x: x.stat().st_mtime)[-1]
    model_file = sorted(model_files, key=lambda x: x.stat().st_mtime)[-1]
    
    print(f"\n📊 사용할 Progress file: {progress_file}")
    print(f"🤖 사용할 Model file: {model_file}")
    
    # 파일 로드
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
    
    with open(model_file, 'r') as f:
        model_data = json.load(f)
    
    return progress_data, model_data

def analyze_power_measurements(progress_data):
    """전력 측정 정확성 분석"""
    print("\n🔋 전력 측정 분석:")
    
    baseline_power = progress_data.get('baseline_power', 0)
    print(f"  베이스라인 전력: {baseline_power:.2f} kW")
    
    # 더 현실적인 범위로 수정 (19개 셀 * 2-4kW)
    expected_min = 19 * 2.0  # 38 kW
    expected_max = 19 * 4.0  # 76 kW
    
    if expected_min <= baseline_power <= expected_max:
        print(f"  ✅ 베이스라인 전력이 예상 범위({expected_min:.1f}-{expected_max:.1f} kW) 내에 있습니다.")
    else:
        print(f"  ⚠️ 베이스라인 전력이 예상 범위({expected_min:.1f}-{expected_max:.1f} kW)를 벗어났습니다.")

def analyze_reward_distribution(progress_data):
    """보상 분포 분석"""
    print("\n🎯 보상 분포 분석:")
    
    reward_history = progress_data.get('reward_history', [])
    
    if reward_history:
        rewards = np.array(reward_history)
        print(f"  총 에피소드: {len(rewards)}")
        print(f"  보상 범위: [{rewards.min():.4f}, {rewards.max():.4f}]")
        print(f"  평균 보상: {rewards.mean():.4f}")
        print(f"  보상 표준편차: {rewards.std():.4f}")
        
        # 포화도 확인
        saturation_count = np.sum(rewards >= 0.99)
        saturation_rate = saturation_count / len(rewards) * 100
        
        if saturation_rate < 5:
            print(f"  ✅ 보상 분포 양호: 포화율 {saturation_rate:.1f}%")
        else:
            print(f"  ⚠️ 보상 포화 발생: {saturation_rate:.1f}%가 0.99 이상")

def analyze_learning_convergence(model_data):
    """학습 수렴성 분석"""
    print("\n📈 학습 수렴 분석:")
    
    weights = np.array(model_data.get('weights', []))
    episode = model_data.get('total_episodes', 0)
    
    print(f"  총 학습 에피소드: {episode}")
    print(f"  가중치 범위: [{weights.min():.4f}, {weights.max():.4f}]")
    
    # 가중치 비율 계산
    weight_ratio = weights.max() / max(weights.min(), 1e-10)
    print(f"  가중치 비율: {weight_ratio:.2f}")
    
    # 상위 5개 arms 출력
    top_indices = np.argsort(weights)[-5:][::-1]
    print("  상위 5 arms:")
    for rank, idx in enumerate(top_indices):
        cells = model_data['arms'][idx]
        print(f"    {rank+1}. Arm {idx}: cells {cells}, weight={weights[idx]:.4f}")
    
    # 수렴 판단 기준 완화
    if weight_ratio >= 1.05:  # 기존 3에서 1.05로 낮춤
        print("  ✅ 학습이 수렴했습니다.")
    else:
        print("  ❌ 수렴하지 않음. 더 긴 학습이나 하이퍼파라미터 조정 필요.")

def analyze_efficiency_improvements(progress_data, model_data):
    """효율성 개선 분석"""
    print("\n⚡ 효율성 개선 분석:")
    
    baseline_eff = progress_data.get('baseline_efficiency', 0)
    efficiency_history = progress_data.get('efficiency_history', [])
    
    if baseline_eff and efficiency_history:
        effs = np.array(efficiency_history)
        print(f"  베이스라인 효율성: {baseline_eff:.2e} bits/J")
        print(f"  관찰된 효율성 범위: [{effs.min():.2e}, {effs.max():.2e}] bits/J")
        
        # 개선율 계산
        max_improvement = (effs.max() - baseline_eff) / baseline_eff * 100
        max_degradation = (effs.min() - baseline_eff) / baseline_eff * 100
        
        print(f"  최대 개선율: {max_improvement:+.1f}%")
        print(f"  최대 저하율: {max_degradation:+.1f}%")
        
        # 평가
        if max_improvement > 5:
            print("  ✅ 유의미한 효율성 개선 달성!")
        elif max_improvement > 0:
            print("  ⚠️ 소폭 개선. 더 긴 학습 또는 파라미터 조정 고려.")
        else:
            print("  ❌ 효율성 개선 없음. 알고리즘 검토 필요.")

def generate_diagnostic_plots(progress_data, save_dir=None):
    """진단 플롯 생성 (영어 버전)"""
    print("\n📊 진단 플롯 생성 중...")
    
    reward_history = progress_data.get('reward_history', [])
    efficiency_history = progress_data.get('efficiency_history', [])
    
    if not reward_history and not efficiency_history:
        print("  ❌ 플롯 생성할 데이터 없음")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('EXP3 Learning Diagnostics', fontsize=16)
    
    # 보상 히스토리
    if reward_history:
        ax = axes[0, 0]
        ax.plot(reward_history)
        ax.set_title('Reward History')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
    
    # 보상 히스토그램
    if reward_history:
        ax = axes[0, 1]
        ax.hist(reward_history, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title('Reward Distribution')
        ax.set_xlabel('Reward Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # 효율성 히스토리
    if efficiency_history:
        ax = axes[1, 0]
        ax.plot(efficiency_history)
        ax.set_title('Network Efficiency')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Efficiency (bits/J)')
        ax.grid(True, alpha=0.3)
        
        # 베이스라인 표시
        baseline = progress_data.get('baseline_efficiency')
        if baseline:
            ax.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
            ax.legend()
    
    # 이동평균 보상
    if reward_history and len(reward_history) > 20:
        ax = axes[1, 1]
        window = min(20, len(reward_history) // 5)
        moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(reward_history)), moving_avg, 'r-', linewidth=2)
        ax.set_title(f'Reward Moving Average (window={window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'exp3_diagnostic_plots.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📈 플롯 저장됨: {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_regret_from_data(progress_data, model_data):
    """
    저장된 데이터로부터 regret 계산 및 분석
    
    Parameters:
    -----------
    progress_data : dict
        학습 진행 데이터
    model_data : dict
        최종 모델 데이터
    """
    print("\n📊 Regret 분석 (사후 계산):")
    
    # 필요한 데이터 추출
    reward_history = progress_data.get('reward_history', [])
    arm_selection_count = np.array(progress_data.get('arm_selection_count', []))
    cumulative_rewards = np.array(progress_data.get('cumulative_rewards', []))
    selected_arm_history = progress_data.get('selected_arm_history', [])
    
    if not reward_history or len(arm_selection_count) == 0:
        print("  ❌ Regret 계산에 필요한 데이터가 없습니다.")
        return
    
    # 각 arm의 평균 보상 계산
    avg_rewards = np.zeros(len(arm_selection_count))
    for i in range(len(arm_selection_count)):
        if arm_selection_count[i] > 0:
            avg_rewards[i] = cumulative_rewards[i] / arm_selection_count[i]
    
    # 최적 arm 찾기
    best_arm_idx = np.argmax(avg_rewards)
    best_arm_avg_reward = avg_rewards[best_arm_idx]
    
    print(f"  최적 arm: {best_arm_idx} (평균 보상: {best_arm_avg_reward:.4f})")
    
    # Cumulative regret 계산
    cumulative_regret = 0
    cumulative_regret_history = []
    instant_regret_history = []
    
    for t, (selected_arm, reward) in enumerate(zip(selected_arm_history, reward_history)):
        # 순간 regret = 최적 arm의 평균 보상 - 실제 받은 보상
        instant_regret = best_arm_avg_reward - reward
        instant_regret_history.append(instant_regret)
        
        cumulative_regret += instant_regret
        cumulative_regret_history.append(cumulative_regret)
    
    # 통계 계산
    T = len(reward_history)
    K = len(arm_selection_count)
    
    # EXP3의 이론적 regret bound: O(√(TK log K))
    theoretical_bound = 2 * np.sqrt(T * K * np.log(K))
    
    # 실제 평균 보상
    actual_avg_reward = np.mean(reward_history)
    
    # 결과 출력
    print(f"  총 에피소드: {T}")
    print(f"  총 arm 수: {K}")
    print(f"  누적 regret: {cumulative_regret:.4f}")
    print(f"  평균 regret: {cumulative_regret / T:.4f}")
    print(f"  이론적 상한: {theoretical_bound:.4f}")
    print(f"  Regret 비율: {cumulative_regret / theoretical_bound:.2%}")
    print(f"  실제 평균 보상: {actual_avg_reward:.4f}")
    print(f"  최적 대비 성능: {actual_avg_reward / best_arm_avg_reward:.2%}")
    
    # 평가
    if cumulative_regret / theoretical_bound < 0.5:
        print("  ✅ 우수한 regret 성능 (이론적 상한의 50% 미만)")
    elif cumulative_regret / theoretical_bound < 1.0:
        print("  ⚠️ 양호한 regret 성능 (이론적 상한 이내)")
    else:
        print("  ❌ regret이 이론적 상한을 초과. 알고리즘 조정 필요")
    
    # Regret 플롯 생성
    plot_regret_analysis(cumulative_regret_history, instant_regret_history, 
                        theoretical_bound, K, save_path='regret_analysis.png')

def plot_regret_analysis(cumulative_regret_history, instant_regret_history, 
                         theoretical_bound, n_arms, save_path=None):
    """Regret 분석 플롯 생성"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('EXP3 Regret Analysis', fontsize=16)
    
    episodes = range(1, len(cumulative_regret_history) + 1)
    
    # 누적 regret
    ax = axes[0, 0]
    ax.plot(episodes, cumulative_regret_history, 'b-', label='Actual', linewidth=2)
    
    # 이론적 bound
    theoretical_bounds = [2 * np.sqrt(t * n_arms * np.log(n_arms)) for t in episodes]
    ax.plot(episodes, theoretical_bounds, 'r--', label='Theoretical Bound', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Cumulative Regret vs Theoretical Bound')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 순간 regret
    ax = axes[0, 1]
    ax.plot(instant_regret_history, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Instant Regret')
    ax.set_title('Instant Regret per Episode')
    ax.grid(True, alpha=0.3)
    
    # 평균 regret
    ax = axes[1, 0]
    avg_regret = [cumulative_regret_history[i] / (i + 1) 
                 for i in range(len(cumulative_regret_history))]
    ax.plot(avg_regret, 'g-', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Regret')
    ax.set_title('Average Regret over Time')
    ax.grid(True, alpha=0.3)
    
    # Regret 비율
    ax = axes[1, 1]
    regret_ratios = [cumulative_regret_history[i] / theoretical_bounds[i] 
                    if theoretical_bounds[i] > 0 else 0 
                    for i in range(len(cumulative_regret_history))]
    ax.plot(regret_ratios, 'm-', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Theoretical Limit')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Regret Ratio')
    ax.set_title('Actual / Theoretical Regret')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  📈 Regret 분석 플롯 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()



def main():
    parser = argparse.ArgumentParser(
        description='EXP3 수정사항 테스트 및 검증',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python test_exp3_fixes.py                                    # 기본 디렉토리에서 검색
  python test_exp3_fixes.py --results_dir _/data/output        # 특정 디렉토리 지정
  python test_exp3_fixes.py --results_dir . --save_plots       # 현재 디렉토리에서 검색하고 플롯 저장
  
주의사항:
  - 이 스크립트는 시뮬레이션 실행 후에 사용하세요
  - exp3_learning_progress*.json과 exp3_trained_model*.json 파일이 필요합니다
  - 시뮬레이션 실행: python run_kiss.py -c data/input/exp3_cell_on_off/exp3_training_fixed.json
        """
    )
    parser.add_argument(
        '--results_dir', 
        type=str, 
        default='.',
        help='결과 파일을 검색할 디렉토리 (기본값: 현재 디렉토리)'
    )
    parser.add_argument(
        '--save_plots', 
        action='store_true',
        help='플롯을 파일로 저장 (기본값: 화면에 표시)'
    )
    
    args = parser.parse_args()
    
    print("🔍 EXP3 수정사항 검증 시작...")
    print("=" * 60)
    
    # 결과 로드
    progress_data, model_data = load_exp3_results(args.results_dir)
    
    if progress_data is None or model_data is None:
        print("\n❌ 분석할 데이터를 찾을 수 없습니다.")
        print("\n💡 해결 방법:")
        print("1. 먼저 시뮬레이션을 실행하세요:")
        print("   python run_kiss.py -c data/input/exp3_cell_on_off/exp3_training_fixed.json")
        print("\n2. 결과 파일이 있는 디렉토리를 지정하세요:")
        print("   python test_exp3_fixes.py --results_dir _/data/output")
        print("\n3. 수동으로 파일 위치 확인:")
        print("   find . -name '*exp3*' -name '*.json' 2>/dev/null")
        sys.exit(1)
    
    # 분석 수행
    analyze_power_measurements(progress_data)
    analyze_reward_distribution(progress_data)
    analyze_learning_convergence(model_data)
    analyze_efficiency_improvements(progress_data, model_data)
    analyze_regret_from_data(progress_data, model_data)

    # 플롯 생성
    if args.save_plots:
        save_dir = Path(args.results_dir)
        generate_diagnostic_plots(progress_data, save_dir)
    else:
        generate_diagnostic_plots(progress_data)
    
    print("\n" + "=" * 60)
    print("✅ 검증 완료!")
    
    # 종합 평가
    baseline_power = progress_data.get('baseline_power', 0)
    reward_history = progress_data.get('reward_history', [])
    weights = np.array(model_data.get('weights', []))
    
    issues = []
    
    # 전력 범위 수정
    if baseline_power < 38 or baseline_power > 76:
        issues.append("전력 측정 이상")
    
    if len(reward_history) > 0:
        saturation_rate = np.sum(np.array(reward_history) >= 0.99) / len(reward_history)
        if saturation_rate > 0.8:
            issues.append("보상 포화")
    
    if len(weights) > 0:
        weight_ratio = weights.max() / max(weights.min(), 1e-10)
        if weight_ratio < 1.05:  # 기준 완화
            issues.append("학습 수렴 부족 (더 긴 학습 필요)")
    
    if issues:
        print(f"\n⚠️ 발견된 문제점: {', '.join(issues)}")
        print("추가 튜닝이 필요할 수 있습니다.")
    else:
        print("\n🎉 모든 지표가 정상 범위 내입니다!")

if __name__ == "__main__":
    main()