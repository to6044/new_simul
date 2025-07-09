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
        print("💡 힌트: 다음 위치에서 파일을 확인해보세요:")
        print("   - _/data/output/**/exp3_trained_model*.json")
        print("   - data/output/**/exp3_trained_model*.json")
        return None, None
    
    # 가장 최근 파일 선택
    progress_file = max(progress_files, key=lambda x: x.stat().st_mtime)
    model_file = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print(f"\n📊 사용할 Progress file: {progress_file}")
    print(f"🤖 사용할 Model file: {model_file}")
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        with open(model_file, 'r') as f:
            model_data = json.load(f)
        return progress_data, model_data
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None, None

def analyze_power_measurements(progress_data):
    """전력 측정 분석"""
    print("\n🔋 전력 측정 분석:")
    
    baseline_power = progress_data.get('baseline_power')
    if baseline_power:
        print(f"  베이스라인 전력: {baseline_power:.2f} kW")
        
        # 합리적인 범위 확인
        expected_min = 15.0  # 19셀 × 최소 0.8kW
        expected_max = 60.0  # 19셀 × 최대 3.2kW
        
        if expected_min <= baseline_power <= expected_max:
            print("  ✅ 베이스라인 전력이 합리적 범위 내입니다.")
        else:
            print(f"  ⚠️ 베이스라인 전력이 예상 범위({expected_min}-{expected_max} kW)를 벗어났습니다.")
    else:
        print("  ❌ 베이스라인 전력 정보 없음")

def analyze_reward_distribution(progress_data):
    """보상 분포 분석"""
    print("\n🎯 보상 분포 분석:")
    
    reward_history = progress_data.get('reward_history', [])
    if not reward_history:
        print("  ❌ 보상 히스토리 없음")
        return
    
    rewards = np.array(reward_history)
    
    print(f"  총 에피소드: {len(rewards)}")
    print(f"  보상 범위: [{rewards.min():.4f}, {rewards.max():.4f}]")
    print(f"  평균 보상: {rewards.mean():.4f}")
    print(f"  보상 표준편차: {rewards.std():.4f}")
    
    # 보상 포화 확인
    saturated_rewards = np.sum(rewards >= 0.99)
    saturation_rate = saturated_rewards / len(rewards) * 100
    
    if saturation_rate > 80:
        print(f"  ⚠️ 보상 포화율 높음: {saturation_rate:.1f}% (임계값: 80%)")
        print("     → 보상 함수 조정 필요")
    elif saturation_rate > 50:
        print(f"  ⚠️ 보상 포화율 중간: {saturation_rate:.1f}% (임계값: 50%)")
        print("     → 모니터링 필요")
    else:
        print(f"  ✅ 보상 분포 양호: 포화율 {saturation_rate:.1f}%")

def analyze_learning_convergence(model_data):
    """학습 수렴 분석"""
    print("\n📈 학습 수렴 분석:")
    
    total_episodes = model_data.get('total_episodes', 0)
    weights = np.array(model_data.get('weights', []))
    
    if len(weights) == 0:
        print("  ❌ 가중치 정보 없음")
        return
    
    print(f"  총 학습 에피소드: {total_episodes}")
    
    # 가중치 분산 분석
    max_weight = weights.max()
    min_weight = weights.min()
    weight_ratio = max_weight / min_weight if min_weight > 0 else float('inf')
    
    print(f"  가중치 범위: [{min_weight:.4f}, {max_weight:.4f}]")
    print(f"  가중치 비율: {weight_ratio:.2f}")
    
    # 상위 arms 분석
    top_indices = np.argsort(weights)[-5:][::-1]
    print(f"  상위 5 arms:")
    
    arms = model_data.get('arms', [])
    for i, idx in enumerate(top_indices):
        if idx < len(arms):
            arm_cells = arms[idx]
            print(f"    {i+1}. Arm {idx}: cells {arm_cells}, weight={weights[idx]:.4f}")
    
    # 수렴 판정
    if weight_ratio > 10:
        print("  ✅ 학습이 수렴한 것으로 보입니다.")
    elif weight_ratio > 3:
        print("  ⚠️ 부분적 수렴. 더 많은 에피소드 필요할 수 있습니다.")
    else:
        print("  ❌ 수렴하지 않음. 하이퍼파라미터 조정 필요.")

def analyze_efficiency_improvements(progress_data, model_data):
    """효율성 개선 분석"""
    print("\n⚡ 효율성 개선 분석:")
    
    baseline_eff = progress_data.get('baseline_efficiency') or model_data.get('baseline_efficiency')
    min_eff = progress_data.get('min_efficiency') or model_data.get('min_efficiency')
    max_eff = progress_data.get('max_efficiency') or model_data.get('max_efficiency')
    
    if not baseline_eff:
        print("  ❌ 베이스라인 효율성 정보 없음")
        return
    
    print(f"  베이스라인 효율성: {baseline_eff:.2e} bits/J")
    
    if min_eff and max_eff:
        print(f"  관찰된 효율성 범위: [{min_eff:.2e}, {max_eff:.2e}] bits/J")
        
        improvement = (max_eff - baseline_eff) / baseline_eff * 100
        degradation = (baseline_eff - min_eff) / baseline_eff * 100
        
        print(f"  최대 개선율: +{improvement:.1f}%")
        print(f"  최대 저하율: -{degradation:.1f}%")
        
        if improvement > 5:
            print("  ✅ 유의미한 효율성 개선 달성!")
        elif improvement > 0:
            print("  ⚠️ 소폭 개선. 더 긴 학습 또는 파라미터 조정 고려.")
        else:
            print("  ❌ 효율성 개선 없음. 알고리즘 검토 필요.")

def generate_diagnostic_plots(progress_data, save_dir=None):
    """진단 플롯 생성"""
    print("\n📊 진단 플롯 생성 중...")
    
    reward_history = progress_data.get('reward_history', [])
    efficiency_history = progress_data.get('efficiency_history', [])
    
    if not reward_history and not efficiency_history:
        print("  ❌ 플롯 생성할 데이터 없음")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('EXP3 학습 진단', fontsize=16)
    
    # 보상 히스토리
    if reward_history:
        ax = axes[0, 0]
        ax.plot(reward_history)
        ax.set_title('보상 히스토리')
        ax.set_xlabel('에피소드')
        ax.set_ylabel('보상')
        ax.grid(True, alpha=0.3)
    
    # 보상 히스토그램
    if reward_history:
        ax = axes[0, 1]
        ax.hist(reward_history, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title('보상 분포')
        ax.set_xlabel('보상값')
        ax.set_ylabel('빈도')
        ax.grid(True, alpha=0.3)
    
    # 효율성 히스토리
    if efficiency_history:
        ax = axes[1, 0]
        ax.plot(efficiency_history)
        ax.set_title('네트워크 효율성')
        ax.set_xlabel('에피소드')
        ax.set_ylabel('효율성 (bits/J)')
        ax.grid(True, alpha=0.3)
        
        # 베이스라인 표시
        baseline = progress_data.get('baseline_efficiency')
        if baseline:
            ax.axhline(y=baseline, color='r', linestyle='--', label='베이스라인')
            ax.legend()
    
    # 이동평균 보상
    if reward_history and len(reward_history) > 20:
        ax = axes[1, 1]
        window = min(20, len(reward_history) // 5)
        moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(reward_history)), moving_avg, 'r-', linewidth=2)
        ax.set_title(f'보상 이동평균 (윈도우={window})')
        ax.set_xlabel('에피소드')
        ax.set_ylabel('평균 보상')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'exp3_diagnostic_plots.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📈 플롯 저장됨: {save_path}")
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
    
    if baseline_power < 15 or baseline_power > 60:
        issues.append("전력 측정 이상")
    
    if len(reward_history) > 0:
        saturation_rate = np.sum(np.array(reward_history) >= 0.99) / len(reward_history)
        if saturation_rate > 0.8:
            issues.append("보상 포화")
    
    if len(weights) > 0:
        weight_ratio = weights.max() / max(weights.min(), 1e-10)
        if weight_ratio < 3:
            issues.append("학습 수렴 부족")
    
    if issues:
        print(f"\n⚠️ 발견된 문제점: {', '.join(issues)}")
        print("추가 튜닝이 필요할 수 있습니다.")
    else:
        print("\n🎉 모든 지표가 정상 범위 내입니다!")

if __name__ == "__main__":
    main()