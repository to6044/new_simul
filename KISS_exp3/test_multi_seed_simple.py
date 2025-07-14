#!/usr/bin/env python3
"""
간단한 다중 시드 테스트 스크립트
exp3_cell_on_off.py 수정 후 작동 확인용

python test_multi_seed_simple.py

"""

import os
import json
import subprocess
import time

def test_single_seed():
    """단일 시드로 빠른 테스트"""
    print("=" * 60)
    print("단일 시드 테스트 시작")
    print("=" * 60)
    
    # 테스트용 설정 생성
    test_config = {
        "experiment_description": "exp3_test_single_seed",
        "experiment_version": "1.0",
        "script_name": "kiss",
        "script_version": "0.8",
        "project_root_dir": ".",
        "debug_logging": True,
        "debug_logging_level": "INFO",
        
        "seed": 42,
        "isd": 500,
        "sim_radius": 1000,
        "h_BS": 25,
        "power_dBm": 43.0,
        "nues": 100,  # 적은 UE로 빠른 테스트
        "h_UT": 1.5,
        "until": 100,  # 짧은 시뮬레이션 시간
        "base_interval": 1.0,
        "ue_noise_power_dBm": -118,
        
        "scenario_profile": "exp3_cell_on_off",
        "scenario_delay": 10.0,
        "exp3_n_cells_off": 2,  # 2개 셀만 끄기
        "exp3_gamma": 0.15,
        "exp3_warmup_episodes": 10,  # 짧은 워밍업
        "exp3_enable_warmup": True,
        "exp3_ensure_min_selection": True,
        "exp3_linear_power_model": True,
        "exp3_power_model_k": 0.08,
        "exp3_learning_log": "test_learning_progress.json",
        "exp3_final_model": "test_trained_model.json",
        
        "mme_cqi_limit": 1,
        "mme_strategy": "best_sinr_cell",
        "mme_anti_pingpong": 5.0,
        "mme_verbosity": 0,
        
        "plotting": False
    }
    
    # 임시 설정 파일 저장
    with open('test_config.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    
    # 시뮬레이션 실행
    try:
        start_time = time.time()
        result = subprocess.run(
            ["python", "run_kiss.py", "-c", "test_config.json"],
            capture_output=True,
            text=True
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ 테스트 성공! (실행 시간: {execution_time:.2f}초)")
            
            # 결과 파일 확인
            output_dir = "_/data/output/exp3_test_single_seed"
            if os.path.exists(output_dir):
                print("\n생성된 파일:")
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.endswith('.json'):
                            filepath = os.path.join(root, file)
                            print(f"  - {filepath}")
                            
                            # 학습 로그 간단 분석
                            if 'learning_progress' in file:
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                    print(f"    Episodes: {data.get('episode', 0)}")
                                    energy_stats = data.get('energy_statistics', {})
                                    if energy_stats:
                                        print(f"    Energy Saving: {energy_stats.get('avg_energy_saving_all_on', 0):.1f}%")
                                    throughput_stats = data.get('throughput_statistics', {})
                                    if throughput_stats:
                                        print(f"    Avg Cell Throughput: {throughput_stats.get('avg_cell_throughput_gbps', 0):.2f} Gbps")
        else:
            print(f"❌ 테스트 실패!")
            print("STDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        # 임시 파일 삭제
        if os.path.exists('test_config.json'):
            os.remove('test_config.json')

def test_multi_seed(n_seeds=3):
    """간단한 다중 시드 테스트"""
    print("\n" + "=" * 60)
    print(f"{n_seeds}개 시드 테스트 시작")
    print("=" * 60)
    
    # exp3_multi_seed_evaluation.py 실행
    try:
        result = subprocess.run(
            ["python", "exp3_multi_seed_evaluation.py", 
             "-c", "data/input/exp3_cell_on_off/exp3_test.json",
             "-n", str(n_seeds)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ 다중 시드 테스트 성공!")
            print("\n마지막 10줄:")
            print('\n'.join(result.stdout.split('\n')[-10:]))
        else:
            print(f"❌ 다중 시드 테스트 실패!")
            print("STDERR:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def check_modifications():
    """exp3_cell_on_off.py 수정 확인"""
    print("=" * 60)
    print("exp3_cell_on_off.py 수정 사항 확인")
    print("=" * 60)
    
    required_methods = [
        'calculate_energy_metrics',
        'calculate_throughput_metrics',
        'update_switching_cost',
        'is_converged',
        '_find_convergence_episode'
    ]
    
    required_vars = [
        'energy_consumption_history',
        'throughput_history',
        'switching_count',
        'baseline_energy_consumption'
    ]
    
    try:
        with open('exp3_cell_on_off.py', 'r') as f:
            content = f.read()
            
        print("메서드 확인:")
        for method in required_methods:
            if f'def {method}' in content:
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ {method} - 추가 필요!")
                
        print("\n변수 확인:")
        for var in required_vars:
            if f'self.{var}' in content:
                print(f"  ✅ {var}")
            else:
                print(f"  ❌ {var} - 추가 필요!")
                
    except FileNotFoundError:
        print("❌ exp3_cell_on_off.py 파일을 찾을 수 없습니다!")

if __name__ == "__main__":
    print("EXP3 다중 시드 평가 테스트")
    print("=" * 80)
    
    # 1. 수정 사항 확인
    check_modifications()
    
    # 2. 단일 시드 테스트
    print("\n")
    test_single_seed()
    
    # 3. 다중 시드 테스트 (옵션)
    response = input("\n다중 시드 테스트를 실행하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        test_multi_seed(3)
    
    print("\n테스트 완료!")
