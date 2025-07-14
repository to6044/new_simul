#!/usr/bin/env python3
"""
출력 디렉토리 구조 확인 및 문제 진단
"""

import os
import json
from pathlib import Path

def check_output_structure():
    """출력 디렉토리 구조 확인"""
    print("=" * 80)
    print("출력 디렉토리 구조 확인")
    print("=" * 80)
    
    output_base = "_/data/output"
    
    # 모든 exp3 관련 디렉토리 찾기
    exp3_dirs = []
    for root, dirs, files in os.walk(output_base):
        for dir_name in dirs:
            if "exp3" in dir_name:
                full_path = os.path.join(root, dir_name)
                exp3_dirs.append(full_path)
    
    print(f"\nEXP3 관련 디렉토리 ({len(exp3_dirs)}개):")
    for dir_path in sorted(exp3_dirs)[-10:]:  # 최근 10개만
        print(f"  {dir_path}")
        
        # JSON 파일 찾기
        json_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        if json_files:
            print(f"    JSON 파일:")
            for json_file in json_files[:5]:  # 최대 5개만
                print(f"      - {os.path.basename(json_file)}")

def test_single_run():
    """단일 실행 테스트 및 출력 확인"""
    print("\n" + "=" * 80)
    print("단일 시뮬레이션 테스트")
    print("=" * 80)
    
    # 테스트 설정 로드
    config_path = "data/input/exp3_cell_on_off/exp3_training_fixed.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 출력 파일 이름 확인
    print(f"\n설정 파일 정보:")
    print(f"  experiment_description: {config.get('experiment_description')}")
    print(f"  exp3_learning_log: {config.get('exp3_learning_log')}")
    print(f"  exp3_final_model: {config.get('exp3_final_model')}")
    
    # 예상 출력 경로
    exp_desc = config.get('experiment_description')
    expected_base = f"_/data/output/{exp_desc}"
    
    print(f"\n예상 출력 경로: {expected_base}")
    
    # 실제로 존재하는지 확인
    if os.path.exists(expected_base):
        print("✅ 기본 출력 디렉토리 존재")
        
        # 날짜 폴더 찾기
        date_dirs = [d for d in os.listdir(expected_base) if os.path.isdir(os.path.join(expected_base, d))]
        print(f"\n날짜 폴더 ({len(date_dirs)}개):")
        for date_dir in sorted(date_dirs)[-5:]:
            print(f"  {date_dir}")
            
            # 시간 폴더 또는 파일 확인
            full_date_path = os.path.join(expected_base, date_dir)
            contents = os.listdir(full_date_path)
            
            print(f"    내용물:")
            for item in contents[:10]:
                item_path = os.path.join(full_date_path, item)
                if os.path.isdir(item_path):
                    print(f"      📁 {item}/")
                else:
                    print(f"      📄 {item}")
    else:
        print("❌ 기본 출력 디렉토리가 존재하지 않습니다")

if __name__ == "__main__":
    check_output_structure()
    test_single_run()
