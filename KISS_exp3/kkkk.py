import json
from pathlib import Path

# 가장 최근 progress 파일 찾기
progress_files = list(Path("data/output").rglob("**/exp3_learning_progress*.json"))
if progress_files:
    latest_file = sorted(progress_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Reading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # energy_statistics 확인
    if 'energy_statistics' in data:
        print(f"\nenergy_statistics: {data['energy_statistics']}")
    else:
        print("\nenergy_statistics not found in data!")
    
    # 다른 관련 필드들
    print(f"\nbaseline_power: {data.get('baseline_power', 'Not found')}")
    print(f"\nKeys in data: {list(data.keys())}")