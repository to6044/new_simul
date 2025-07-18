import argparse
from datetime import datetime
import json
import os
import subprocess
import time
import numpy as np
import signal
import multiprocessing as mp
from pathlib import Path
import sys

# 분석 모듈 import 추가
from exp3_analysis import EXP3MultiSeedAnalyzer


def signal_handler(signum, frame):
    """
    Signal handler function that is called when a signal is received.

    Args:
        signum (int): The signal number.
        frame (frame): The current stack frame.

    Returns:
        None
    """
    print(f"Received signal {signum}, exiting...")
    remove_temp_json(json_file_list)
    exit(1)


def generate_config_dict_list(config_file, execution_timestamp=None):
    """
    Generate a list of dictionaries containing different configurations based on the input JSON file.

    Args:
        config_file (str): The path to the JSON file containing the configuration.
        execution_timestamp (str): The execution timestamp to use for all configs (YYYYMMDD_HHMMSS format)

    Returns:
        list: A list of dictionaries, where each dictionary represents a different configuration.
    """
    # Load the contents of the JSON file into a dictionary
    with open(config_file) as f:
        config = json.load(f)

    seed_max = config['seed']
    power_dBm = config['power_dBm']
    power_dBm_end = config['power_dBm_end']
    power_dBm_step = config['power_dBm_step']

    # Define the range of power values to test
    if power_dBm == power_dBm_end:
        power_values = [power_dBm]
    # If the finish power value is `-inf` then the range of power values will be from the start power value to `0dBm` in steps of `power_dBm_step`, and then an additional value of `-np.inf` will be added to the end of the list
    elif power_dBm_end == "-np.inf":
        power_values = np.arange(power_dBm, 0, -power_dBm_step)
        power_values = np.append(power_values, -np.inf)
    else:
        power_values = np.arange(power_dBm, power_dBm_end-power_dBm_step, -power_dBm_step)

    # Seed values to test 
    seeds = list(range(seed_max))

    # Define the number range of variable power cells
    n_variable_power_cells = list(range(1, config['n_variable_power_cells']+1))

    # Create a list of dictionaries
    config_dict_list = []
    for seed in seeds:
        for power_dBm in power_values:
            if len(n_variable_power_cells) == 0:
                # Create a new dictionary object for each iteration
                config_copy = config.copy()
                
                # Update the seed and power_dBm value in the new dictionary object
                config_copy['seed'] = seed
                config_copy['variable_cell_power_dBm'] = power_dBm
                config_copy['n_variable_power_cells'] = 0
                
                # Add execution timestamp if provided
                if execution_timestamp:
                    config_copy['execution_timestamp'] = execution_timestamp
                
                # Append the new dictionary object to the list
                config_dict_list.append(config_copy)
            else:
                # Create a new dictionary object for each iteration
                for n_cells in n_variable_power_cells:
                    config_copy = config.copy()
                    
                    # Update the values in the new dictionary object
                    config_copy['seed'] = seed
                    config_copy['variable_cell_power_dBm'] = power_dBm
                    config_copy['n_variable_power_cells'] = n_cells
                    
                    # Add execution timestamp if provided
                    if execution_timestamp:
                        config_copy['execution_timestamp'] = execution_timestamp
                    
                    # Append the new dictionary object to the list
                    config_dict_list.append(config_copy)

    return config_dict_list


def create_updated_json_file(config_dict, output_file):
    """
    Create an updated JSON file with the given configuration dictionary.

    Args:
        config_dict (dict): The configuration dictionary to write to the JSON file.
        output_file (str): The path to the output JSON file.

    Returns:
        None
    """
    # Write the updated dictionary to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=4)


# Define a function that takes a list of dictionaries and dumps each dictionary to a separate JSON file
def dump_json(config_dict_list):
    # Create a list to store the JSON file paths
    json_file_list = []

    # Loop through each dictionary in the list
    for i, config_dict in enumerate(config_dict_list):
        # Create a unique JSON file name
        json_file_name = f"temp_{i}.json"

        # Dump the dictionary to the JSON file
        with open(json_file_name, 'w') as f:
            json.dump(config_dict, f, indent=4)

        # Append the JSON file path to the list
        json_file_list.append(json_file_name)

    # Return the list of JSON file paths
    return json_file_list


# Define a function that takes a single json file and runs kiss.py against it
def run_kiss(temp_json_file):
    # Define the command to execute
    command = ["python", "kiss.py", "-c", f"{temp_json_file}"]

    # Execute the command as a subprocess
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print the output of the subprocess
    print(process.stdout.decode())
    if process.returncode != 0:
        print(process.stderr.decode())
        return process.returncode
    else:
        return process.returncode


# Define a function that takes a list of temp JSON file paths and removes them from the file system
def remove_temp_json(json_file_list):
    # Loop through each JSON file in the list
    for json_file in json_file_list:
        # Remove the JSON file from the file system
        try:
            os.remove(json_file)
            # test if file exists
            if os.path.exists(json_file):
                # tried removing and it still exists, so raise an error by moving to the except block below
                raise OSError
            else:
                # Get the json file name
                json_file_name = os.path.basename(json_file)
                print(f"File {json_file_name} removed successfully")
        except OSError as e:
            print(f"Error deleting file: {e}")


# Callback function to update progress
def update_progress(result):
    with progress_counter.get_lock():
        progress_counter.value += 1
        print(f"Processed {progress_counter.value} of {total_files} files")


# Global variables for signal handler
json_file_list = []
progress_counter = None
total_files = 0


def run_analysis(config_file, output_base_dir="_/data/output", current_run_timestamp=None):
    """
    실험 완료 후 자동 분석 실행
    
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
        # 현재 실행의 정확한 폴더 지정
        search_dir = Path(output_base_dir) / experiment_name / current_run_timestamp
        print(f"📁 현재 실행 결과 디렉토리: {search_dir}")
        
        # 디렉토리 존재 확인
        if not search_dir.exists():
            print(f"⚠️ 경고: 디렉토리가 존재하지 않습니다: {search_dir}")
            print("📂 현재 존재하는 디렉토리 확인 중...")
            
            # 상위 디렉토리에서 찾기
            parent_dir = Path(output_base_dir) / experiment_name
            if parent_dir.exists():
                subdirs = [d for d in parent_dir.iterdir() if d.is_dir()]
                print(f"   발견된 디렉토리: {[d.name for d in subdirs[-5:]]}")  # 최근 5개만 표시
            
            # 대체 경로 시도 (data 폴더)
            alt_search_dir = Path("data/output") / experiment_name / current_run_timestamp.replace("_/", "")
            if alt_search_dir.exists():
                print(f"✅ 대체 경로에서 발견: {alt_search_dir}")
                search_dir = alt_search_dir
            else:
                print("❌ 결과 디렉토리를 찾을 수 없습니다.")
                return
        
        # 분석 결과를 같은 디렉토리에 저장
        analysis_output_dir = search_dir / "analysis_results"
    else:
        search_dir = Path(output_base_dir)
        # 타임스탬프가 없으면 별도 디렉토리에 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_output_dir = Path(f"exp3_analysis_{experiment_name}_{timestamp}")
    
    # 분석기 실행
    analyzer = EXP3MultiSeedAnalyzer(
        results_dir=search_dir,
        config_file=config_file,
        output_dir=analysis_output_dir
    )
    
    try:
        analyzer.run()
        print(f"\n✅ 분석 결과가 '{analysis_output_dir}' 디렉토리에 저장되었습니다.")
        
        # 주요 결과 요약 출력
        summary_file = analysis_output_dir / 'analysis_summary.txt'
        if summary_file.exists():
            print("\n📋 분석 요약:")
            print("-" * 40)
            with open(summary_file, 'r') as f:
                print(f.read())
                
    except Exception as e:
        print(f"\n❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Register the signal handler function to handle the SIGINT and SIGTERM signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description='Run kiss.py against a specified config value.')
    parser.add_argument(
        '-c',
        '--config-file',
        type=str,
        required=True,
        default='_test/data/input/kiss/test_kiss.json'
    )
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help='분석을 건너뛰고 시뮬레이션만 실행'
    )
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='시뮬레이션을 건너뛰고 기존 결과만 분석'
    )
    parser.add_argument(
        '--analysis-date',
        type=str,
        help='분석할 실행 날짜 (YYYY_MM_DD 형식, --analysis-only와 함께 사용)'
    )
    
    args = parser.parse_args()

    # 분석만 수행하는 경우
    if args.analysis_only:
        run_analysis(args.config_file, current_run_timestamp=args.analysis_date)
        exit(0)

    # Start the timer
    start_time = time.time()
    
    # 현재 실행의 타임스탬프 생성 (모든 시드가 공유)
    current_run_date = datetime.now().strftime("%Y_%m_%d")
    current_run_time = datetime.now().strftime("%H%M%S")
    execution_timestamp = f"{current_run_date}_{current_run_time}"

    print(f"🚀 Starting execution with timestamp: {execution_timestamp}")

    config_dict_list = generate_config_dict_list(args.config_file, execution_timestamp=execution_timestamp)

    # Dump the list of dictionaries to individual JSON files
    json_file_list = dump_json(config_dict_list)
    total_files = len(json_file_list)

    # Initialize a multiprocessing pool with more processes for your 14-core CPU
    # Using 12 processes to leave some CPU headroom for the system
    num_processes = 12
    num_tasks_per_child = 1
    progress_counter = mp.Value('i', 0)  # Shared variable to keep track of progress

    print(f"Starting parallel processing with {num_processes} processes for {total_files} simulations...")
    print(f"Results will be saved in: data/output/<experiment>/{current_run_date}/{current_run_time}/")

    with mp.Pool(processes=num_processes, maxtasksperchild=num_tasks_per_child) as pool:
        # Store async results
        async_results = []
        
        # Submit all tasks asynchronously
        for json_file in json_file_list:
            # Apply async with callback for progress tracking
            result = pool.apply_async(run_kiss, args=(json_file,), callback=update_progress)
            async_results.append(result)
        
        # Close the pool to prevent any more tasks from being submitted
        pool.close()
        
        # Wait for all processes to complete
        pool.join()
        
        # Optionally, check for any errors in the results
        for i, result in enumerate(async_results):
            try:
                return_code = result.get(timeout=1)  # Should already be done
                if return_code != 0:
                    print(f"Warning: Task {i+1} returned non-zero exit code: {return_code}")
            except Exception as e:
                print(f"Error in task {i+1}: {e}")

    # Remove the temporary JSON files
    remove_temp_json(json_file_list)

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"All subprocesses completed in {elapsed_time:.2f} seconds")
    print(f"Average time per simulation: {elapsed_time/total_files:.2f} seconds")
    
    # 자동 분석 실행 (--no-analysis 옵션이 없는 경우)
    if not args.no_analysis:
        # 잠시 대기 (파일 시스템 동기화)
        time.sleep(2)
        
        # 분석 실행 - 전체 타임스탬프 전달
        run_analysis(args.config_file, current_run_timestamp=f"{current_run_date}/{current_run_time}")
    else:
        print("\n💡 분석을 건너뛰었습니다. 나중에 분석하려면 다음 명령을 실행하세요:")
        print(f"   python run_kiss.py -c {args.config_file} --analysis-only --analysis-date {current_run_date}/{current_run_time}")