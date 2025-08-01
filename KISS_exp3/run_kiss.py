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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 분석 모듈 import 추가
from exp3_analysis import EXP3MultiSeedAnalyzer

def send_email_notification(config_file, elapsed_time, timestamp):
    """
    시뮬레이션 완료 시 이메일 알림을 보내는 함수
    """
    # 환경변수에서 이메일 설정 읽기
    sender_email = os.environ.get('KISS_EMAIL_SENDER')
    sender_password = os.environ.get('KISS_EMAIL_PASSWORD')
    receiver_email = os.environ.get('KISS_EMAIL_RECEIVER')
    
    # 환경변수가 설정되지 않은 경우 경고
    if not all([sender_email, sender_password, receiver_email]):
        print("⚠️  이메일 환경변수가 설정되지 않았습니다.")
        print("다음 환경변수를 설정하세요:")
        print("  export KISS_EMAIL_SENDER='your_email@gmail.com'")
        print("  export KISS_EMAIL_PASSWORD='your_app_password'")
        print("  export KISS_EMAIL_RECEIVER='receiver_email@gmail.com'")
        return
    
    # 이메일 내용 구성
    subject = "KISS 시뮬레이션 완료"
    body = f"""
KISS 시뮬레이션이 완료되었습니다.

실행 정보:
- 설정 파일: {config_file}
- 실행 시간: {elapsed_time:.2f}초
- 타임스탬프: {timestamp}
- 결과 경로: data/output/<experiment>/{timestamp.replace('/', '/')}/

자동 생성된 알림입니다.
"""
    
    # 이메일 메시지 생성
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    
    try:
        # Gmail SMTP 서버 연결
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # 이메일 전송
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        
        print("✉️  이메일 알림이 성공적으로 전송되었습니다.")
    except Exception as e:
        print(f"⚠️  이메일 전송 실패: {e}")


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


def run_analysis(config_file, output_base_dir="data/output", current_run_timestamp=None):
    """
    실험 완료 후 자동 분석 실행 - 새로운 디렉토리 구조 지원
    
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
        # 타임스탬프 파싱 (YYYY_MM_DD/HHMMSS 형식)
        date_part, time_part = current_run_timestamp.split('/')
        results_dir = Path(output_base_dir) / experiment_name / date_part / time_part
        
        print(f"📁 분석 대상 디렉토리: {results_dir}")
        
        # 디렉토리 존재 확인
        if not results_dir.exists():
            print(f"⚠️ 경고: 디렉토리가 존재하지 않습니다: {results_dir}")
            
            # 대체 경로들 시도
            alt_paths = [
                Path(output_base_dir.replace("_/", "")) / experiment_name / date_part / time_part,
                Path("output") / experiment_name / date_part / time_part,
                Path(".") / "output" / experiment_name / date_part / time_part,
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    print(f"✅ 대체 경로에서 발견: {alt_path}")
                    results_dir = alt_path
                    break
            else:
                print("❌ 결과 디렉토리를 찾을 수 없습니다.")
                print("\n🔍 현재 디렉토리 구조:")
                try:
                    base = Path(output_base_dir) / experiment_name
                    if base.exists():
                        for date_dir in sorted(base.iterdir())[-3:]:  # 최근 3개 날짜
                            print(f"  {date_dir.name}/")
                            for time_dir in sorted(date_dir.iterdir())[-3:]:  # 최근 3개 시간
                                print(f"    {time_dir.name}/")
                except:
                    pass
                return
    else:
        # 최신 실행 찾기
        exp_dir = Path(output_base_dir) / experiment_name
        if not exp_dir.exists():
            print(f"❌ 실험 디렉토리를 찾을 수 없습니다: {exp_dir}")
            return
        
        # 최신 날짜 찾기
        dates = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
        if not dates:
            print("❌ 실행 결과를 찾을 수 없습니다.")
            return
        
        latest_date = dates[0]
        
        # 최신 시간 찾기
        times = sorted([t for t in latest_date.iterdir() if t.is_dir()], reverse=True)
        if not times:
            print("❌ 실행 결과를 찾을 수 없습니다.")
            return
        
        latest_time = times[0]
        results_dir = latest_time
        
        print(f"📁 최신 결과 디렉토리: {results_dir}")
    
    
    # 분석기 실행
    analyzer = EXP3MultiSeedAnalyzer(
        results_dir=results_dir,
        config_file=config_file
    )
    
    try:
        # 새로운 메서드 이름 사용
        analyzer.run_analysis()
        
        print(f"\n✅ 분석 완료! 결과는 다음 위치에 저장되었습니다:")
        print(f"   {analyzer.output_dir}")
        
        # 주요 결과 요약 출력
        summary_file = analyzer.output_dir / 'analysis_summary.txt'
        if summary_file.exists():
            print("\n📋 분석 요약:")
            print("-" * 40)
            # 요약의 일부만 출력 (처음 10줄)
            with open(summary_file, 'r') as f:
                lines = f.readlines()
                for line in lines[:15]:  # 처음 15줄만
                    print(line.rstrip())
                if len(lines) > 15:
                    print("... (자세한 내용은 파일을 확인하세요)")
                
    except Exception as e:
        print(f"\n❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 디버깅 정보 출력
        print("\n🔍 디버깅 정보:")
        print(f"  - 결과 디렉토리: {results_dir}")
        print(f"  - 디렉토리 존재 여부: {results_dir.exists()}")
        if results_dir.exists():
            print(f"  - 하위 폴더 수: {len(list(results_dir.iterdir()))}")
            # 처음 5개 하위 폴더 출력
            for i, item in enumerate(results_dir.iterdir()):
                if i >= 5:
                    print("    ...")
                    break
                print(f"    - {item.name}")



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
    
    # 이메일 알림 옵션 추가
    parser.add_argument(
        '--email-notification',
        action='store_true',
        help='시뮬레이션 완료 시 이메일 알림 전송'
    )
    
    args = parser.parse_args()

    # 분석만 수행하는 경우
    if args.analysis_only:
        if args.analysis_date:
            # 날짜가 지정된 경우 정확한 형식으로 전달
            run_analysis(args.config_file, current_run_timestamp=args.analysis_date)
        else:
            # 날짜가 없으면 최신 결과 분석
            run_analysis(args.config_file)
        exit(0)

    # Start the timer
    start_time = time.time()
    
    # 현재 실행의 타임스탬프 생성 (모든 시드가 공유)
    current_run_date_raw = datetime.now().strftime("%Y%m%d")  # "20250721"
    current_run_date = datetime.now().strftime("%Y_%m_%d")    # "2025_07_21" (디렉토리용)
    current_run_time = datetime.now().strftime("%H%M%S")      # "015124"
    execution_timestamp = f"{current_run_date_raw}_{current_run_time}"  # "20250721_015124"

    print(f"🚀 Starting execution with timestamp: {execution_timestamp}")
    print(f"📁 Results will be saved in: data/output/<experiment>/{current_run_date}/{current_run_time}/")
    
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
        print("\n⏳ 파일 시스템 동기화 대기 중...")
        time.sleep(2)
        
        # 분석 실행 - 전체 타임스탬프 전달
        timestamp = f"{current_run_date}/{current_run_time}"
        print(f"\n🔍 분석 시작: {timestamp}")
        run_analysis(args.config_file, current_run_timestamp=timestamp)
    else:
        print("\n💡 분석을 건너뛰었습니다. 나중에 분석하려면 다음 명령을 실행하세요:")
        print(f"   python run_kiss.py -c {args.config_file} --analysis-only --analysis-date {current_run_date}/{current_run_time}")
        

    timestamp = f"{current_run_date}/{current_run_time}"
    send_email_notification(args.config_file, elapsed_time, timestamp)