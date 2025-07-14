#!/usr/bin/env python3
"""
EXP3 Multi-seed Evaluation Script (병렬 실행 버전)
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
from multiprocessing import Pool, cpu_count
import concurrent.futures
import glob

class EXP3MultiSeedEvaluatorParallel:
    def __init__(self, base_config_path: str, n_seeds: int = 10, output_base_dir: str = "_/data/output", n_parallel: int = None):
        """
        Parameters:
        -----------
        base_config_path : str
            기본 설정 파일 경로
        n_seeds : int
            실험할 시드 개수
        output_base_dir : str
            결과 저장 기본 디렉토리
        n_parallel : int
            병렬 실행 프로세스 수 (None이면 CPU 코어 수 - 1)
        """
        self.base_config_path = base_config_path
        self.n_seeds = n_seeds
        self.output_base_dir = output_base_dir
        self.n_parallel = n_parallel or max(1, cpu_count() - 1)
        
        # 시간 기반 폴더 생성
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
        
    def run_single_seed_wrapper(self, args):
        """병렬 실행을 위한 래퍼 함수"""
        seed, config_dict = args
        return self.run_single_seed(seed, config_dict)
        
    def run_single_seed(self, seed: int, config_dict: dict = None) -> Dict:
        """단일 시드로 실험 실행"""
        print(f"\n{'='*60}")
        print(f"[Seed {seed}] Starting experiment")
        print(f"{'='*60}")
        
        # 설정 파일 로드 (제공되지 않은 경우)
        if config_dict is None:
            with open(self.base_config_path, 'r') as f:
                config_dict = json.load(f)
        
        # 시드 및 출력 경로 수정
        config_dict['seed'] = seed
        config_dict['exp3_learning_log'] = f"exp3_learning_seed_{seed}.json"
        config_dict['exp3_final_model'] = f"exp3_model_seed_{seed}.json"
        
        # 각 시드별 개별 폴더 생성
        seed_output_dir = os.path.join(self.experiment_dir, f"seed_{seed}")
        os.makedirs(seed_output_dir, exist_ok=True)
        
        # 임시 설정 파일 생성
        temp_config_path = os.path.join(seed_output_dir, f"config_seed_{seed}.json")
        with open(temp_config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # 시뮬레이션 실행
        start_time = time.time()
        process = subprocess.run(
            ["python", "run_kiss.py", "-c", temp_config_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()  # 현재 작업 디렉토리 명시
        )
        
        if process.returncode != 0:
            print(f"[Seed {seed}] Error: {process.stderr}")
            return None
            
        execution_time = time.time() - start_time
        print(f"[Seed {seed}] Completed in {execution_time:.2f} seconds")
        
        # 결과 파일 찾기 - 더 유연한 방법
        result_files = self._find_result_files(config_dict, seed)
        
        if not result_files['learning_log']:
            print(f"[Seed {seed}] Warning: Learning log not found, searching alternative locations...")
            result_files = self._find_result_files_alternative(config_dict, seed)
        
        if not result_files['learning_log']:
            print(f"[Seed {seed}] Error: Could not find learning log file")
            return None
        
        # 결과 로드
        try:
            with open(result_files['learning_log'], 'r') as f:
                learning_data = json.load(f)
                
            model_data = {}
            if result_files['model'] and os.path.exists(result_files['model']):
                with open(result_files['model'], 'r') as f:
                    model_data = json.load(f)
                    
            print(f"[Seed {seed}] Successfully loaded results")
            
            return {
                'seed': seed,
                'learning_data': learning_data,
                'model_data': model_data,
                'execution_time': execution_time,
                'result_files': result_files
            }
            
        except Exception as e:
            print(f"[Seed {seed}] Error loading results: {e}")
            return None
    
    def _find_result_files(self, config: dict, seed: int) -> dict:
        """결과 파일 찾기 (기본 방법)"""
        exp_desc = config.get('experiment_description')
        base_path = os.path.join(self.output_base_dir, exp_desc)
        
        learning_log_name = config.get('exp3_learning_log', f'exp3_learning_seed_{seed}.json')
        model_name = config.get('exp3_final_model', f'exp3_model_seed_{seed}.json')
        
        result_files = {
            'learning_log': None,
            'model': None
        }
        
        # 날짜/시간 폴더 구조 탐색
        if os.path.exists(base_path):
            # 가장 최근 날짜 폴더 찾기
            date_dirs = sorted([d for d in os.listdir(base_path) 
                              if os.path.isdir(os.path.join(base_path, d))])
            
            if date_dirs:
                latest_date = date_dirs[-1]
                date_path = os.path.join(base_path, latest_date)
                
                # 시간 폴더가 있는지 확인
                time_dirs = sorted([d for d in os.listdir(date_path) 
                                  if os.path.isdir(os.path.join(date_path, d))])
                
                if time_dirs:
                    # 가장 최근 시간 폴더
                    search_path = os.path.join(date_path, time_dirs[-1])
                else:
                    # 날짜 폴더 직접 사용
                    search_path = date_path
                
                # 파일 찾기
                learning_log_path = os.path.join(search_path, learning_log_name)
                model_path = os.path.join(search_path, model_name)
                
                if os.path.exists(learning_log_path):
                    result_files['learning_log'] = learning_log_path
                if os.path.exists(model_path):
                    result_files['model'] = model_path
        
        return result_files
    
    def _find_result_files_alternative(self, config: dict, seed: int) -> dict:
        """결과 파일 찾기 (대체 방법)"""
        result_files = {
            'learning_log': None,
            'model': None
        }
        
        # 전체 출력 디렉토리에서 검색
        search_patterns = [
            f"**/exp3_learning*seed_{seed}.json",
            f"**/exp3_learning_seed_{seed}.json",
            f"**/exp3_learning_progress*seed_{seed}.json",
            f"**/*seed_{seed}*learning*.json"
        ]
        
        for pattern in search_patterns:
            files = glob.glob(os.path.join(self.output_base_dir, pattern), recursive=True)
            if files:
                # 가장 최근 파일 선택
                files.sort(key=os.path.getmtime)
                result_files['learning_log'] = files[-1]
                print(f"[Seed {seed}] Found learning log: {result_files['learning_log']}")
                break
        
        # 모델 파일도 비슷하게 검색
        if result_files['learning_log']:
            model_patterns = [
                f"**/exp3_model*seed_{seed}.json",
                f"**/exp3_trained*seed_{seed}.json",
                f"**/*seed_{seed}*model*.json"
            ]
            
            for pattern in model_patterns:
                files = glob.glob(os.path.join(self.output_base_dir, pattern), recursive=True)
                if files:
                    files.sort(key=os.path.getmtime)
                    result_files['model'] = files[-1]
                    break
        
        return result_files
    
    def run_all_seeds_parallel(self):
        """모든 시드를 병렬로 실행"""
        print(f"\n{'='*80}")
        print(f"Starting parallel multi-seed evaluation")
        print(f"Seeds: {self.n_seeds}, Parallel processes: {self.n_parallel}")
        print(f"Output directory: {self.experiment_dir}")
        print(f"{'='*80}")
        
        # 기본 설정 로드
        with open(self.base_config_path, 'r') as f:
            base_config = json.load(f)
        
        # 각 시드에 대한 설정 준비
        seed_configs = [(seed, base_config.copy()) for seed in range(1, self.n_seeds + 1)]
        
        # 병렬 실행
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
            # 작업 제출
            future_to_seed = {
                executor.submit(self.run_single_seed, seed, config): seed 
                for seed, config in seed_configs
            }
            
            # 결과 수집
            for future in concurrent.futures.as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    result = future.result()
                    if result:
                        self.all_results.append(result)
                        self.seed_results[seed] = result
                        print(f"[Seed {seed}] ✅ Successfully processed")
                    else:
                        print(f"[Seed {seed}] ❌ Failed to process")
                except Exception as e:
                    print(f"[Seed {seed}] ❌ Exception: {e}")
        
        print(f"\n{'='*80}")
        print(f"Completed {len(self.all_results)}/{self.n_seeds} experiments successfully")
        print(f"{'='*80}")
    
    def calculate_metrics(self):
        """평가 지표 계산 (기존 코드와 동일)"""
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
            if rewards:
                cumulative_reward = np.cumsum(rewards)
                metrics['cumulative_rewards'].append(cumulative_reward)
            
            # 에너지 통계
            energy_stats = learning_data.get('energy_statistics', {})
            if energy_stats:
                energy_saving = energy_stats.get('avg_energy_saving_all_on', 0)
                metrics['energy_savings'].append(energy_saving)
            
            # Throughput 통계
            throughput_stats = learning_data.get('throughput_statistics', {})
            if throughput_stats:
                avg_throughput = throughput_stats.get('avg_cell_throughput_gbps', 0)
                metrics['average_throughputs'].append(avg_throughput)
            
            # 전환 비용
            switching_rate = learning_data.get('switching_rate', 0)
            metrics['switching_costs'].append(switching_rate * 100)
            
            # 수렴 에피소드
            convergence_episode = learning_data.get('convergence_episode', -1)
            metrics['convergence_episodes'].append(convergence_episode)
            
            # Regret 정보
            regret_stats = learning_data.get('regret_statistics', {})
            if regret_stats:
                metrics['cumulative_regrets'].append(regret_stats.get('cumulative_regret', 0))
                metrics['average_regrets'].append(regret_stats.get('average_regret', 0))
            
            # 효율성
            max_efficiency = learning_data.get('max_efficiency', 0)
            metrics['final_efficiencies'].append(max_efficiency)
            
            # 최적 arm
            if model_data and 'weights' in model_data:
                weights = np.array(model_data['weights'])
                best_arm = np.argmax(weights) if len(weights) > 0 else -1
                metrics['best_arm_selections'].append(best_arm)
        
        return metrics
    
    # 나머지 메서드들 (generate_plots, save_summary_report 등)은 기존 코드와 동일
    def generate_plots(self, metrics: Dict):
        """평가 결과 플롯 생성"""
        # 기존 EXP3MultiSeedEvaluator의 generate_plots 메서드와 동일
        # (코드 길이 관계로 생략 - 기존 코드 복사)
        pass
    
    def save_summary_report(self, metrics: Dict):
        """종합 보고서 저장"""
        # 기존 EXP3MultiSeedEvaluator의 save_summary_report 메서드와 동일
        # (코드 길이 관계로 생략 - 기존 코드 복사)
        pass


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='EXP3 Multi-seed Parallel Evaluation')
    parser.add_argument('-c', '--config', type=str, required=True,
                      help='Base configuration file path')
    parser.add_argument('-n', '--n-seeds', type=int, default=10,
                      help='Number of seeds to run (default: 10)')
    parser.add_argument('-p', '--n-parallel', type=int, default=None,
                      help='Number of parallel processes (default: CPU count - 1)')
    parser.add_argument('-o', '--output-dir', type=str, default='_/data/output',
                      help='Output directory (default: _/data/output)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode (run only 1 seed)')
    
    args = parser.parse_args()
    
    if args.debug:
        print("🐛 Debug mode: Running only 1 seed")
        args.n_seeds = 1
    
    # 평가기 생성 및 실행
    evaluator = EXP3MultiSeedEvaluatorParallel(
        base_config_path=args.config,
        n_seeds=args.n_seeds,
        output_base_dir=args.output_dir,
        n_parallel=args.n_parallel
    )
    
    # 병렬로 모든 시드 실행
    evaluator.run_all_seeds_parallel()
    
    # 메트릭 계산
    metrics = evaluator.calculate_metrics()
    
    if metrics:
        # 플롯 생성
        evaluator.generate_plots(metrics)
        
        # 보고서 저장
        evaluator.save_summary_report(metrics)
        
        print(f"\n{'='*80}")
        print(f"✅ Evaluation complete!")
        print(f"📁 Results saved in: {evaluator.experiment_dir}")
        print(f"{'='*80}")
    else:
        print("❌ No metrics to analyze. Please check the experiment results.")


if __name__ == "__main__":
    main()
