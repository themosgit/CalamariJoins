#!/usr/bin/env python3
import subprocess
import time
import statistics
import json
import multiprocessing
import re
import platform
import os
from datetime import datetime

QUERY_REGEX = re.compile(r"Query\s*([\w\d]+)\s*>>\s*.*?Runtime:\s*(\d+)\s*ms", re.DOTALL)
TOTAL_REGEX = re.compile(r"Total runtime:\s*(\d+)\s*ms")

def parse_benchmark_output(output):
    """Parse benchmark output and extract query times and total time."""
    query_times = {m.group(1): int(m.group(2)) for m in QUERY_REGEX.finditer(output)}
    total_match = TOTAL_REGEX.search(output)
    total_time = int(total_match.group(1)) if total_match else None
    
    if not query_times or total_time is None:
        print(f"\n[Parsing ERROR] Found {len(query_times)} queries, total_time={total_time}")
        print(f"--- Output ---\n{output}\n---")
        return None, None
    
    return query_times, total_time

def run_command(cmd, cwd):
    """Run a command and return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"\nERROR: Command failed with code {result.returncode}")
        print(f"STDOUT: {result.stdout.strip()}\nSTDERR: {result.stderr.strip()}")
        raise Exception(f"Command failed: {' '.join(cmd)}")
    return result.stdout

def calculate_stats(runs):
    """Calculate statistics for run times."""
    if not runs:
        return {'median': 0, 'average': 0, 'min': 0, 'max': 0, 'std_dev': 0, 'runs': []}
    return {
        'median': statistics.median(runs),
        'average': statistics.mean(runs),
        'min': min(runs),
        'max': max(runs),
        'std_dev': statistics.stdev(runs) if len(runs) > 1 else 0.0,
        'runs': runs
    }

def save_results(executable, results, timestamp, arch, results_dir):
    """Save results to JSON file."""
    report = {
        'executable': executable,
        'timestamp': timestamp,
        'architecture': arch,
        'iterations_measured': len(results['total_times']),
        'raw_data': {
            'total_times': results['total_times'],
            'queries': results['queries']
        },
        'statistics': {
            'total_runtime': calculate_stats(results['total_times']),
            'per_query': {qid: calculate_stats(runs) for qid, runs in results['queries'].items()}
        }
    }
    
    filename = f"results_{executable}_{arch}_{timestamp.replace(':', '-').replace(' ', '_')}.json"

    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved to {filepath}")

def main():
    executables = ['faster']
    project_dir = '.'
    iterations = 10

    RESULTS_DIR = '../project-site/benchmark_results/part2'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    num_cores = str(multiprocessing.cpu_count())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    arch = platform.machine()
    
    results_data = {exec: {'queries': {}, 'total_times': []} for exec in executables}
    
    try:
        print(f"\n{'='*60}")
        print(f"Starting benchmark...")
        print(f"Timestamp: {timestamp}")
        print(f"Architecture: {arch}")
        print(f"Using {num_cores} cores for parallel build.")
        print(f"Running {iterations} measured runs per executable (plus 1 cold run).")
        print(f"{'='*60}")
        
        for executable in executables:
            try:
                # Build
                print(f"\nBuilding executable: {executable}...")
                start = time.time()
                run_command(['cmake', '--build', 'build', '--', '-j', num_cores, executable], project_dir)
                print(f"Build complete in {time.time() - start:.2f}s.")
                
                # Run benchmarks
                print(f"Running {iterations + 1} iterations for {executable} (1 cold + {iterations} measured)...")
                
                for i in range(iterations + 1):
                    is_cold = i == 0
                    label = "[Cold Run]" if is_cold else f"[Run {i}/{iterations}]"
                    print(f"  {label}...", end=' ', flush=True)
                    
                    try:
                        stdout = run_command([f'./build/{executable}', 'plans.json'], project_dir)
                        query_times, total_time = parse_benchmark_output(stdout)
                        
                        if query_times is None or total_time is None:
                            print("SKIPPED (parsing error)")
                            continue
                        
                        print(f"Total: {total_time} ms")
                        
                        if not is_cold:
                            results_data[executable]['total_times'].append(total_time)
                            for qid, time_ms in query_times.items():
                                results_data[executable]['queries'].setdefault(qid, []).append(time_ms)
                    
                    except Exception as e:
                        print(f"SKIPPED ({e})")
                
                # Save results
                print(f"\nSaving results for {executable}...")
                save_results(executable, results_data[executable], timestamp, arch, RESULTS_DIR)
            
            except Exception as e:
                print(f"\nERROR: Failed to benchmark {executable}: {e}")
                print("Skipping and continuing with next executable...")
        
        print(f"\n{'='*60}")
        print("Benchmark complete!")
        print(f"{'='*60}")
    
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")

if __name__ == '__main__':
    main()
