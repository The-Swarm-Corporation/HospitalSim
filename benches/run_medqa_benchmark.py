#!/usr/bin/env python3
"""
Run MedQA Benchmark on HospitalSim

This script runs the MedQA benchmark using the hospital simulation from example.py
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from hospital_sim.main import HospitalSimulation
from medqa_benchmark import MedQABenchmark
from dotenv import load_dotenv

def main():
    """Run the MedQA benchmark on the hospital simulation."""
    # Load environment variables
    load_dotenv()
    
    # Create hospital simulation (same as example.py)
    hospital = HospitalSimulation("City General Hospital")
    
    # Initialize MedQA benchmark
    benchmark = MedQABenchmark(hospital)
    
    # Run benchmark
    num_questions = 5
    
    try:
        results = benchmark.run_benchmark(num_questions=num_questions)
        
        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"medqa_benchmark_results_{timestamp}.json"
        benchmark.save_results(results, results_file)
        
    except Exception as e:
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
