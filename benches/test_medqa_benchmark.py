#!/usr/bin/env python3
"""
Test script for MedQA Benchmark

This script runs a quick test of the MedQA benchmark to ensure everything works correctly.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from hospital_sim.main import HospitalSimulation
from medqa_benchmark import MedQABenchmark
from dotenv import load_dotenv

def test_medqa_benchmark():
    """Test the MedQA benchmark with a small set of questions."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize hospital simulation
        hospital = HospitalSimulation("Test Hospital")
        
        # Initialize benchmark
        benchmark = MedQABenchmark(hospital)
        
        # Load sample questions
        questions = benchmark.load_sample_questions(3)  # Just 3 questions for testing
        
        # Test a single question
        test_question = questions[0]
        
        # Get available agents
        available_agents = [name for name, staff in hospital.staff.items() 
                           if staff.role.value in ['doctor', 'nurse']]
        
        if available_agents:
            test_agent = available_agents[0]
            
            # Run single question
            benchmark.run_question_on_agent(test_question, test_agent)
        
        # Run mini benchmark
        benchmark.run_benchmark(num_questions=2)
        
        return True
        
    except Exception:
        return False

def main():
    """Main test function."""
    success = test_medqa_benchmark()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
