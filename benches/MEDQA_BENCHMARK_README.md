# MedQA Benchmark for HospitalSim

This directory contains comprehensive benchmarking tools for evaluating the HospitalSim system on medical question-answering tasks using the MedQA dataset format.

## Overview

The MedQA benchmark evaluates the hospital simulation's medical agents (doctors, nurses) on their ability to answer USMLE-style medical questions accurately. This provides a standardized way to assess the medical knowledge and reasoning capabilities of the AI agents in the hospital simulation.

## Files

- `medqa_benchmark.py` - Core benchmark implementation
- `run_medqa_benchmark.py` - Simple script to run the benchmark
- `advanced_medqa_benchmark.py` - Advanced benchmarking with detailed analysis
- `test_medqa_benchmark.py` - Test script to verify everything works
- `MEDQA_BENCHMARK_README.md` - This documentation

## Quick Start

### 1. Test the Benchmark

First, run the test script to ensure everything is working:

```bash
python test_medqa_benchmark.py
```

### 2. Run Basic Benchmark

Run a simple benchmark with a few questions:

```bash
python run_medqa_benchmark.py
```

### 3. Run Advanced Benchmark

For comprehensive analysis with detailed reports:

```bash
python advanced_medqa_benchmark.py --questions 20 --comparative
```

## Usage Examples

### Basic Usage

```python
from hospital_sim.main import HospitalSimulation
from medqa_benchmark import MedQABenchmark

# Initialize hospital
hospital = HospitalSimulation("My Hospital")

# Create benchmark
benchmark = MedQABenchmark(hospital)

# Run benchmark
results = benchmark.run_benchmark(num_questions=10)

# Print results
benchmark.print_results_summary(results)
```

### Advanced Usage

```python
from advanced_medqa_benchmark import AdvancedMedQABenchmark

# Initialize advanced benchmark
advanced_benchmark = AdvancedMedQABenchmark(hospital)

# Run comprehensive benchmark
results = advanced_benchmark.run_comprehensive_benchmark(
    num_questions=50,
    agents=["Dr. Zara Nightingale", "Dr. Kai Thunderheart"]
)

# Generate detailed report
html_file = advanced_benchmark.save_detailed_report(results)
```

## Command Line Options

### Basic Benchmark (`run_medqa_benchmark.py`)

No command line options - runs with default settings (5 questions).

### Advanced Benchmark (`advanced_medqa_benchmark.py`)

- `--questions N` - Number of questions to test (default: 20)
- `--agents AGENT1 AGENT2` - Specific agents to test (default: all medical agents)
- `--output-dir DIR` - Output directory for reports (default: "benchmark_reports")
- `--comparative` - Run comparative analysis between agents

## Question Types

The benchmark includes questions covering:

- **Cardiology** - Heart conditions, ECG interpretation, heart failure
- **Neurology** - Meningitis, stroke, headaches, neurological symptoms
- **Surgery** - Appendicitis, cholecystitis, surgical emergencies
- **Endocrinology** - Diabetes, thyroid disorders, metabolic conditions
- **Pulmonology** - Lung cancer, respiratory conditions
- **Emergency Medicine** - Acute presentations, triage decisions

## Evaluation Metrics

The benchmark provides comprehensive evaluation metrics:

### Overall Performance
- **Accuracy** - Percentage of correct answers
- **Response Time** - Average time to answer questions
- **Total Questions** - Number of questions processed
- **Correct Answers** - Number of correct responses

### Agent-Specific Metrics
- Individual agent performance
- Confidence levels
- Response times per agent
- Accuracy by medical specialty

### Subject-Specific Analysis
- Performance by medical subject area
- Difficulty level analysis
- Specialized knowledge assessment

## Output Files

### JSON Results
- `medqa_benchmark_results_TIMESTAMP.json` - Detailed results in JSON format
- Contains all question-answer pairs, agent responses, and metrics

### HTML Report
- `medqa_report_TIMESTAMP.html` - Comprehensive visual report
- Includes charts, tables, and detailed analysis
- Can be opened in any web browser

## Customizing Questions

### Adding Your Own Questions

```python
from medqa_benchmark import MedQAQuestion

# Create custom question
custom_question = MedQAQuestion(
    question_id="custom_001",
    question="Your medical question here...",
    options=["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4", "E) Option 5"],
    correct_answer="A) Option 1",
    explanation="Explanation of why this is correct...",
    subject="Cardiology",
    difficulty="Medium"
)

# Use in benchmark
questions = [custom_question] + benchmark.load_sample_questions(10)
results = benchmark.run_benchmark(questions=questions)
```

### Loading from File

```python
# Load questions from JSON file
questions = benchmark.load_questions_from_file("my_questions.json")
results = benchmark.run_benchmark(questions=questions)
```

## Interpreting Results

### Accuracy Scores
- **90%+** - Excellent medical knowledge
- **80-89%** - Good medical knowledge
- **70-79%** - Adequate medical knowledge
- **<70%** - Needs improvement

### Response Times
- **<5 seconds** - Very fast
- **5-15 seconds** - Normal
- **15-30 seconds** - Slow
- **>30 seconds** - Very slow

### Confidence Levels
- **80%+** - High confidence
- **60-79%** - Moderate confidence
- **40-59%** - Low confidence
- **<40%** - Very low confidence

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check that HospitalSim is properly installed

2. **Agent Not Found**
   - Verify agent names match those in the hospital simulation
   - Check that agents are properly initialized

3. **Memory Issues**
   - Reduce number of questions for testing
   - Use smaller question sets initially

4. **Slow Performance**
   - Reduce number of questions
   - Test with fewer agents
   - Check system resources

### Getting Help

If you encounter issues:

1. Run the test script first: `python test_medqa_benchmark.py`
2. Check the logs for error messages
3. Verify your HospitalSim installation
4. Ensure all required dependencies are installed

## Dependencies

- `hospital-sim` - The main hospital simulation package
- `swarms` - For AI agent functionality
- `chromadb` - For EHR system
- `loguru` - For logging
- `python-dotenv` - For environment variables

## License

This benchmark tool is part of the HospitalSim project and follows the same MIT license.

## Contributing

To contribute to the MedQA benchmark:

1. Add new question types or medical specialties
2. Improve evaluation metrics
3. Add new analysis features
4. Optimize performance
5. Fix bugs or issues

## Citation

If you use this MedQA benchmark in your research, please cite the HospitalSim project:

```bibtex
@software{hospitalsim2025,
  title={HospitalSim: Enterprise-Grade Hospital Management \& Simulation System},
  author={The Swarm Corporation},
  year={2025},
  url={https://github.com/The-Swarm-Corporation/HospitalSim},
  license={MIT}
}
```
