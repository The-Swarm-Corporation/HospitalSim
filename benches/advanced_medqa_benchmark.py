#!/usr/bin/env python3
"""
Advanced MedQA Benchmark for HospitalSim

This script provides comprehensive benchmarking capabilities including:
- Multiple question sets
- Performance analysis
- Comparative evaluation
- Detailed reporting
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from hospital_sim.main import HospitalSimulation
from medqa_benchmark import MedQABenchmark, MedQAQuestion, MedQABenchmarkResults
from dotenv import load_dotenv


class AdvancedMedQABenchmark:
    """Advanced MedQA benchmarking with multiple test sets and analysis."""
    
    def __init__(self, hospital: HospitalSimulation):
        """Initialize the advanced benchmark."""
        self.hospital = hospital
        self.benchmark = MedQABenchmark(hospital)
        self.results_history: List[MedQABenchmarkResults] = []
    
    def create_comprehensive_question_set(self, num_questions: int = 100) -> List[MedQAQuestion]:
        """Create a comprehensive set of MedQA questions covering all medical specialties."""
        questions = []
        
        # Cardiology questions
        cardiology_questions = [
            MedQAQuestion(
                question_id="card_001",
                question="A 55-year-old man presents with chest pain that started 30 minutes ago. The pain is described as crushing, substernal, and radiating to the left arm and jaw. He has a history of hypertension and smoking. ECG shows ST elevation in leads V1-V4. What is the most likely diagnosis?",
                options=[
                    "A) Unstable angina",
                    "B) Acute anterior myocardial infarction",
                    "C) Pericarditis",
                    "D) Aortic dissection",
                    "E) Pulmonary embolism"
                ],
                correct_answer="B) Acute anterior myocardial infarction",
                explanation="ST elevation in leads V1-V4 indicates anterior wall MI. The crushing chest pain with radiation and risk factors support this diagnosis.",
                subject="Cardiology",
                difficulty="Medium"
            ),
            MedQAQuestion(
                question_id="card_002",
                question="A 70-year-old woman presents with dyspnea on exertion and orthopnea. Physical examination reveals bilateral rales and peripheral edema. Echocardiogram shows an ejection fraction of 30%. What is the most appropriate initial treatment?",
                options=[
                    "A) Digoxin",
                    "B) ACE inhibitor",
                    "C) Beta-blocker",
                    "D) Diuretic",
                    "E) Anticoagulation"
                ],
                correct_answer="B) ACE inhibitor",
                explanation="ACE inhibitors are first-line therapy for heart failure with reduced ejection fraction. They improve survival and reduce hospitalizations.",
                subject="Cardiology",
                difficulty="Medium"
            )
        ]
        
        # Neurology questions
        neurology_questions = [
            MedQAQuestion(
                question_id="neuro_001",
                question="A 30-year-old woman presents with fever, headache, and neck stiffness. Physical examination reveals positive Kernig's and Brudzinski's signs. What is the most appropriate initial diagnostic test?",
                options=[
                    "A) Blood culture",
                    "B) Lumbar puncture",
                    "C) CT scan of head",
                    "D) MRI of brain",
                    "E) EEG"
                ],
                correct_answer="B) Lumbar puncture",
                explanation="The clinical presentation suggests meningitis. Lumbar puncture is the gold standard for diagnosing meningitis.",
                subject="Neurology",
                difficulty="Easy"
            ),
            MedQAQuestion(
                question_id="neuro_002",
                question="A 60-year-old man presents with sudden onset of severe headache described as 'the worst headache of my life.' Physical examination reveals neck stiffness and photophobia. What is the most likely diagnosis?",
                options=[
                    "A) Tension headache",
                    "B) Migraine",
                    "C) Subarachnoid hemorrhage",
                    "D) Cluster headache",
                    "E) Sinusitis"
                ],
                correct_answer="C) Subarachnoid hemorrhage",
                explanation="The sudden onset of severe headache with neck stiffness and photophobia is classic for subarachnoid hemorrhage.",
                subject="Neurology",
                difficulty="Medium"
            )
        ]
        
        # Surgery questions
        surgery_questions = [
            MedQAQuestion(
                question_id="surg_001",
                question="A 25-year-old woman presents with acute onset of severe right lower quadrant pain. Physical examination reveals rebound tenderness and guarding. McBurney's point is tender. What is the most likely diagnosis?",
                options=[
                    "A) Ovarian cyst rupture",
                    "B) Appendicitis",
                    "C) Ectopic pregnancy",
                    "D) Diverticulitis",
                    "E) Kidney stone"
                ],
                correct_answer="B) Appendicitis",
                explanation="The classic presentation of appendicitis includes acute RLQ pain, rebound tenderness, and tenderness at McBurney's point.",
                subject="Surgery",
                difficulty="Easy"
            ),
            MedQAQuestion(
                question_id="surg_002",
                question="A 45-year-old woman presents with severe abdominal pain and vomiting. Physical examination reveals a positive Murphy's sign. What is the most likely diagnosis?",
                options=[
                    "A) Appendicitis",
                    "B) Cholecystitis",
                    "C) Pancreatitis",
                    "D) Peptic ulcer disease",
                    "E) Gastroenteritis"
                ],
                correct_answer="B) Cholecystitis",
                explanation="Murphy's sign (pain on inspiration during palpation of the right upper quadrant) is specific for cholecystitis.",
                subject="Surgery",
                difficulty="Easy"
            )
        ]
        
        # Endocrinology questions
        endo_questions = [
            MedQAQuestion(
                question_id="endo_001",
                question="A 40-year-old woman presents with polyuria, polydipsia, and polyphagia. Random glucose is 350 mg/dL. What is the most appropriate initial treatment?",
                options=[
                    "A) Metformin",
                    "B) Insulin",
                    "C) Sulfonylurea",
                    "D) DPP-4 inhibitor",
                    "E) GLP-1 agonist"
                ],
                correct_answer="B) Insulin",
                explanation="With glucose >300 mg/dL and classic symptoms, this patient likely has type 1 diabetes or severe type 2 diabetes requiring immediate insulin therapy.",
                subject="Endocrinology",
                difficulty="Medium"
            ),
            MedQAQuestion(
                question_id="endo_002",
                question="A 35-year-old woman presents with fatigue, weight gain, and cold intolerance. Physical examination reveals dry skin and bradycardia. TSH is elevated at 15 mIU/L. What is the most likely diagnosis?",
                options=[
                    "A) Hyperthyroidism",
                    "B) Hypothyroidism",
                    "C) Adrenal insufficiency",
                    "D) Diabetes mellitus",
                    "E) Anemia"
                ],
                correct_answer="B) Hypothyroidism",
                explanation="The symptoms of fatigue, weight gain, cold intolerance, dry skin, and bradycardia with elevated TSH are classic for hypothyroidism.",
                subject="Endocrinology",
                difficulty="Easy"
            )
        ]
        
        # Pulmonology questions
        pulm_questions = [
            MedQAQuestion(
                question_id="pulm_001",
                question="A 65-year-old man with a history of smoking presents with hemoptysis and weight loss. Chest X-ray shows a 3cm mass in the right upper lobe. What is the most likely diagnosis?",
                options=[
                    "A) Pneumonia",
                    "B) Tuberculosis",
                    "C) Lung cancer",
                    "D) Pulmonary embolism",
                    "E) Bronchiectasis"
                ],
                correct_answer="C) Lung cancer",
                explanation="The combination of hemoptysis, weight loss, smoking history, and lung mass on imaging strongly suggests lung cancer.",
                subject="Pulmonology",
                difficulty="Easy"
            )
        ]
        
        # Combine all questions
        all_questions = (cardiology_questions + neurology_questions + 
                        surgery_questions + endo_questions + pulm_questions)
        
        # Add more questions to reach the requested number
        while len(all_questions) < num_questions:
            # Duplicate and modify existing questions
            base_question = all_questions[len(all_questions) % len(all_questions)]
            new_question = MedQAQuestion(
                question_id=f"q{len(all_questions) + 1:03d}",
                question=base_question.question,
                options=base_question.options.copy(),
                correct_answer=base_question.correct_answer,
                explanation=base_question.explanation,
                subject=base_question.subject,
                difficulty=base_question.difficulty
            )
            all_questions.append(new_question)
        
        return all_questions[:num_questions]
    
    def run_comprehensive_benchmark(self, 
                                  num_questions: int = 50,
                                  agents: List[str] = None) -> MedQABenchmarkResults:
        """Run a comprehensive benchmark with detailed analysis."""
        # Create comprehensive question set
        questions = self.create_comprehensive_question_set(num_questions)
        
        # Run benchmark
        results = self.benchmark.run_benchmark(questions=questions, agents=agents)
        
        # Store results in history
        self.results_history.append(results)
        
        return results
    
    def run_comparative_analysis(self, 
                               num_questions: int = 30) -> Dict[str, Any]:
        """Run comparative analysis between different agents."""
        # Get all available medical agents
        medical_agents = [name for name, staff in self.hospital.staff.items() 
                         if staff.role.value in ['doctor', 'nurse']]
        
        comparative_results = {}
        
        # Test each agent individually
        for agent_name in medical_agents:
            results = self.benchmark.run_benchmark(
                num_questions=num_questions,
                agents=[agent_name]
            )
            comparative_results[agent_name] = {
                'accuracy': results.accuracy,
                'avg_response_time': results.average_response_time,
                'total_questions': results.total_questions,
                'correct_answers': results.correct_answers
            }
        
        return comparative_results
    
    def generate_detailed_report(self, results: MedQABenchmarkResults) -> str:
        """Generate a detailed HTML report of the benchmark results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MedQA Benchmark Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                .correct {{ color: green; }}
                .incorrect {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 MedQA Benchmark Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Hospital: {self.hospital.hospital_name}</p>
            </div>
            
            <div class="section">
                <h2>Overall Performance</h2>
                <div class="metric">
                    <strong>Total Questions:</strong> {results.total_questions}
                </div>
                <div class="metric">
                    <strong>Correct Answers:</strong> {results.correct_answers}
                </div>
                <div class="metric">
                    <strong>Accuracy:</strong> {results.accuracy:.2%}
                </div>
                <div class="metric">
                    <strong>Avg Response Time:</strong> {results.average_response_time:.2f}s
                </div>
            </div>
            
            <div class="section">
                <h2>Performance by Agent</h2>
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Accuracy</th>
                        <th>Correct/Total</th>
                        <th>Avg Response Time</th>
                        <th>Avg Confidence</th>
                    </tr>
        """
        
        for agent_name, agent_results in results.results_by_agent.items():
            html_content += f"""
                    <tr>
                        <td>{agent_name}</td>
                        <td>{agent_results['accuracy']:.2%}</td>
                        <td>{agent_results['correct_answers']}/{agent_results['total_questions']}</td>
                        <td>{agent_results['average_response_time']:.2f}s</td>
                        <td>{agent_results['average_confidence']:.1f}%</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        if results.results_by_subject:
            html_content += """
            <div class="section">
                <h2>Performance by Subject</h2>
                <table>
                    <tr>
                        <th>Subject</th>
                        <th>Accuracy</th>
                        <th>Correct/Total</th>
                        <th>Avg Response Time</th>
                    </tr>
            """
            
            for subject, subject_results in results.results_by_subject.items():
                html_content += f"""
                        <tr>
                            <td>{subject}</td>
                            <td>{subject_results['accuracy']:.2%}</td>
                            <td>{subject_results['correct_answers']}/{subject_results['total_questions']}</td>
                            <td>{subject_results['average_response_time']:.2f}s</td>
                        </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # Add detailed question results
        html_content += """
            <div class="section">
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Question ID</th>
                        <th>Agent</th>
                        <th>Predicted</th>
                        <th>Correct</th>
                        <th>Result</th>
                        <th>Confidence</th>
                        <th>Response Time</th>
                    </tr>
        """
        
        for result in results.detailed_results[:20]:  # Show first 20 results
            result_class = "correct" if result.is_correct else "incorrect"
            result_text = "✅ Correct" if result.is_correct else "❌ Incorrect"
            
            html_content += f"""
                    <tr>
                        <td>{result.question_id}</td>
                        <td>{result.agent_name}</td>
                        <td>{result.predicted_answer}</td>
                        <td>{result.correct_answer}</td>
                        <td class="{result_class}">{result_text}</td>
                        <td>{result.confidence:.1f}%</td>
                        <td>{result.response_time:.2f}s</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def save_detailed_report(self, results: MedQABenchmarkResults, output_dir: str = "benchmark_reports"):
        """Save detailed report to HTML file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate HTML report
        html_content = self.generate_detailed_report(results)
        
        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = output_path / f"medqa_report_{timestamp}.html"
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        return html_file


def main():
    """Main function for advanced MedQA benchmarking."""
    parser = argparse.ArgumentParser(description="Advanced MedQA Benchmark for HospitalSim")
    parser.add_argument("--questions", type=int, default=20, help="Number of questions to test")
    parser.add_argument("--agents", nargs="+", help="Specific agents to test")
    parser.add_argument("--output-dir", default="benchmark_reports", help="Output directory for reports")
    parser.add_argument("--comparative", action="store_true", help="Run comparative analysis")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize hospital simulation
    hospital = HospitalSimulation("MedQA Benchmark Hospital")
    
    # Initialize advanced benchmark
    advanced_benchmark = AdvancedMedQABenchmark(hospital)
    
    # Run comprehensive benchmark
    results = advanced_benchmark.run_comprehensive_benchmark(
        num_questions=args.questions,
        agents=args.agents
    )
    
    # Run comparative analysis if requested
    if args.comparative:
        comparative_results = advanced_benchmark.run_comparative_analysis(
            num_questions=min(10, args.questions)
        )
    
    # Save detailed report
    html_file = advanced_benchmark.save_detailed_report(results, args.output_dir)
    
    # Save JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = Path(args.output_dir) / f"medqa_results_{timestamp}.json"
    advanced_benchmark.benchmark.save_results(results, str(json_file))
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
