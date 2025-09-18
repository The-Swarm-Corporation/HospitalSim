#!/usr/bin/env python3
"""
MedQA Benchmark Integration for HospitalSim

This module provides comprehensive benchmarking capabilities for the HospitalSim
system using the MedQA dataset, which contains USMLE-style medical questions.

The benchmark evaluates the hospital simulation's medical agents (doctors, nurses)
on their ability to answer medical questions accurately.
"""

import json
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from hospital_sim.main import HospitalSimulation, Patient, StaffRole


@dataclass
class MedQAQuestion:
    """Represents a single MedQA question."""
    
    question_id: str
    question: str
    options: List[str]
    correct_answer: str
    explanation: Optional[str] = None
    subject: Optional[str] = None
    difficulty: Optional[str] = None


@dataclass
class MedQAResult:
    """Represents the result of answering a MedQA question."""
    
    question_id: str
    predicted_answer: str
    correct_answer: str
    is_correct: bool
    confidence: float
    reasoning: str
    response_time: float
    agent_name: str
    subject: Optional[str] = None
    difficulty: Optional[str] = None


@dataclass
class MedQABenchmarkResults:
    """Comprehensive results from MedQA benchmarking."""
    
    total_questions: int
    correct_answers: int
    accuracy: float
    average_response_time: float
    results_by_agent: Dict[str, Dict[str, Any]]
    results_by_subject: Dict[str, Dict[str, Any]]
    results_by_difficulty: Dict[str, Dict[str, Any]]
    detailed_results: List[MedQAResult]
    timestamp: datetime = field(default_factory=datetime.now)


class MedQABenchmark:
    """MedQA benchmark implementation for HospitalSim."""
    
    def __init__(self, hospital_sim: HospitalSimulation):
        """Initialize the MedQA benchmark with a hospital simulation instance."""
        self.hospital_sim = hospital_sim
        self.questions: List[MedQAQuestion] = []
        self.results: List[MedQAResult] = []
        
    def load_sample_questions(self, num_questions: int = 50) -> List[MedQAQuestion]:
        """Load sample MedQA questions for testing."""
        # Sample MedQA-style questions covering various medical topics
        sample_questions = [
            MedQAQuestion(
                question_id="q001",
                question="A 45-year-old man presents with chest pain that started 2 hours ago. The pain is described as crushing, substernal, and radiating to the left arm. He has a history of hypertension and diabetes. ECG shows ST elevation in leads II, III, and aVF. What is the most likely diagnosis?",
                options=[
                    "A) Unstable angina",
                    "B) Acute myocardial infarction (inferior wall)",
                    "C) Pericarditis",
                    "D) Aortic dissection",
                    "E) Pulmonary embolism"
                ],
                correct_answer="B) Acute myocardial infarction (inferior wall)",
                explanation="ST elevation in leads II, III, and aVF indicates inferior wall MI. The crushing chest pain with radiation to left arm and risk factors (HTN, DM) support this diagnosis.",
                subject="Cardiology",
                difficulty="Medium"
            ),
            MedQAQuestion(
                question_id="q002",
                question="A 30-year-old woman presents with fever, headache, and neck stiffness. Physical examination reveals positive Kernig's and Brudzinski's signs. What is the most appropriate initial diagnostic test?",
                options=[
                    "A) Blood culture",
                    "B) Lumbar puncture",
                    "C) CT scan of head",
                    "D) MRI of brain",
                    "E) EEG"
                ],
                correct_answer="B) Lumbar puncture",
                explanation="The clinical presentation suggests meningitis. Lumbar puncture is the gold standard for diagnosing meningitis and should be performed immediately after CT scan if needed.",
                subject="Neurology",
                difficulty="Easy"
            ),
            MedQAQuestion(
                question_id="q003",
                question="A 65-year-old man with a history of smoking presents with hemoptysis and weight loss. Chest X-ray shows a 3cm mass in the right upper lobe. What is the most likely diagnosis?",
                options=[
                    "A) Pneumonia",
                    "B) Tuberculosis",
                    "C) Lung cancer",
                    "D) Pulmonary embolism",
                    "E) Bronchiectasis"
                ],
                correct_answer="C) Lung cancer",
                explanation="The combination of hemoptysis, weight loss, smoking history, and lung mass on imaging strongly suggests lung cancer. Further workup with biopsy is needed.",
                subject="Pulmonology",
                difficulty="Easy"
            ),
            MedQAQuestion(
                question_id="q004",
                question="A 25-year-old woman presents with acute onset of severe right lower quadrant pain. Physical examination reveals rebound tenderness and guarding. McBurney's point is tender. What is the most likely diagnosis?",
                options=[
                    "A) Ovarian cyst rupture",
                    "B) Appendicitis",
                    "C) Ectopic pregnancy",
                    "D) Diverticulitis",
                    "E) Kidney stone"
                ],
                correct_answer="B) Appendicitis",
                explanation="The classic presentation of appendicitis includes acute RLQ pain, rebound tenderness, and tenderness at McBurney's point. This is a surgical emergency.",
                subject="Surgery",
                difficulty="Easy"
            ),
            MedQAQuestion(
                question_id="q005",
                question="A 50-year-old man presents with progressive dyspnea and peripheral edema. Echocardiogram shows an ejection fraction of 25%. What is the most appropriate initial treatment?",
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
            ),
            MedQAQuestion(
                question_id="q006",
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
                question_id="q007",
                question="A 60-year-old man presents with sudden onset of severe headache described as 'the worst headache of my life.' Physical examination reveals neck stiffness and photophobia. What is the most likely diagnosis?",
                options=[
                    "A) Tension headache",
                    "B) Migraine",
                    "C) Subarachnoid hemorrhage",
                    "D) Cluster headache",
                    "E) Sinusitis"
                ],
                correct_answer="C) Subarachnoid hemorrhage",
                explanation="The sudden onset of severe headache with neck stiffness and photophobia is classic for subarachnoid hemorrhage. Immediate CT scan and lumbar puncture are needed.",
                subject="Neurology",
                difficulty="Medium"
            ),
            MedQAQuestion(
                question_id="q008",
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
            ),
            MedQAQuestion(
                question_id="q009",
                question="A 70-year-old man presents with acute confusion and agitation. His family reports he was fine yesterday. Physical examination reveals fever and nuchal rigidity. What is the most appropriate initial diagnostic test?",
                options=[
                    "A) Blood glucose",
                    "B) Lumbar puncture",
                    "C) CT scan of head",
                    "D) Chest X-ray",
                    "E) Urinalysis"
                ],
                correct_answer="B) Lumbar puncture",
                explanation="Acute onset of confusion with fever and neck stiffness suggests meningitis. Lumbar puncture should be performed immediately after ruling out increased ICP with CT scan.",
                subject="Neurology",
                difficulty="Medium"
            ),
            MedQAQuestion(
                question_id="q010",
                question="A 45-year-old woman presents with severe abdominal pain and vomiting. Physical examination reveals a positive Murphy's sign. What is the most likely diagnosis?",
                options=[
                    "A) Appendicitis",
                    "B) Cholecystitis",
                    "C) Pancreatitis",
                    "D) Peptic ulcer disease",
                    "E) Gastroenteritis"
                ],
                correct_answer="B) Cholecystitis",
                explanation="Murphy's sign (pain on inspiration during palpation of the right upper quadrant) is specific for cholecystitis. This is a surgical emergency.",
                subject="Surgery",
                difficulty="Easy"
            )
        ]
        
        # Add more questions to reach the requested number
        while len(sample_questions) < num_questions:
            # Duplicate and modify existing questions to create more variety
            base_question = random.choice(sample_questions[:10])
            new_question = MedQAQuestion(
                question_id=f"q{len(sample_questions) + 1:03d}",
                question=base_question.question,
                options=base_question.options.copy(),
                correct_answer=base_question.correct_answer,
                explanation=base_question.explanation,
                subject=base_question.subject,
                difficulty=base_question.difficulty
            )
            sample_questions.append(new_question)
        
        return sample_questions[:num_questions]
    
    def load_questions_from_file(self, file_path: str) -> List[MedQAQuestion]:
        """Load MedQA questions from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            questions = []
            for item in data:
                question = MedQAQuestion(
                    question_id=item.get('id', ''),
                    question=item.get('question', ''),
                    options=item.get('options', []),
                    correct_answer=item.get('correct_answer', ''),
                    explanation=item.get('explanation'),
                    subject=item.get('subject'),
                    difficulty=item.get('difficulty')
                )
                questions.append(question)
            
            return questions
            
        except FileNotFoundError:
            return self.load_sample_questions()
        except Exception as e:
            return self.load_sample_questions()
    
    def create_medical_patient_from_question(self, question: MedQAQuestion) -> Patient:
        """Create a patient based on the medical question for realistic simulation."""
        # Extract key medical information from the question
        question_text = question.question.lower()
        
        # Determine chief complaint based on question content
        if "chest pain" in question_text:
            chief_complaint = "Chest pain"
            symptoms = ["chest pain", "shortness of breath"]
        elif "headache" in question_text:
            chief_complaint = "Headache"
            symptoms = ["headache", "nausea"]
        elif "abdominal pain" in question_text:
            chief_complaint = "Abdominal pain"
            symptoms = ["abdominal pain", "nausea", "vomiting"]
        elif "dyspnea" in question_text or "shortness of breath" in question_text:
            chief_complaint = "Shortness of breath"
            symptoms = ["dyspnea", "fatigue"]
        elif "fever" in question_text:
            chief_complaint = "Fever"
            symptoms = ["fever", "chills"]
        else:
            chief_complaint = "General medical consultation"
            symptoms = ["general weakness"]
        
        # Create patient with relevant medical history
        patient = Patient(
            name=f"Patient_{question.question_id}",
            age=random.randint(25, 75),
            gender=random.choice(["Male", "Female"]),
            chief_complaint=chief_complaint,
            symptoms=symptoms,
            medical_history=["hypertension"] if random.random() > 0.5 else [],
            current_medications=["aspirin"] if random.random() > 0.7 else [],
            allergies=["penicillin"] if random.random() > 0.8 else []
        )
        
        return patient
    
    def run_question_on_agent(self, question: MedQAQuestion, agent_name: str) -> MedQAResult:
        """Run a single MedQA question on a specific hospital agent."""
        start_time = time.time()
        
        try:
            # Get the agent from hospital staff
            if agent_name not in self.hospital_sim.staff:
                raise ValueError(f"Agent {agent_name} not found in hospital staff")
            
            agent = self.hospital_sim.staff[agent_name]
            
            # Create a patient based on the question for context
            patient = self.create_medical_patient_from_question(question)
            
            # Format the question for the medical agent
            formatted_question = f"""
            MEDICAL QUESTION FOR DIAGNOSIS:
            
            {question.question}
            
            Options:
            {chr(10).join(question.options)}
            
            Please provide:
            1. Your answer choice (A, B, C, D, or E)
            2. Your reasoning for this diagnosis
            3. Your confidence level (0-100%)
            
            Consider this as a patient presenting with these symptoms and choose the most appropriate diagnosis.
            """
            
            # Run the question through the agent
            response = agent.agent.run(formatted_question)
            
            # Parse the response to extract answer and reasoning
            predicted_answer, reasoning, confidence = self._parse_agent_response(response)
            
            # Determine if the answer is correct
            is_correct = self._check_answer_correctness(predicted_answer, question.correct_answer)
            
            response_time = time.time() - start_time
            
            result = MedQAResult(
                question_id=question.question_id,
                predicted_answer=predicted_answer,
                correct_answer=question.correct_answer,
                is_correct=is_correct,
                confidence=confidence,
                reasoning=reasoning,
                response_time=response_time,
                agent_name=agent_name,
                subject=question.subject,
                difficulty=question.difficulty
            )
            
            return result
            
        except Exception as e:
            return MedQAResult(
                question_id=question.question_id,
                predicted_answer="ERROR",
                correct_answer=question.correct_answer,
                is_correct=False,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                response_time=time.time() - start_time,
                agent_name=agent_name,
                subject=question.subject,
                difficulty=question.difficulty
            )
    
    def _parse_agent_response(self, response: str) -> Tuple[str, str, float]:
        """Parse agent response to extract answer, reasoning, and confidence."""
        response_lower = response.lower()
        
        # Extract answer choice
        predicted_answer = "UNKNOWN"
        for option in ["A)", "B)", "C)", "D)", "E)"]:
            if option.lower() in response_lower:
                predicted_answer = option
                break
        
        # Extract confidence (look for percentage or number)
        confidence = 50.0  # Default confidence
        import re
        confidence_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        else:
            # Look for confidence in text
            if "high confidence" in response_lower or "very confident" in response_lower:
                confidence = 85.0
            elif "moderate confidence" in response_lower or "somewhat confident" in response_lower:
                confidence = 65.0
            elif "low confidence" in response_lower or "not sure" in response_lower:
                confidence = 35.0
        
        # Use the full response as reasoning
        reasoning = response.strip()
        
        return predicted_answer, reasoning, confidence
    
    def _check_answer_correctness(self, predicted: str, correct: str) -> bool:
        """Check if the predicted answer matches the correct answer."""
        # Extract just the letter from the answer
        predicted_letter = predicted.split(')')[0] if ')' in predicted else predicted
        correct_letter = correct.split(')')[0] if ')' in correct else correct
        
        return predicted_letter.strip() == correct_letter.strip()
    
    def run_benchmark(self, 
                     questions: Optional[List[MedQAQuestion]] = None,
                     agents: Optional[List[str]] = None,
                     num_questions: int = 50) -> MedQABenchmarkResults:
        """Run the complete MedQA benchmark."""
        # Load questions if not provided
        if questions is None:
            questions = self.load_sample_questions(num_questions)
        
        # Use all available medical agents if not specified
        if agents is None:
            agents = [name for name, staff in self.hospital_sim.staff.items() 
                     if staff.role in [StaffRole.DOCTOR, StaffRole.NURSE]]
        
        all_results = []
        
        # Run each question on each agent
        for i, question in enumerate(questions):
            for agent_name in agents:
                result = self.run_question_on_agent(question, agent_name)
                all_results.append(result)
        
        # Calculate comprehensive results
        benchmark_results = self._calculate_benchmark_results(all_results, questions)
        
        logger.info(f"Benchmark completed. Overall accuracy: {benchmark_results.accuracy:.2%}")
        return benchmark_results
    
    def _calculate_benchmark_results(self, results: List[MedQAResult], questions: List[MedQAQuestion]) -> MedQABenchmarkResults:
        """Calculate comprehensive benchmark results."""
        total_questions = len(questions)
        correct_answers = sum(1 for r in results if r.is_correct)
        accuracy = correct_answers / len(results) if results else 0.0
        average_response_time = sum(r.response_time for r in results) / len(results) if results else 0.0
        
        # Results by agent
        results_by_agent = {}
        for agent_name in set(r.agent_name for r in results):
            agent_results = [r for r in results if r.agent_name == agent_name]
            agent_correct = sum(1 for r in agent_results if r.is_correct)
            results_by_agent[agent_name] = {
                "total_questions": len(agent_results),
                "correct_answers": agent_correct,
                "accuracy": agent_correct / len(agent_results) if agent_results else 0.0,
                "average_response_time": sum(r.response_time for r in agent_results) / len(agent_results) if agent_results else 0.0,
                "average_confidence": sum(r.confidence for r in agent_results) / len(agent_results) if agent_results else 0.0
            }
        
        # Results by subject
        results_by_subject = {}
        for subject in set(r.subject for r in results if r.subject):
            subject_results = [r for r in results if r.subject == subject]
            subject_correct = sum(1 for r in subject_results if r.is_correct)
            results_by_subject[subject] = {
                "total_questions": len(subject_results),
                "correct_answers": subject_correct,
                "accuracy": subject_correct / len(subject_results) if subject_results else 0.0,
                "average_response_time": sum(r.response_time for r in subject_results) / len(subject_results) if subject_results else 0.0
            }
        
        # Results by difficulty
        results_by_difficulty = {}
        for difficulty in set(r.difficulty for r in results if r.difficulty):
            difficulty_results = [r for r in results if r.difficulty == difficulty]
            difficulty_correct = sum(1 for r in difficulty_results if r.is_correct)
            results_by_difficulty[difficulty] = {
                "total_questions": len(difficulty_results),
                "correct_answers": difficulty_correct,
                "accuracy": difficulty_correct / len(difficulty_results) if difficulty_results else 0.0,
                "average_response_time": sum(r.response_time for r in difficulty_results) / len(difficulty_results) if difficulty_results else 0.0
            }
        
        return MedQABenchmarkResults(
            total_questions=total_questions,
            correct_answers=correct_answers,
            accuracy=accuracy,
            average_response_time=average_response_time,
            results_by_agent=results_by_agent,
            results_by_subject=results_by_subject,
            results_by_difficulty=results_by_difficulty,
            detailed_results=results
        )
    
    def save_results(self, results: MedQABenchmarkResults, file_path: str):
        """Save benchmark results to a JSON file."""
        results_dict = {
            "timestamp": results.timestamp.isoformat(),
            "total_questions": results.total_questions,
            "correct_answers": results.correct_answers,
            "accuracy": results.accuracy,
            "average_response_time": results.average_response_time,
            "results_by_agent": results.results_by_agent,
            "results_by_subject": results.results_by_subject,
            "results_by_difficulty": results.results_by_difficulty,
            "detailed_results": [
                {
                    "question_id": r.question_id,
                    "predicted_answer": r.predicted_answer,
                    "correct_answer": r.correct_answer,
                    "is_correct": r.is_correct,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "response_time": r.response_time,
                    "agent_name": r.agent_name,
                    "subject": r.subject,
                    "difficulty": r.difficulty
                }
                for r in results.detailed_results
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
    


