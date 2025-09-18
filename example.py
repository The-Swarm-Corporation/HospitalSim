#!/usr/bin/env python3
"""
HospitalSim Example - Improved Agent Behavior

This script demonstrates the enhanced HospitalSim with:
- No future prediction - agents work only with current information
- Progressive information disclosure - natural conversation flow
- Human uncertainty and realistic behavior - hesitation and clarification
- Strict information isolation - agents only know what they should know
- Fast performance with random model selection

Examples include:
1. Basic simulation with improved agents
2. Custom patient scenarios
3. Emergency patient handling
4. EHR system demonstration
5. Interactive simulation
"""

import os
from hospital_sim.main import HospitalSimulation, Patient
from dotenv import load_dotenv

load_dotenv()


def check_api_key():
    """Check if OpenAI API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file with:")
        print("OPENAI_API_KEY=your-api-key-here")
        return False
    return True


def basic_simulation_example():
    """Basic hospital simulation with improved agent behavior."""
    print("Basic HospitalSim Example")
    print("=" * 40)
    print("* Using random models for fast performance")
    print("* No future prediction - realistic agent behavior")
    print("* Progressive information disclosure")
    print("* Human uncertainty and hesitation")
    print("=" * 40)
    
    # Create hospital
    hospital = HospitalSimulation(
        hospital_name="City General Hospital",
        description="A small hospital with realistic AI agents"
    )

    # Generate initial patients
    print("\nGenerating sample patients...")
    hospital.generate_patients(3)
    print(f"OK {hospital.simulation_stats['total_patients']} patients added")

    # Run short simulation
    print("\nStarting simulation...")
    hospital.run_simulation(
        duration_minutes=10, patient_arrival_rate=0.1
    )


def custom_patient_example():
    """Example with custom patient creation and realistic interactions."""
    print("Custom Patient Example")
    print("=" * 40)
    print("Creating patients with specific medical scenarios")
    print("Watch how agents handle different conditions realistically")
    print("=" * 40)
    
    # Create hospital
    hospital = HospitalSimulation("Specialty Medical Center")

    # Create custom patients with realistic scenarios
    custom_patients = [
        Patient(
            name="Aurora Dreamweaver",
            age=35,
            gender="Female",
            chief_complaint="Severe migraine with aura",
            symptoms=[
                "intense headache",
                "visual disturbances",
                "nausea",
                "light sensitivity",
                "vomiting",
            ],
            medical_history=["migraines", "anxiety", "depression"],
            current_medications=["propranolol", "escitalopram"],
            allergies=["aspirin", "ibuprofen"],
            system_prompt="You are Aurora Dreamweaver, a 35-year-old woman experiencing a severe migraine with visual disturbances. You have a history of migraines but this one feels different and more intense. The light is extremely bothersome and you feel nauseous. You're concerned about the visual symptoms. Be cooperative with medical staff and describe your symptoms when asked. Only respond to what medical staff actually say to you.",
        ),
        Patient(
            name="Jasper Stormcloud",
            age=52,
            gender="Male",
            chief_complaint="Chest tightness and pressure",
            symptoms=[
                "chest tightness",
                "pressure in chest",
                "shortness of breath",
                "sweating",
                "anxiety",
            ],
            medical_history=["hypertension", "high cholesterol", "diabetes"],
            current_medications=["lisinopril", "atorvastatin", "metformin"],
            allergies=["sulfa drugs", "penicillin"],
            system_prompt="You are Jasper Stormcloud, a 52-year-old man experiencing chest tightness and pressure. You have a history of heart problems and you're very concerned this might be serious. You're sweating and feeling anxious. Be cooperative with medical staff and describe your symptoms when asked. Only respond to what medical staff actually say to you.",
        ),
        Patient(
            name="Nova Celestine",
            age=28,
            gender="Female",
            chief_complaint="High fever and severe sore throat",
            symptoms=[
                "high fever",
                "severe sore throat",
                "difficulty swallowing",
                "fatigue",
                "body aches",
            ],
            medical_history=["tonsillitis", "strep throat"],
            current_medications=[],
            allergies=["penicillin", "amoxicillin"],
            system_prompt="You are Nova Celestine, a 28-year-old woman with a high fever and extremely sore throat. You can barely swallow and you're very tired. You have a history of throat infections and you're concerned this might be strep throat again. Be cooperative with medical staff and describe your symptoms when asked. Only respond to what medical staff actually say to you.",
        ),
    ]

    # Add patients to hospital
    print("\nAdding custom patients...")
    for patient in custom_patients:
        hospital.add_patient(patient)
        print(f"OK Added {patient.name} - {patient.chief_complaint}")

    # Run simulation
    print("\nStarting simulation with custom patients...")
    hospital.run_simulation(
        duration_minutes=15, patient_arrival_rate=0.15
    )


def emergency_scenario_example():
    """Example focusing on emergency patient handling with realistic urgency."""
    print("Emergency Scenario Example")
    print("=" * 40)
    print("Simulating emergency patients with high priority")
    print("Watch how triage and emergency staff respond")
    print("=" * 40)
    
    # Create hospital
    hospital = HospitalSimulation("Emergency Medical Center")

    # Create emergency patients with realistic emergency scenarios
    emergency_patients = [
        Patient(
            name="Phoenix Razorcrest",
            age=45,
            gender="Male",
            chief_complaint="Chest pain radiating to left arm",
            symptoms=[
                "severe chest pain",
                "pain radiating to left arm",
                "shortness of breath",
                "sweating",
                "nausea",
                "anxiety",
            ],
            medical_history=["hypertension", "diabetes", "smoking"],
            current_medications=["metformin", "amlodipine", "aspirin"],
            allergies=[],
            system_prompt="You are Phoenix Razorcrest, a 45-year-old man experiencing severe chest pain that radiates to your left arm. You're very scared this might be a heart attack. You're sweating profusely and having trouble breathing. Be cooperative with medical staff and describe your symptoms when asked. Only respond to what medical staff actually say to you.",
        ),
        Patient(
            name="Sage Whisperwind",
            age=67,
            gender="Female",
            chief_complaint="Sudden severe headache and confusion",
            symptoms=[
                "sudden severe headache",
                "confusion",
                "weakness on right side",
                "speech difficulty",
                "vision problems",
            ],
            medical_history=["hypertension", "atrial fibrillation", "stroke"],
            current_medications=["warfarin", "metoprolol", "lisinopril"],
            allergies=["heparin"],
            system_prompt="You are Sage Whisperwind, a 67-year-old woman who suddenly developed a severe headache and confusion. You're having trouble speaking clearly and your right side feels weak. You're very frightened and your family is extremely concerned. Be cooperative with medical staff and describe your symptoms when asked. Only respond to what medical staff actually say to you.",
        ),
        Patient(
            name="Titan Shadowmere",
            age=38,
            gender="Male",
            chief_complaint="Severe abdominal pain and vomiting",
            symptoms=[
                "severe abdominal pain",
                "nausea",
                "vomiting",
                "fever",
                "loss of appetite",
            ],
            medical_history=["appendicitis", "diverticulitis"],
            current_medications=[],
            allergies=["morphine", "codeine"],
            system_prompt="You are Titan Shadowmere, a 38-year-old man with severe abdominal pain that started suddenly. You've been vomiting and have a fever. The pain is getting worse and you're very concerned. Be cooperative with medical staff and describe your symptoms when asked. Only respond to what medical staff actually say to you.",
        ),
    ]

    # Add patients to hospital
    print("\nAdding emergency patients...")
    for patient in emergency_patients:
        hospital.add_patient(patient)
        print(f"OK Added {patient.name} - Priority: {patient.priority_score}")

    # Run simulation
    print("\nStarting emergency simulation...")
    hospital.run_simulation(
        duration_minutes=20, patient_arrival_rate=0.2
    )


def ehr_demo_example():
    """Example demonstrating EHR system capabilities with realistic medical records."""
    print("EHR System Demo")
    print("=" * 40)
    print("Demonstrating Electronic Health Record system")
    print("Searching for similar cases and medical history")
    print("=" * 40)
    
    # Create hospital
    hospital = HospitalSimulation("Digital Health Hospital")

    # Create a patient with complex medical history
    complex_patient = Patient(
        name="Nebula Starforge",
        age=58,
        gender="Female",
        chief_complaint="Chronic back pain with new symptoms",
        symptoms=[
            "chronic back pain",
            "leg numbness",
            "difficulty walking",
            "bladder control issues",
            "weakness in legs",
        ],
        medical_history=[
            "herniated disc L4-L5",
            "diabetes type 2",
            "hypertension",
            "depression",
            "osteoporosis",
        ],
        current_medications=[
            "gabapentin",
            "metformin",
            "sertraline",
            "hydrochlorothiazide",
            "calcium supplements",
        ],
        allergies=["codeine", "morphine", "tramadol"],
        system_prompt="You are Nebula Starforge, a 58-year-old woman with chronic back pain that has recently gotten worse. You're experiencing new symptoms including leg numbness and bladder control issues. You're very worried about these new symptoms. Be cooperative with medical staff and describe your symptoms when asked. Only respond to what medical staff actually say to you.",
    )

    # Add patient
    print("\nAdding complex patient...")
    hospital.add_patient(complex_patient)
    print(f"OK Added {complex_patient.name}")

    # Search for similar cases
    print("\nSearching for similar cases in EHR...")
    similar_cases = hospital.ehr_system.search_similar_cases(
        symptoms=["back pain", "leg numbness", "bladder issues"],
        diagnosis="herniated disc",
    )
    
    if similar_cases:
        print(f"OK Found {len(similar_cases)} similar cases")
        for i, case in enumerate(similar_cases[:2], 1):
            print(f"   Case {i}: {case['metadata'].get('patient_name', 'Unknown')}")
    else:
        print("INFO No similar cases found (normal for new EHR system)")

    # Run simulation to generate EHR data
    print("\nRunning simulation to generate EHR data...")
    hospital.run_simulation(
        duration_minutes=12, patient_arrival_rate=0.1
    )


def interactive_simulation_example():
    """Interactive simulation with user input and customization."""
    print("Interactive Simulation")
    print("=" * 40)
    print("Customize your hospital simulation")
    print("=" * 40)
    
    # Get user preferences
    hospital_name = input(
        "Enter hospital name (or press Enter for 'Interactive Hospital'): "
    ).strip()
    if not hospital_name:
        hospital_name = "Interactive Hospital"

    try:
        duration = int(
            input(
                "Enter simulation duration in minutes (default 20): "
            )
            or "20"
        )
    except ValueError:
        duration = 20

    try:
        arrival_rate = float(
            input(
                "Enter patient arrival rate per minute (default 0.15): "
            )
            or "0.15"
        )
    except ValueError:
        arrival_rate = 0.15

    try:
        num_patients = int(
            input(
                "Enter number of initial patients (default 2): "
            )
            or "2"
        )
    except ValueError:
        num_patients = 2

    # Create hospital
    print(f"\nCreating {hospital_name}...")
    hospital = HospitalSimulation(hospital_name)

    # Generate patients
    print(f"\nGenerating {num_patients} initial patients...")
    hospital.generate_patients(num_patients)

    # Run simulation
    print(f"\nStarting {duration}-minute simulation...")
    print(f"Patient arrival rate: {arrival_rate} per minute")
    hospital.run_simulation(
        duration_minutes=duration, patient_arrival_rate=arrival_rate
    )


def main():
    """Main function to run examples with improved interface."""
    if not check_api_key():
        return
    
    print("HospitalSim Examples - Improved Agent Behavior")
    print("=" * 60)
    print("Choose an example to run:")
    print("1. Basic Simulation - Simple hospital scenario")
    print("2. Custom Patients - Specific medical conditions")
    print("3. Emergency Scenario - High-priority patients")
    print("4. EHR Demo - Electronic Health Record system")
    print("5. Interactive Simulation - Customize your own")
    print("0. Exit")
    print("=" * 60)
    
    examples = [
        ("Basic Simulation", basic_simulation_example),
        ("Custom Patients", custom_patient_example),
        ("Emergency Scenario", emergency_scenario_example),
        ("EHR Demo", ehr_demo_example),
        ("Interactive Simulation", interactive_simulation_example),
    ]

    while True:
        try:
            choice = input("\nSelect an example (0-5): ").strip()
            if choice == "0":
                print("Goodbye!")
                break

            choice_num = int(choice)
            if 1 <= choice_num <= len(examples):
                name, func = examples[choice_num - 1]
                print(f"\nRunning: {name}")
                print("-" * 40)
                try:
                    func()
                except KeyboardInterrupt:
                    print("\nExample stopped by user")
                except Exception as e:
                    print(f"\nError running example: {e}")
                
                input("\nPress Enter to continue...")
            else:
                print("Invalid choice. Please select 0-5.")

        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
