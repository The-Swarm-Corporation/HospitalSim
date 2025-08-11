#!/usr/bin/env python3
"""
Hospital Simulation Example

This script demonstrates how to use the hospital simulation system
with different scenarios and configurations.
"""

from hospital_sim import HospitalSimulation, Patient
from dotenv import load_dotenv

load_dotenv()


def basic_simulation_example():
    """Basic hospital simulation example."""
    # Create hospital
    hospital = HospitalSimulation("City General Hospital")

    # Generate initial patients
    hospital.generate_patients(3)

    # Run short simulation
    hospital.run_simulation(
        duration_minutes=10, patient_arrival_rate=0.1
    )


def custom_patient_example():
    """Example with custom patient creation."""
    # Create hospital
    hospital = HospitalSimulation("Specialty Hospital")

    # Create custom patients
    custom_patients = [
        Patient(
            name="Isabella Fernandez",
            age=35,
            gender="Female",
            chief_complaint="Severe migraine",
            symptoms=[
                "headache",
                "nausea",
                "light sensitivity",
                "vomiting",
            ],
            medical_history=["migraines", "anxiety"],
            current_medications=["propranolol"],
            allergies=["aspirin"],
        ),
        Patient(
            name="Amir Hassan",
            age=52,
            gender="Male",
            chief_complaint="Chest tightness",
            symptoms=[
                "chest tightness",
                "shortness of breath",
                "sweating",
            ],
            medical_history=["hypertension", "high cholesterol"],
            current_medications=["lisinopril", "atorvastatin"],
            allergies=["sulfa drugs"],
        ),
        Patient(
            name="Yuki Tanaka",
            age=28,
            gender="Female",
            chief_complaint="Fever and sore throat",
            symptoms=[
                "fever",
                "sore throat",
                "difficulty swallowing",
                "fatigue",
            ],
            medical_history=["tonsillitis"],
            current_medications=[],
            allergies=["penicillin"],
        ),
    ]

    # Add patients to hospital
    for patient in custom_patients:
        hospital.add_patient(patient)

    # Run simulation
    hospital.run_simulation(
        duration_minutes=15, patient_arrival_rate=0.15
    )


def emergency_scenario_example():
    """Example focusing on emergency patient handling."""
    # Create hospital
    hospital = HospitalSimulation("Emergency Medical Center")

    # Create emergency patients
    emergency_patients = [
        Patient(
            name="Rafael Mendoza",
            age=45,
            gender="Male",
            chief_complaint="Chest pain radiating to arm",
            symptoms=[
                "chest pain",
                "left arm pain",
                "shortness of breath",
                "sweating",
            ],
            medical_history=["hypertension", "diabetes"],
            current_medications=["metformin", "amlodipine"],
            allergies=[],
        ),
        Patient(
            name="Fatima Al-Zahra",
            age=67,
            gender="Female",
            chief_complaint="Sudden severe headache",
            symptoms=[
                "sudden severe headache",
                "confusion",
                "weakness",
                "speech difficulty",
            ],
            medical_history=["hypertension", "atrial fibrillation"],
            current_medications=["warfarin", "metoprolol"],
            allergies=["heparin"],
        ),
        Patient(
            name="Kwame Osei",
            age=38,
            gender="Male",
            chief_complaint="Severe abdominal pain",
            symptoms=[
                "severe abdominal pain",
                "nausea",
                "vomiting",
                "fever",
            ],
            medical_history=["appendicitis"],
            current_medications=[],
            allergies=["morphine"],
        ),
    ]

    # Add patients to hospital
    for patient in emergency_patients:
        hospital.add_patient(patient)

    # Run simulation
    hospital.run_simulation(
        duration_minutes=20, patient_arrival_rate=0.2
    )


def ehr_demo_example():
    """Example demonstrating EHR system capabilities."""
    # Create hospital
    hospital = HospitalSimulation("Digital Health Hospital")

    # Create a patient with complex medical history
    complex_patient = Patient(
        name="Elena Rossi",
        age=58,
        gender="Female",
        chief_complaint="Chronic back pain worsening",
        symptoms=[
            "chronic back pain",
            "leg numbness",
            "difficulty walking",
            "bladder issues",
        ],
        medical_history=[
            "herniated disc",
            "diabetes",
            "hypertension",
            "depression",
        ],
        current_medications=[
            "gabapentin",
            "metformin",
            "sertraline",
            "hydrochlorothiazide",
        ],
        allergies=["codeine", "morphine"],
    )

    # Add patient
    hospital.add_patient(complex_patient)

    # Search for similar cases
    similar_cases = hospital.ehr_system.search_similar_cases(
        symptoms=["back pain", "leg numbness"],
        diagnosis="herniated disc",
    )

    # Run simulation to generate EHR data
    hospital.run_simulation(
        duration_minutes=12, patient_arrival_rate=0.1
    )


def interactive_simulation_example():
    """Interactive simulation with user input."""
    # Get user preferences
    hospital_name = input(
        "Enter hospital name (or press Enter for default): "
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

    # Create hospital
    hospital = HospitalSimulation(hospital_name)

    # Generate patients
    hospital.generate_patients(1)

    # Run simulation
    hospital.run_simulation(
        duration_minutes=duration, patient_arrival_rate=arrival_rate
    )


def main():
    """Main function to run examples."""
    examples = [
        ("Basic Simulation", basic_simulation_example),
        ("Custom Patients", custom_patient_example),
        ("Emergency Scenario", emergency_scenario_example),
        ("EHR Demo", ehr_demo_example),
        ("Interactive Simulation", interactive_simulation_example),
    ]

    while True:
        for i, (name, _) in enumerate(examples, 1):
            pass

        try:
            choice = input("Select an example (0-5): ").strip()
            if choice == "0":
                break

            choice_num = int(choice)
            if 1 <= choice_num <= len(examples):
                name, func = examples[choice_num - 1]
                try:
                    func()
                except KeyboardInterrupt:
                    break
                except Exception:
                    pass
            else:
                pass

        except ValueError:
            pass
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
