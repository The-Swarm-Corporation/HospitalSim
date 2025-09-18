#!/usr/bin/env python3
"""
Emergency Scenario Example

This script demonstrates hospital simulation focusing on emergency patient handling.
"""

from hospital_sim.main import HospitalSimulation, Patient
from dotenv import load_dotenv

load_dotenv()

# Create hospital
hospital = HospitalSimulation("Emergency Medical Center")

# Create emergency patients
emergency_patients = [
    Patient(
        name="Phoenix Razorcrest",
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
        name="Sage Whisperwind",
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
        name="Titan Shadowmere",
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
