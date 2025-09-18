#!/usr/bin/env python3
"""
Custom Patients Example

This script demonstrates hospital simulation with custom patient creation.
"""

from hospital_sim.main import HospitalSimulation, Patient
from dotenv import load_dotenv

load_dotenv()

# Create hospital
hospital = HospitalSimulation("Specialty Hospital")

# Create custom patients
custom_patients = [
    Patient(
        name="Aurora Dreamweaver",
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
        name="Jasper Stormcloud",
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
        name="Nova Celestine",
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
