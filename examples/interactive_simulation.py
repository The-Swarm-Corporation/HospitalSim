#!/usr/bin/env python3
"""
Interactive Simulation Example

This script demonstrates interactive simulation with user input.
"""

from hospital_sim.main import HospitalSimulation
from dotenv import load_dotenv

load_dotenv()

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
