#!/usr/bin/env python3
"""
Hospital Simulation Example

This script demonstrates how to use the hospital simulation system
with different scenarios and configurations.
"""

from hospital_sim.main import HospitalSimulation
from dotenv import load_dotenv

load_dotenv()


def basic_simulation_example():
    """Basic hospital simulation example."""
    # Create hospital
    hospital = HospitalSimulation(
        "City General Hospital",
        description="A small hospital in the city of San Francisco",
    )

    # Generate initial patients
    hospital.generate_patients(1)

    # Run short simulation
    hospital.run_simulation(
        duration_minutes=10, patient_arrival_rate=0.1
    )


if __name__ == "__main__":
    basic_simulation_example()
