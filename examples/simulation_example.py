from dotenv import load_dotenv

from hospital_sim.main import HospitalSimulation

load_dotenv()


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


