from hospital_sim.main import HospitalSimulation
from dotenv import load_dotenv

load_dotenv()

# Create hospital
hospital = HospitalSimulation("City General Hospital")

# Generate initial patients
hospital.generate_patients(3)

# Run short simulation
hospital.run_simulation(
    duration_minutes=10, patient_arrival_rate=0.1
)
