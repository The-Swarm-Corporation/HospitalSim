from hospital_sim.main import HospitalSimulation, Patient
from dotenv import load_dotenv

load_dotenv()

# Create hospital
hospital = HospitalSimulation("Digital Health Hospital")

# Create a patient with complex medical history
complex_patient = Patient(
    name="Nebula Starforge",
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
