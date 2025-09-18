
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import PriorityQueue
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from loguru import logger
from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

CHROMADB_AVAILABLE = True


class PatientStatus(Enum):
    """Patient status enumeration."""

    WAITING = "waiting"
    IN_TRIAGE = "in_triage"
    WITH_DOCTOR = "with_doctor"
    TREATED = "treated"
    DISCHARGED = "discharged"
    EMERGENCY = "emergency"


class StaffRole(Enum):
    """Hospital staff role enumeration."""

    EXECUTIVE = "executive"
    DOCTOR = "doctor"
    NURSE = "nurse"
    RECEPTIONIST = "receptionist"
    SPECIALIST = "specialist"


@dataclass
class Patient:
    """Patient model with medical information and agent capabilities."""

    patient_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    age: int = 0
    gender: str = ""
    chief_complaint: str = ""
    symptoms: List[str] = field(default_factory=list)
    medical_history: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    vital_signs: Dict[str, Any] = field(default_factory=dict)
    status: PatientStatus = PatientStatus.WAITING
    arrival_time: datetime = field(default_factory=datetime.now)
    priority_score: int = 0
    assigned_doctor: Optional[str] = None
    assigned_nurse: Optional[str] = None
    diagnosis: Optional[str] = None
    treatment_plan: Optional[str] = None
    discharge_notes: Optional[str] = None
    system_prompt: str = ""
    agent: Optional[Agent] = field(default=None, init=False)

    def __post_init__(self):
        """Calculate priority score based on symptoms and vital signs and create agent."""
        logger.info(
            f"Creating patient: {self.name}, Age: {self.age}, Chief Complaint: {self.chief_complaint}"
        )
        self.calculate_priority()
        self._create_agent()
        logger.debug(
            f"Patient {self.name} initialized with priority score: {self.priority_score}"
        )

    def calculate_priority(self):
        """Calculate patient priority score (higher = more urgent)."""
        score = 0

        # Emergency symptoms
        emergency_symptoms = [
            "chest pain",
            "shortness of breath",
            "severe bleeding",
            "unconscious",
            "seizure",
            "stroke symptoms",
            "trauma",
        ]

        for symptom in self.symptoms:
            if any(
                emergency in symptom.lower()
                for emergency in emergency_symptoms
            ):
                score += 10
            elif "pain" in symptom.lower():
                score += 5
            elif "fever" in symptom.lower():
                score += 3

        # Vital signs priority
        if self.vital_signs:
            if (
                self.vital_signs.get("blood_pressure", {}).get(
                    "systolic", 0
                )
                > 180
            ):
                score += 8
            if self.vital_signs.get("heart_rate", 0) > 120:
                score += 6
            if self.vital_signs.get("temperature", 0) > 103:
                score += 7

        self.priority_score = min(score, 20)  # Cap at 20
        logger.debug(
            f"Patient {self.name} priority calculated: {self.priority_score} (Emergency: {score >= 15})"
        )

    def _create_agent(self):
        """Create an Agent instance from patient data."""
        if not self.system_prompt:
            self.system_prompt = (
                self._generate_default_system_prompt()
            )

        self.agent = Agent(
            agent_name=f"Patient_{self.name.replace(' ', '_')}",
            system_prompt=self.system_prompt,
            random_models_on=True,
            max_loops=1,
        )
        logger.debug(f"Patient agent created for {self.name}")

    def _generate_default_system_prompt(self) -> str:
        """Generate a default system prompt for the patient agent."""
        return f"""You are {self.name}, a {self.age}-year-old {self.gender.lower()} patient who has come to the hospital seeking medical care.
        
        Your medical information:
        - Chief Complaint: {self.chief_complaint}
        - Current Symptoms: {', '.join(self.symptoms) if self.symptoms else 'None reported'}
        - Medical History: {', '.join(self.medical_history) if self.medical_history else 'No significant history'}
        - Current Medications: {', '.join(self.current_medications) if self.current_medications else 'None'}
        - Known Allergies: {', '.join(self.allergies) if self.allergies else 'No known allergies'}
        
        CONVERSATION GUIDELINES:
        - NEVER predict, assume, or speak for what medical staff will say or do
        - Only respond to what has actually been said to you
        - Answer questions honestly and directly when asked
        - Focus on describing your own symptoms and concerns
        - Let medical staff speak for themselves in their own words
        
        INTERACTION GUIDELINES:
        - Respond naturally and conversationally to healthcare staff
        - Answer questions directly and specifically when asked
        - Volunteer relevant information about your symptoms and concerns
        - Express your pain levels on a 1-10 scale when asked
        - Share how long you've been experiencing symptoms
        - Mention what makes your symptoms better or worse
        - Express your emotions (worry, fear, hope, relief) appropriately
        - Ask questions about your condition and treatment when appropriate
        - Be honest about your medical history and current medications
        
        PERSONALITY & COMMUNICATION:
        - Be cooperative and polite with medical staff
        - Show realistic emotional responses to your condition severity
        - Express urgency if your condition is serious
        - Thank staff for their care and attention
        - Use natural, everyday language rather than medical terminology
        
        CONSISTENCY:
        - Always stay true to your symptoms and medical history
        - If you're in pain, express it consistently throughout interactions
        - React appropriately to medical procedures (taking vital signs, examinations)
        - Remember what you've already told previous staff members
        
        Remember: You are seeking help and want to get better. Engage authentically with the medical team. Speak only for yourself."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert patient to dictionary for storage."""
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "chief_complaint": self.chief_complaint,
            "symptoms": self.symptoms,
            "medical_history": self.medical_history,
            "current_medications": self.current_medications,
            "allergies": self.allergies,
            "vital_signs": self.vital_signs,
            "status": self.status.value,
            "arrival_time": self.arrival_time.isoformat(),
            "priority_score": self.priority_score,
            "assigned_doctor": self.assigned_doctor,
            "assigned_nurse": self.assigned_nurse,
            "diagnosis": self.diagnosis,
            "treatment_plan": self.treatment_plan,
            "discharge_notes": self.discharge_notes,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Patient":
        """Create patient from dictionary."""
        data["status"] = PatientStatus(data["status"])
        data["arrival_time"] = datetime.fromisoformat(
            data["arrival_time"]
        )
        # Ensure system_prompt is included, set default if missing
        if "system_prompt" not in data:
            data["system_prompt"] = ""
        return cls(**data)


class EHRSystem:
    """Electronic Health Record system using ChromaDB for RAG."""

    def __init__(
        self,
        collection_name: str = "hospital_ehr",
        persist_directory: str = "./hospital_data",
    ):
        """Initialize the EHR system."""
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        if CHROMADB_AVAILABLE:
            self._initialize_chromadb()
        else:
            print(
                "Warning: Using in-memory storage instead of ChromaDB"
            )
            self._initialize_memory_storage()

    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            # Create or get collection
            try:
                self.collection = self.client.get_collection(
                    self.collection_name
                )
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": (
                            "Hospital Electronic Health Records"
                        )
                    },
                )

            logger.info(
                f"EHR System initialized with ChromaDB collection: {self.collection_name}"
            )
            print(
                f"✅ EHR System initialized with ChromaDB collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            print(f"Error initializing ChromaDB: {e}")
            self._initialize_memory_storage()

    def _initialize_memory_storage(self):
        """Initialize in-memory storage as fallback."""
        self.memory_storage = {}
        logger.info("EHR System initialized with in-memory storage")
        print("✅ EHR System initialized with in-memory storage")

    def add_patient_record(
        self, patient: Patient, medical_notes: str, doctor_name: str
    ) -> str:
        """Add patient medical record to EHR."""
        record_id = f"record_{patient.patient_id}_{int(time.time())}"
        logger.info(
            f"Adding patient record to EHR: {patient.name} (ID: {patient.patient_id}) - Doctor: {doctor_name}"
        )

        # Create comprehensive medical record
        record_content = f"""
        PATIENT MEDICAL RECORD
        Patient ID: {patient.patient_id}
        Name: {patient.name}
        Age: {patient.age}
        Gender: {patient.gender}
        Chief Complaint: {patient.chief_complaint}
        Symptoms: {', '.join(patient.symptoms)}
        Medical History: {', '.join(patient.medical_history)}
        Current Medications: {', '.join(patient.current_medications)}
        Allergies: {', '.join(patient.allergies)}
        Vital Signs: {json.dumps(patient.vital_signs, indent=2)}
        Diagnosis: {patient.diagnosis or 'Pending'}
        Treatment Plan: {patient.treatment_plan or 'Pending'}
        Doctor: {doctor_name}
        Timestamp: {datetime.now().isoformat()}
        
        MEDICAL NOTES:
        {medical_notes}
        """

        metadata = {
            "patient_id": patient.patient_id,
            "patient_name": patient.name,
            "doctor": doctor_name,
            "timestamp": datetime.now().isoformat(),
            "record_type": "medical_record",
        }

        if CHROMADB_AVAILABLE and self.collection:
            try:
                self.collection.add(
                    documents=[record_content],
                    metadatas=[metadata],
                    ids=[record_id],
                )
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")
                self._add_to_memory(
                    record_id, record_content, metadata
                )
        else:
            self._add_to_memory(record_id, record_content, metadata)

        return record_id

    def _add_to_memory(
        self, record_id: str, content: str, metadata: Dict[str, Any]
    ):
        """Add record to in-memory storage."""
        self.memory_storage[record_id] = {
            "content": content,
            "metadata": metadata,
        }

    def query_patient_history(
        self, patient_id: str, query: str = None
    ) -> List[Dict[str, Any]]:
        """Query patient medical history."""
        if CHROMADB_AVAILABLE and self.collection:
            try:
                if query:
                    results = self.collection.query(
                        query_texts=[query],
                        where={"patient_id": patient_id},
                        n_results=10,
                    )
                else:
                    results = self.collection.get(
                        where={"patient_id": patient_id}
                    )

                return self._format_chromadb_results(results)
            except Exception as e:
                print(f"Error querying ChromaDB: {e}")
                return self._query_memory(patient_id, query)
        else:
            return self._query_memory(patient_id, query)

    def _query_memory(
        self, patient_id: str, query: str = None
    ) -> List[Dict[str, Any]]:
        """Query in-memory storage."""
        results = []
        for record_id, record in self.memory_storage.items():
            if record["metadata"]["patient_id"] == patient_id:
                if (
                    query is None
                    or query.lower() in record["content"].lower()
                ):
                    results.append(
                        {
                            "id": record_id,
                            "content": record["content"],
                            "metadata": record["metadata"],
                        }
                    )
        return results

    def _format_chromadb_results(
        self, results
    ) -> List[Dict[str, Any]]:
        """Format ChromaDB query results."""
        formatted_results = []

        if "ids" in results and results["ids"]:
            for i in range(len(results["ids"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "content": (
                            results["documents"][0][i]
                            if "documents" in results
                            else ""
                        ),
                        "metadata": (
                            results["metadatas"][0][i]
                            if "metadatas" in results
                            else {}
                        ),
                        "distance": (
                            results["distances"][0][i]
                            if "distances" in results
                            else 0.0
                        ),
                    }
                )

        return formatted_results

    def search_similar_cases(
        self, symptoms: List[str], diagnosis: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar medical cases."""
        query_text = f"Symptoms: {', '.join(symptoms)}"
        if diagnosis:
            query_text += f" Diagnosis: {diagnosis}"

        if CHROMADB_AVAILABLE and self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query_text], n_results=5
                )
                return self._format_chromadb_results(results)
            except Exception as e:
                print(f"Error searching similar cases: {e}")
                return self._search_memory_similar(
                    symptoms, diagnosis
                )
        else:
            return self._search_memory_similar(symptoms, diagnosis)

    def _search_memory_similar(
        self, symptoms: List[str], diagnosis: str = None
    ) -> List[Dict[str, Any]]:
        """Search in-memory storage for similar cases."""
        results = []
        for record_id, record in self.memory_storage.items():
            content_lower = record["content"].lower()
            symptom_matches = sum(
                1
                for symptom in symptoms
                if symptom.lower() in content_lower
            )

            if symptom_matches > 0:
                results.append(
                    {
                        "id": record_id,
                        "content": record["content"],
                        "metadata": record["metadata"],
                        "relevance_score": (
                            symptom_matches / len(symptoms)
                        ),
                    }
                )

        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:5]


class HospitalStaff:
    """Base class for hospital staff members."""

    def __init__(self, name: str, role: StaffRole, agent: Agent):
        """Initialize hospital staff member."""
        self.name = name
        self.role = role
        self.agent = agent
        self.is_available = True
        self.current_patient = None
        self.patients_seen = []
        self.performance_metrics = {
            "patients_treated": 0,
            "average_treatment_time": 0.0,
            "patient_satisfaction": 0.0,
        }

    def assign_patient(self, patient: Patient) -> bool:
        """Assign a patient to this staff member."""
        if self.is_available and not self.current_patient:
            self.current_patient = patient
            self.is_available = False
            return True
        return False

    def release_patient(self):
        """Release the current patient."""
        if self.current_patient:
            self.patients_seen.append(self.current_patient)
            self.current_patient = None
            self.is_available = True

    def update_metrics(
        self, treatment_time: float, satisfaction_score: float = None
    ):
        """Update performance metrics."""
        self.performance_metrics["patients_treated"] += 1

        # Update average treatment time
        current_avg = self.performance_metrics[
            "average_treatment_time"
        ]
        total_patients = self.performance_metrics["patients_treated"]
        self.performance_metrics["average_treatment_time"] = (
            current_avg * (total_patients - 1) + treatment_time
        ) / total_patients

        if satisfaction_score is not None:
            # Update patient satisfaction
            current_sat = self.performance_metrics[
                "patient_satisfaction"
            ]
            self.performance_metrics["patient_satisfaction"] = (
                current_sat * (total_patients - 1)
                + satisfaction_score
            ) / total_patients


class PatientQueue:
    """Priority queue for patient management."""

    def __init__(self):
        """Initialize the patient queue."""
        self.queue = PriorityQueue()
        self.waiting_patients = []
        self.treatment_history = []

    def add_patient(self, patient: Patient):
        """Add patient to queue with priority."""
        # Priority queue uses negative priority (higher priority = lower negative number)
        self.queue.put(
            (-patient.priority_score, time.time(), patient)
        )
        self.waiting_patients.append(patient)

    def get_next_patient(self) -> Optional[Patient]:
        """Get next patient from queue."""
        if not self.queue.empty():
            _, _, patient = self.queue.get()
            if patient in self.waiting_patients:
                self.waiting_patients.remove(patient)
            return patient
        return None

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "total_waiting": len(self.waiting_patients),
            "queue_length": self.queue.qsize(),
            "waiting_patients": [
                p.name for p in self.waiting_patients
            ],
            "estimated_wait_times": self._calculate_wait_times(),
        }

    def _calculate_wait_times(self) -> Dict[str, int]:
        """Calculate estimated wait times for patients."""
        wait_times = {}
        for i, patient in enumerate(self.waiting_patients):
            # Base wait time: 15 minutes per patient ahead
            base_wait = (i + 1) * 15

            # Adjust based on priority
            if patient.priority_score >= 15:
                wait_times[patient.name] = max(
                    5, base_wait // 3
                )  # Emergency cases
            elif patient.priority_score >= 10:
                wait_times[patient.name] = max(
                    10, base_wait // 2
                )  # Urgent cases
            else:
                wait_times[patient.name] = base_wait

        return wait_times


class HospitalSimulation:
    """Main hospital simulation orchestrator."""

    def __init__(
        self,
        hospital_name: str = "General Hospital",
        description: str = "A general hospital",
        random_models_on: bool = True,
    ):
        """Initialize the hospital simulation."""
        self.hospital_name = hospital_name
        self.description = description
        self.random_models_on = random_models_on

        self.ehr_system = EHRSystem()
        self.patient_queue = PatientQueue()

        # Staff members
        self.staff = {}
        self.executives = []
        self.doctors = []
        self.nurses = []
        self.receptionists = []

        # Simulation state
        self.is_running = False
        self.simulation_time = datetime.now()
        self.patients_processed = 0
        self.total_wait_time = 0.0
        self.simulation_stats = {
            "total_patients": 0,
            "patients_treated": 0,
            "average_wait_time": 0.0,
            "patient_satisfaction": 0.0,
            "revenue": 0.0,
            "costs": 0.0,
        }

        # Initialize staff
        self._initialize_staff()

        # Simulation thread
        self.simulation_thread = None
        self.stop_event = threading.Event()

    def _initialize_staff(self):
        """Initialize hospital staff with specialized agents."""

        # Executive Team
        ceo = Agent(
            agent_name="Alexander Goldwin",
            system_prompt="""You are Alexander Goldwin, the Chief Executive Officer of a hospital with a visionary approach to healthcare leadership. Your responsibilities include:
            - Strategic planning and hospital growth
            - Financial management and revenue optimization
            - Quality of care oversight
            - Staff management and resource allocation
            - Patient satisfaction and community relations
            - Cost control while maintaining quality
            
            CONVERSATION GUIDELINES:
            - NEVER predict, assume, or speak for what other executives or staff will say
            - Only respond to what has actually been said to you
            - Ask direct questions and wait for actual responses
            - Focus on your own strategic insights and leadership decisions
            - Let others speak for themselves in their own words
            
            Focus on:
            1. Increasing patient volume through marketing and community outreach
            2. Optimizing operational efficiency to reduce costs
            3. Maintaining high quality of care standards
            4. Staff satisfaction and retention
            5. Financial sustainability and growth
            
            Always consider the balance between cost, quality, and patient satisfaction. Speak only for yourself.""",
            random_models_on=True,
            max_loops=1,
        )

        cfo = Agent(
            agent_name="Isabella Silverstone",
            system_prompt="""You are Isabella Silverstone, the Chief Financial Officer of a hospital known for your analytical precision and financial acumen. Your responsibilities include:
            - Financial planning and budgeting
            - Cost analysis and optimization
            - Revenue cycle management
            - Insurance and billing optimization
            - Financial reporting and compliance
            - Investment and capital planning
            
            CONVERSATION GUIDELINES:
            - NEVER predict, assume, or speak for what other executives or staff will say
            - Only respond to what has actually been said to you
            - Ask direct questions and wait for actual responses
            - Focus on your own financial analysis and recommendations
            - Let others speak for themselves in their own words
            
            Focus on:
            1. Reducing operational costs without compromising quality
            2. Optimizing billing and insurance processes
            3. Maximizing revenue from patient services
            4. Financial risk management
            5. Cost-benefit analysis of medical procedures
            
            Always provide data-driven financial recommendations. Speak only for yourself.""",
            random_models_on=True,
            max_loops=1,
        )

        cmo = Agent(
            agent_name="Dr. Marcus Healwright",
            system_prompt="""You are Dr. Marcus Healwright, the Chief Medical Officer of a hospital with a passion for clinical excellence and innovation. Your responsibilities include:
            - Medical quality assurance and standards
            - Clinical protocol development
            - Physician credentialing and oversight
            - Patient safety and risk management
            - Medical staff development and training
            - Clinical research and innovation
            
            CONVERSATION GUIDELINES:
            - NEVER predict, assume, or speak for what other executives or staff will say
            - Only respond to what has actually been said to you
            - Ask direct questions and wait for actual responses
            - Focus on your own medical expertise and quality initiatives
            - Let others speak for themselves in their own words
            
            Focus on:
            1. Maintaining highest standards of medical care
            2. Implementing evidence-based clinical protocols
            3. Continuous quality improvement
            4. Patient safety and risk reduction
            5. Medical staff development and satisfaction
            
            Always prioritize patient safety and quality of care. Speak only for yourself.""",
            random_models_on=True,
            max_loops=1,
        )

        # Doctors
        emergency_doctor = Agent(
            agent_name="Dr. Zara Nightingale",
            system_prompt="""You are Dr. Zara Nightingale, an Emergency Medicine physician with 12 years of experience. Your responsibilities include:
            - Rapid patient assessment and triage
            - Emergency treatment and stabilization
            - Critical care management
            - Patient diagnosis and treatment planning
            - Coordination with specialists
            - Emergency procedures and interventions
            
            CONVERSATION GUIDELINES:
            - NEVER predict, assume, or speak for what patients or other staff will say
            - Only respond to what has actually been said to you
            - Ask direct questions and wait for actual responses
            - Focus on your own medical assessment and recommendations
            - Let others speak for themselves in their own words
            
            When treating patients:
            1. Always start with ABC (Airway, Breathing, Circulation)
            2. Assess vital signs and symptoms thoroughly
            3. Ask targeted questions to understand the problem
            4. Consider differential diagnoses
            5. Order appropriate tests and imaging
            6. Provide clear treatment plans
            7. Document everything in the EHR system
            
            Be thorough, professional, and compassionate. Speak only for yourself.""",
            random_models_on=True,
            max_loops=1,
        )

        general_doctor = Agent(
            agent_name="Dr. Kai Thunderheart",
            system_prompt="""You are Dr. Kai Thunderheart, a General Practice physician known for your methodical approach and warm bedside manner. Your responsibilities include:
            - Comprehensive patient evaluation
            - Diagnosis and treatment of common conditions
            - Preventive care and health maintenance
            - Chronic disease management
            - Referral to specialists when needed
            - Patient education and counseling
            
            CONVERSATION GUIDELINES:
            - NEVER predict, assume, or speak for what patients or other staff will say
            - Only respond to what has actually been said to you
            - Ask direct questions and wait for actual responses
            - Focus on your own medical assessment and recommendations
            - Let others speak for themselves in their own words
            
            When treating patients:
            1. Take a complete medical history
            2. Perform thorough physical examination
            3. Ask systematic questions about symptoms
            4. Consider differential diagnoses
            5. Order appropriate diagnostic tests
            6. Develop comprehensive treatment plans
            7. Document everything in the EHR system
            
            Be thorough, caring, and patient-focused. Speak only for yourself.""",
            random_models_on=True,
            max_loops=1,
        )

        # Nurses
        triage_nurse = Agent(
            agent_name="Nurse Raven Stormborn",
            system_prompt="""You are Nurse Raven Stormborn, an experienced emergency department triage nurse with 8 years of experience. You are professional, efficient, and compassionate.

Your role is to conduct a thorough triage assessment through direct conversation with patients. You should:

CONVERSATION GUIDELINES:
- NEVER predict, assume, or speak for what patients or other staff will say
- Only respond to what has actually been said to you
- Ask direct questions and wait for actual responses
- Focus on your own assessment and nursing care
- Let others speak for themselves in their own words

GREETING & INTRODUCTION:
- Introduce yourself warmly and professionally
- Ask for the patient's name to personalize the interaction
- Make them feel welcome and at ease

ASSESSMENT PROCESS:
- Ask specific questions about their chief complaint
- Inquire about pain levels (scale 1-10), symptom duration, and severity
- Take vital signs (blood pressure, heart rate, temperature, respiratory rate, oxygen saturation)
- Ask about current medications, allergies, and relevant medical history
- Assess their immediate needs and comfort level

COMMUNICATION STYLE:
- Ask one question at a time and wait for patient responses
- Use clear, simple language patients can understand
- Show empathy for their concerns
- Be thorough but efficient
- Document vital signs with specific numbers (e.g., "Your blood pressure is 140/90")

TRIAGE PRIORITY ASSESSMENT:
- Emergency (immediate): Life-threatening conditions, severe pain (8-10/10), abnormal vital signs
- Urgent (within 30 minutes): Moderate to severe symptoms, concerning vital signs
- Standard (within 2 hours): Stable patients with non-urgent conditions

Always engage in natural conversation and respond directly to what the patient tells you. Ask follow-up questions based on their responses. Speak only for yourself.""",
            random_models_on=True,
            max_loops=1,
        )

        floor_nurse = Agent(
            agent_name="Nurse Phoenix Brightwater",
            system_prompt="""You are Nurse Phoenix Brightwater, a dedicated Floor Nurse known for your attention to detail and caring nature. Your responsibilities include:
            - Patient care and monitoring
            - Medication administration
            - Treatment implementation
            - Patient education and support
            - Documentation and charting
            - Communication with medical team
            
            CONVERSATION GUIDELINES:
            - NEVER predict, assume, or speak for what patients or other staff will say
            - Only respond to what has actually been said to you
            - Ask direct questions and wait for actual responses
            - Focus on your own nursing care and observations
            - Let others speak for themselves in their own words
            
            When caring for patients:
            1. Follow doctor's orders precisely
            2. Monitor patient response to treatment
            3. Document all care provided
            4. Communicate patient status to doctors
            5. Provide patient education and support
            6. Maintain patient comfort and safety
            
            Be attentive, caring, and professional. Speak only for yourself.""",
            random_models_on=True,
            max_loops=1,
        )

        # Receptionists
        receptionist = Agent(
            agent_name="Crystal Moonwhisper",
            system_prompt="""You are Crystal Moonwhisper, a Hospital Receptionist known for your warm personality and organizational skills. Your responsibilities include:
            - Patient check-in and registration
            - Appointment scheduling and management
            - Insurance verification and billing
            - Patient communication and information
            - Queue management and patient flow
            - Administrative support
            
            CONVERSATION GUIDELINES:
            - NEVER predict, assume, or speak for what patients or other staff will say
            - Only respond to what has actually been said to you
            - Ask direct questions and wait for actual responses
            - Focus on your own administrative tasks and patient service
            - Let others speak for themselves in their own words
            
            When working with patients:
            1. Greet patients warmly and professionally
            2. Collect necessary information efficiently
            3. Verify insurance and payment information
            4. Explain wait times and procedures
            5. Direct patients to appropriate areas
            6. Maintain organized patient flow
            
            Be welcoming, efficient, and helpful. Speak only for yourself.""",
            random_models_on=True,
            max_loops=1,
        )

        # Create staff objects
        self.executives = [
            HospitalStaff(
                "Alexander Goldwin", StaffRole.EXECUTIVE, ceo
            ),
            HospitalStaff(
                "Isabella Silverstone", StaffRole.EXECUTIVE, cfo
            ),
            HospitalStaff(
                "Dr. Marcus Healwright", StaffRole.EXECUTIVE, cmo
            ),
        ]

        self.doctors = [
            HospitalStaff(
                "Dr. Zara Nightingale",
                StaffRole.DOCTOR,
                emergency_doctor,
            ),
            HospitalStaff(
                "Dr. Kai Thunderheart",
                StaffRole.DOCTOR,
                general_doctor,
            ),
        ]

        self.nurses = [
            HospitalStaff(
                "Nurse Raven Stormborn", StaffRole.NURSE, triage_nurse
            ),
            HospitalStaff(
                "Nurse Phoenix Brightwater",
                StaffRole.NURSE,
                floor_nurse,
            ),
        ]

        self.receptionists = [
            HospitalStaff(
                "Crystal Moonwhisper",
                StaffRole.RECEPTIONIST,
                receptionist,
            )
        ]

        # Add all staff to main staff dictionary
        for staff_list in [
            self.executives,
            self.doctors,
            self.nurses,
            self.receptionists,
        ]:
            for staff in staff_list:
                self.staff[staff.name] = staff

        logger.info(
            f"Hospital staff initialized: {len(self.staff)} staff members"
        )
        print(
            f"✅ Hospital staff initialized: {len(self.staff)} staff members"
        )

    def add_patient(self, patient: Patient):
        """Add a new patient to the hospital."""
        logger.info(
            f"New patient arrival: {patient.name} - {patient.chief_complaint}"
        )
        # Receptionist processes patient check-in
        receptionist = self.receptionists[0]
        if receptionist.is_available:
            receptionist.assign_patient(patient)

            # Process check-in with patient interaction
            check_in_prompt = "Greet and check in the patient. Ask for their name, age, chief complaint, and gather initial information."
            check_in_result = receptionist.agent.run(check_in_prompt)

            # Patient responds to check-in
            patient_response = patient.agent.run(
                f"You are at the hospital reception desk. The receptionist is checking you in and asking for information. "
                f"Respond appropriately and provide your information: {check_in_result}"
            )

            # Add to queue
            self.patient_queue.add_patient(patient)
            self.simulation_stats["total_patients"] += 1

            receptionist.release_patient()

            logger.info(
                f"Patient {patient.name} checked in and added to queue"
            )
            logger.debug(
                f"Reception interaction: {check_in_result[:100]}..."
            )
            logger.debug(
                f"Patient response: {patient_response[:100]}..."
            )
            print(
                f"✅ Patient {patient.name} checked in and added to queue"
            )
            print(
                f"   Reception interaction: {check_in_result[:100]}..."
            )
            print(f"   Patient response: {patient_response[:100]}...")
        else:
            logger.warning(
                f"Receptionist busy, patient {patient.name} waiting"
            )
            print(
                f"⚠️ Receptionist busy, patient {patient.name} waiting"
            )

    def process_patient_queue(self):
        """Process patients in the queue."""
        if self.patient_queue.queue.empty():
            return

        # Get next patient
        patient = self.patient_queue.get_next_patient()
        if not patient:
            return

        logger.info(
            f"Processing patient from queue: {patient.name} (Priority: {patient.priority_score})"
        )

        # Find available staff
        available_nurse = next(
            (n for n in self.nurses if n.is_available), None
        )
        available_doctor = next(
            (d for d in self.doctors if d.is_available), None
        )

        if available_nurse and available_doctor:
            logger.info(
                f"Starting triage for {patient.name} with {available_nurse.name}"
            )
            # Triage with nurse using the enhanced interactive method
            triage_result = self._triage_assessment(patient)

            logger.info(
                f"Triage completed for {patient.name}, starting consultation with {available_doctor.name}"
            )
            # Doctor consultation
            available_doctor.assign_patient(patient)
            patient.status = PatientStatus.WITH_DOCTOR

            # Get patient history from EHR
            patient_history = self.ehr_system.query_patient_history(
                patient.patient_id
            )
            history_context = ""
            if patient_history:
                history_context = f"\n\nPATIENT HISTORY:\n{patient_history[0]['content']}"

            # Doctor consultation
            doctor_prompt = (
                f"You are consulting with a patient. Review their triage results and medical history if available. "
                f"Vital Signs: {json.dumps(patient.vital_signs, indent=2)}"
                f"{history_context}\n\n"
                f"Greet the patient, ask relevant questions about their condition, perform examination, "
                f"and work towards a diagnosis and treatment plan."
            )
            consultation_result = available_doctor.agent.run(
                doctor_prompt
            )

            # Patient responds to doctor
            patient_consultation_response = patient.agent.run(
                f"You are now with the doctor for your consultation. The doctor is examining you and asking questions. "
                f"Answer their questions honestly and describe your symptoms in detail: {consultation_result}"
            )

            # Doctor processes patient responses and finalizes diagnosis
            final_consultation = available_doctor.agent.run(
                f"Based on the patient's responses and your examination, provide your final diagnosis and treatment plan. "
                f"Patient responses: {patient_consultation_response}"
            )

            # Extract diagnosis and treatment plan
            diagnosis, treatment_plan = (
                self._extract_diagnosis_and_treatment(
                    final_consultation
                )
            )
            patient.diagnosis = diagnosis
            patient.treatment_plan = treatment_plan

            # Save to EHR
            medical_notes = f"TRIAGE: {triage_result}\n\nCONSULTATION: {consultation_result}\n\nPATIENT CONSULTATION RESPONSE: {patient_consultation_response}\n\nFINAL DIAGNOSIS: {final_consultation}"
            self.ehr_system.add_patient_record(
                patient, medical_notes, available_doctor.name
            )

            # Update patient status
            patient.status = PatientStatus.TREATED
            available_doctor.release_patient()

            # Update metrics
            treatment_time = (
                datetime.now() - patient.arrival_time
            ).total_seconds() / 60  # minutes
            available_doctor.update_metrics(treatment_time)

            self.simulation_stats["patients_treated"] += 1
            self.total_wait_time += treatment_time

            logger.info(
                f"Patient {patient.name} treated by {available_doctor.name} - Diagnosis: {diagnosis} - Time: {treatment_time:.1f} minutes"
            )
            print(
                f"✅ Patient {patient.name} treated by {available_doctor.name}"
            )
            print(f"   Diagnosis: {diagnosis}")
            print(f"   Treatment Time: {treatment_time:.1f} minutes")

    def _extract_vital_signs(
        self, triage_result: str
    ) -> Dict[str, Any]:
        """Extract vital signs from triage result."""
        vital_signs = {}

        # Extract actual vital signs mentioned in triage result
        # This is a simplified parser - in production, use proper medical NLP
        lines = triage_result.lower().split("\n")
        for line in lines:
            if "blood pressure" in line or "bp" in line:
                # Look for numbers like "120/80" or "120 over 80"
                import re

                bp_match = re.search(r"(\d+)[/\s]+(\d+)", line)
                if bp_match:
                    vital_signs["blood_pressure"] = {
                        "systolic": int(bp_match.group(1)),
                        "diastolic": int(bp_match.group(2)),
                    }
            elif "heart rate" in line or "pulse" in line:
                hr_match = re.search(r"(\d+)", line)
                if hr_match:
                    vital_signs["heart_rate"] = int(hr_match.group(1))
            elif "temperature" in line or "temp" in line:
                temp_match = re.search(r"(\d+\.?\d*)", line)
                if temp_match:
                    vital_signs["temperature"] = float(
                        temp_match.group(1)
                    )
            elif "oxygen" in line or "o2" in line:
                o2_match = re.search(r"(\d+)", line)
                if o2_match:
                    vital_signs["oxygen_saturation"] = int(
                        o2_match.group(1)
                    )

        return vital_signs

    def _extract_diagnosis_and_treatment(
        self, consultation_result: str
    ) -> Tuple[str, str]:
        """Extract diagnosis and treatment plan from consultation result."""
        # Simple extraction - in production, use more sophisticated NLP
        lines = consultation_result.split("\n")
        diagnosis = "Diagnosis pending"
        treatment_plan = "Treatment plan pending"

        for i, line in enumerate(lines):
            if "diagnosis" in line.lower():
                diagnosis = line.strip()
            elif (
                "treatment" in line.lower() and "plan" in line.lower()
            ):
                treatment_plan = line.strip()

        return diagnosis, treatment_plan

    def executive_meeting(self):
        """Hold executive meeting to discuss hospital performance and strategy."""
        if not self.executives:
            return

        # Create executive swarm
        executive_swarm = HierarchicalSwarm(
            name="Executive Team",
            description="Hospital executive team for strategic planning",
            agents=[executive.agent for executive in self.executives],
            max_loops=2,
            verbose=True,
        )

        # Prepare meeting agenda
        agenda = f"""
        EXECUTIVE TEAM MEETING - {self.hospital_name}
        
        Current Hospital Status:
        - Total Patients: {self.simulation_stats['total_patients']}
        - Patients Treated: {self.simulation_stats['patients_treated']}
        - Average Wait Time: {self.simulation_stats['average_wait_time']:.1f} minutes
        - Revenue: ${self.simulation_stats['revenue']:.2f}
        - Costs: ${self.simulation_stats['costs']:.2f}
        
        Discussion Topics:
        1. Patient volume and growth strategies
        2. Operational efficiency improvements
        3. Quality of care metrics
        4. Financial performance and cost optimization
        5. Staff satisfaction and retention
        6. Community outreach and marketing
        
        Please provide strategic recommendations for improving hospital performance.
        """

        # Run executive meeting
        meeting_result = executive_swarm.run(agenda)

        print(f"🏥 Executive Meeting Results:\n{meeting_result}")

        # Update simulation based on executive decisions
        self._implement_executive_decisions(meeting_result)

    def _implement_executive_decisions(self, meeting_result: str):
        """Implement decisions from executive meeting."""
        # Parse actual executive decisions from meeting results
        # This would integrate with real hospital management systems
        print("📋 Executive decisions recorded for implementation")

        # In a real system, this would:
        # 1. Parse structured decisions from the meeting
        # 2. Create action items and assignees
        # 3. Update hospital policies and procedures
        # 4. Modify resource allocation
        # 5. Update performance targets

    def update_statistics(self):
        """Update hospital statistics."""
        if self.simulation_stats["patients_treated"] > 0:
            self.simulation_stats["average_wait_time"] = (
                self.total_wait_time
                / self.simulation_stats["patients_treated"]
            )

        # Calculate revenue and costs
        self.simulation_stats["revenue"] = (
            self.simulation_stats["patients_treated"] * 150
        )  # $150 per patient
        self.simulation_stats["costs"] = (
            self.simulation_stats["patients_treated"] * 100
        )  # $100 per patient

        # Calculate patient satisfaction (simplified)
        if self.simulation_stats["patients_treated"] > 0:
            wait_time_factor = max(
                0,
                1 - (self.simulation_stats["average_wait_time"] / 60),
            )  # 1 hour max
            self.simulation_stats["patient_satisfaction"] = (
                wait_time_factor * 100
            )

    def generate_patients(self, num_patients: int = 5):
        """Generate sample patients for simulation."""
        sample_patients = [
            Patient(
                name="Xavier Delacroix",
                age=45,
                gender="Male",
                chief_complaint="Chest pain",
                symptoms=[
                    "chest pain",
                    "shortness of breath",
                    "sweating",
                ],
                medical_history=["hypertension", "diabetes"],
                current_medications=["metformin", "lisinopril"],
                allergies=["penicillin"],
                system_prompt="You are Xavier Delacroix, a 45-year-old man experiencing chest pain. You're worried this might be a heart attack because you have a history of high blood pressure and diabetes. You're sweating and feeling anxious. Be cooperative with medical staff but express your concerns about your heart.",
            ),
            Patient(
                name="Zara Al-Rashid",
                age=32,
                gender="Female",
                chief_complaint="Severe headache",
                symptoms=["headache", "nausea", "light sensitivity"],
                medical_history=["migraines"],
                current_medications=["sumatriptan"],
                allergies=[],
                system_prompt="You are Zara Al-Rashid, a 32-year-old woman with a severe migraine. You have a history of migraines but this one feels different and more intense. The light is bothering you and you feel nauseous. You're hoping to get something stronger for the pain.",
            ),
            Patient(
                name="Kofi Asante",
                age=58,
                gender="Male",
                chief_complaint="Fever and cough",
                symptoms=["fever", "cough", "fatigue", "body aches"],
                medical_history=["asthma"],
                current_medications=["albuterol inhaler"],
                allergies=["sulfa drugs"],
                system_prompt="You are Kofi Asante, a 58-year-old man with flu-like symptoms. You have asthma and you're concerned about your breathing. You've been coughing a lot and feel very tired. You want to make sure it's not something serious affecting your lungs.",
            ),
            Patient(
                name="Priya Sharma",
                age=28,
                gender="Female",
                chief_complaint="Abdominal pain",
                symptoms=["abdominal pain", "nausea", "vomiting"],
                medical_history=["appendicitis"],
                current_medications=[],
                allergies=[],
                system_prompt="You are Priya Sharma, a 28-year-old woman with severe abdominal pain. You had your appendix removed before, so you're worried about what could be causing this pain. You've been vomiting and the pain is getting worse.",
            ),
            Patient(
                name="Dmitri Volkov",
                age=67,
                gender="Male",
                chief_complaint="Dizziness",
                symptoms=["dizziness", "confusion", "weakness"],
                medical_history=["hypertension", "heart disease"],
                current_medications=["atenolol", "aspirin"],
                allergies=["codeine"],
                system_prompt="You are Dmitri Volkov, a 67-year-old man feeling dizzy and confused. You have heart problems and high blood pressure, so you're worried this might be related. You feel weak and unsteady. Your family brought you in because they're concerned.",
            ),
        ]

        for patient in sample_patients[:num_patients]:
            self.add_patient(patient)

    def run(
        self,
        patient_record: Dict[str, Any],
        symptoms: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the hospital pipeline for a specific patient.

        Args:
            patient_record: Dictionary containing patient information
            symptoms: List of symptoms (if not already in patient_record)

        Returns:
            Dictionary containing treatment results and recommendations
        """
        print(
            f"🏥 Processing patient: {patient_record.get('name', 'Unknown')}"
        )

        # Create patient object from record
        if symptoms:
            patient_record["symptoms"] = symptoms

        try:
            patient = Patient(
                name=patient_record.get("name", ""),
                age=patient_record.get("age", 0),
                gender=patient_record.get("gender", ""),
                chief_complaint=patient_record.get(
                    "chief_complaint", ""
                ),
                symptoms=patient_record.get("symptoms", []),
                medical_history=patient_record.get(
                    "medical_history", []
                ),
                current_medications=patient_record.get(
                    "current_medications", []
                ),
                allergies=patient_record.get("allergies", []),
            )
        except Exception as e:
            logger.error(f"Failed to create patient: {str(e)}")
            return {
                "error": f"Failed to create patient: {str(e)}",
                "status": "failed",
            }

        # Add patient to hospital
        self.add_patient(patient)

        # Process patient through the pipeline
        result = self._process_single_patient(patient)

        return result

    def _process_single_patient(
        self, patient: Patient
    ) -> Dict[str, Any]:
        """Process a single patient through the complete hospital pipeline."""
        start_time = time.time()

        try:
            # Step 1: Reception check-in
            print(f"📋 {patient.name} checking in...")
            reception_result = self._reception_checkin(patient)

            # Step 2: Triage assessment
            print(f"🔍 {patient.name} undergoing triage...")
            triage_result = self._triage_assessment(patient)

            # Step 3: Doctor consultation
            print(f"👨‍⚕️ {patient.name} consulting with doctor...")
            consultation_result = self._doctor_consultation(patient)

            # Step 4: Treatment planning
            print(f"💊 {patient.name} receiving treatment plan...")
            treatment_result = self._treatment_planning(patient)

            # Step 5: EHR documentation
            print(f"📝 {patient.name} records being documented...")
            ehr_result = self._document_in_ehr(
                patient,
                triage_result,
                consultation_result,
                treatment_result,
            )

            # Calculate total time
            total_time = (time.time() - start_time) / 60  # minutes

            # Update metrics
            if patient.assigned_doctor:
                self.staff[patient.assigned_doctor].update_metrics(
                    total_time
                )

            return {
                "patient_id": patient.patient_id,
                "patient_name": patient.name,
                "status": "completed",
                "total_time_minutes": total_time,
                "reception": reception_result,
                "triage": triage_result,
                "consultation": consultation_result,
                "treatment": treatment_result,
                "ehr_record_id": ehr_result,
                "diagnosis": patient.diagnosis,
                "treatment_plan": patient.treatment_plan,
                "priority_score": patient.priority_score,
                "assigned_doctor": patient.assigned_doctor,
                "assigned_nurse": patient.assigned_nurse,
            }

        except Exception as e:
            logger.error(
                f"Error processing patient {patient.name}: {str(e)}"
            )
            return {
                "patient_id": patient.patient_id,
                "patient_name": patient.name,
                "status": "error",
                "error": str(e),
                "total_time_minutes": (time.time() - start_time) / 60,
            }

    def _reception_checkin(self, patient: Patient) -> str:
        """Handle patient check-in at reception."""
        receptionist = self.receptionists[0]
        if not receptionist.is_available:
            # Wait for receptionist to become available
            while not receptionist.is_available:
                time.sleep(0.1)

        receptionist.assign_patient(patient)

        # Receptionist greets and checks in patient
        check_in_prompt = "Greet the patient and process their check-in. Ask for their information and explain the process."
        check_in_result = receptionist.agent.run(check_in_prompt)

        # Patient responds to receptionist
        patient_check_in_response = patient.agent.run(
            f"You are at the hospital reception desk for check-in. The receptionist is greeting you and asking for information. "
            f"Respond appropriately: {check_in_result}"
        )

        receptionist.release_patient()
        return f"RECEPTION: {check_in_result}\n\nPATIENT: {patient_check_in_response}"

    def _triage_assessment(self, patient: Patient) -> str:
        """Perform triage assessment with nurse."""
        available_nurse = next(
            (n for n in self.nurses if n.is_available), None
        )
        if not available_nurse:
            # Wait for nurse to become available
            while (
                not available_nurse
                or not available_nurse.is_available
            ):
                available_nurse = next(
                    (n for n in self.nurses if n.is_available), None
                )
                time.sleep(0.1)

        available_nurse.assign_patient(patient)
        patient.status = PatientStatus.IN_TRIAGE

        # Multi-step triage conversation
        conversation_log = []

        # Step 1: Nurse greeting and initial questions
        initial_prompt = "A patient has just arrived for triage assessment. Begin by greeting them professionally, introducing yourself, and asking for their name and what brought them to the hospital today."
        nurse_greeting = available_nurse.agent.run(initial_prompt)
        conversation_log.append(f"NURSE: {nurse_greeting}")

        # Step 2: Patient responds to greeting
        patient_response_1 = patient.agent.run(
            f"You have just been called in for triage assessment. The triage nurse is greeting you. "
            f"Respond naturally to their greeting and questions. Here's what they said: {nurse_greeting}"
        )
        conversation_log.append(f"PATIENT: {patient_response_1}")

        # Step 3: Nurse takes vital signs and asks detailed questions
        detailed_assessment_prompt = f"The patient has responded to your greeting. Now take their vital signs and ask detailed questions about their symptoms, pain level (1-10 scale), how long they've been experiencing symptoms, and any relevant medical history. Patient's response was: {patient_response_1}"
        nurse_assessment = available_nurse.agent.run(
            detailed_assessment_prompt
        )
        conversation_log.append(f"NURSE: {nurse_assessment}")

        # Step 4: Patient provides detailed symptom information
        patient_response_2 = patient.agent.run(
            f"The nurse is now taking your vital signs and asking detailed questions about your symptoms. "
            f"Provide specific information about your pain level, how you're feeling, and answer their questions honestly. "
            f"The nurse said: {nurse_assessment}"
        )
        conversation_log.append(f"PATIENT: {patient_response_2}")

        # Step 5: Nurse completes assessment and determines priority
        final_assessment_prompt = (
            "Based on the patient's responses and the vital signs you've taken, complete your triage assessment. "
            "Provide specific vital sign measurements, assign a priority level (Emergency/Urgent/Standard), and explain next steps to the patient. "
            f"Patient's detailed response was: {patient_response_2}"
        )
        nurse_conclusion = available_nurse.agent.run(
            final_assessment_prompt
        )
        conversation_log.append(f"NURSE: {nurse_conclusion}")

        # Step 6: Patient's final response
        patient_response_3 = patient.agent.run(
            f"The nurse has completed your triage assessment and is explaining the results and next steps. "
            f"Respond appropriately to what they've told you. The nurse said: {nurse_conclusion}"
        )
        conversation_log.append(f"PATIENT: {patient_response_3}")

        # Extract vital signs from the conversation
        full_conversation = "\n".join(conversation_log)
        patient.vital_signs = self._extract_vital_signs(
            full_conversation
        )
        patient.calculate_priority()
        patient.assigned_nurse = available_nurse.name

        available_nurse.release_patient()
        return full_conversation

    def _doctor_consultation(self, patient: Patient) -> str:
        """Perform doctor consultation."""
        available_doctor = next(
            (d for d in self.doctors if d.is_available), None
        )
        if not available_doctor:
            # Wait for doctor to become available
            while (
                not available_doctor
                or not available_doctor.is_available
            ):
                available_doctor = next(
                    (d for d in self.doctors if d.is_available), None
                )
                time.sleep(0.1)

        available_doctor.assign_patient(patient)
        patient.status = PatientStatus.WITH_DOCTOR
        patient.assigned_doctor = available_doctor.name

        # Get patient history from EHR
        patient_history = self.ehr_system.query_patient_history(
            patient.patient_id
        )
        history_context = ""
        if patient_history:
            history_context = f"\n\nPATIENT HISTORY:\n{patient_history[0]['content']}"

        # Doctor consultation
        doctor_prompt = (
            f"You are consulting with a patient. Review their triage results and medical history if available. "
            f"Vital Signs: {json.dumps(patient.vital_signs, indent=2)}"
            f"{history_context}\n\n"
            f"Greet the patient, ask relevant questions about their condition, perform examination, "
            f"and work towards a diagnosis and treatment plan."
        )
        consultation_result = available_doctor.agent.run(
            doctor_prompt
        )

        # Patient responds to doctor
        patient_consultation_response = patient.agent.run(
            f"You are now with the doctor for your consultation. The doctor is examining you and asking questions. "
            f"Answer their questions honestly and describe your symptoms in detail: {consultation_result}"
        )

        # Doctor processes patient responses and finalizes diagnosis
        final_consultation = available_doctor.agent.run(
            f"Based on the patient's responses and your examination, provide your final diagnosis and treatment plan. "
            f"Patient responses: {patient_consultation_response}"
        )

        # Extract diagnosis and treatment plan
        diagnosis, treatment_plan = (
            self._extract_diagnosis_and_treatment(final_consultation)
        )
        patient.diagnosis = diagnosis
        patient.treatment_plan = treatment_plan

        available_doctor.release_patient()
        return f"CONSULTATION: {consultation_result}\n\nPATIENT: {patient_consultation_response}\n\nFINAL: {final_consultation}"

    def _treatment_planning(self, patient: Patient) -> str:
        """Create comprehensive treatment plan."""
        # This could involve multiple specialists or treatment protocols
        treatment_summary = f"""
        TREATMENT PLAN FOR {patient.name}
        
        Diagnosis: {patient.diagnosis}
        Primary Treatment: {patient.treatment_plan}
        
        Follow-up Recommendations:
        - Schedule follow-up appointment
        - Monitor symptoms
        - Medication compliance
        - Lifestyle modifications if applicable
        """

        return treatment_summary

    def _document_in_ehr(
        self,
        patient: Patient,
        triage_result: str,
        consultation_result: str,
        treatment_result: str,
    ) -> str:
        """Document all information in EHR system."""
        medical_notes = f"""
        TRIAGE ASSESSMENT:
        {triage_result}
        
        DOCTOR CONSULTATION:
        {consultation_result}
        
        TREATMENT PLAN:
        {treatment_result}
        """

        record_id = self.ehr_system.add_patient_record(
            patient,
            medical_notes,
            patient.assigned_doctor or "Unknown",
        )

        # Update patient status
        patient.status = PatientStatus.TREATED

        return record_id

    def run_simulation(
        self,
        duration_minutes: int = 60,
        patient_arrival_rate: float = 0.1,
    ):
        """Run the hospital simulation."""
        logger.info(
            f"Starting {self.hospital_name} simulation for {duration_minutes} minutes (arrival rate: {patient_arrival_rate})"
        )
        print(
            f"🏥 Starting {self.hospital_name} simulation for {duration_minutes} minutes"
        )
        self.is_running = True
        self.stop_event.clear()

        start_time = time.time()
        last_patient_time = start_time

        try:
            while (
                time.time() - start_time < duration_minutes * 60
                and not self.stop_event.is_set()
            ):
                current_time = time.time()

                # Generate new patients based on arrival rate
                if (current_time - last_patient_time) > (
                    1 / patient_arrival_rate
                ) * 60:  # Convert to seconds
                    if (
                        random.random() < 0.3
                    ):  # 30% chance of new patient
                        logger.debug(
                            "Generating new random patient arrival"
                        )
                        self.generate_patients(1)
                        last_patient_time = current_time

                # Process patient queue
                self.process_patient_queue()

                # Hold executive meeting every 15 minutes
                if (
                    int((current_time - start_time) / 60) % 15 == 0
                    and int((current_time - start_time) / 60) > 0
                ):
                    logger.info("Holding executive meeting")
                    self.executive_meeting()

                # Update statistics
                self.update_statistics()

                # Print status every 5 minutes
                if (
                    int((current_time - start_time) / 60) % 5 == 0
                    and int((current_time - start_time) / 60) > 0
                ):
                    self.print_status()

                time.sleep(1)  # 1 second simulation tick

        except KeyboardInterrupt:
            logger.info("Simulation stopped by user")
            print("\n⏹️ Simulation stopped by user")
        finally:
            self.is_running = False
            logger.info("Simulation ended")
            self.print_final_report()

    def print_status(self):
        """Print current hospital status."""
        print(f"\n🏥 {self.hospital_name} - Status Update")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(
            f"Patients in Queue: {self.patient_queue.get_queue_status()['total_waiting']}"
        )
        print(
            f"Patients Treated: {self.simulation_stats['patients_treated']}"
        )
        print(
            f"Average Wait Time: {self.simulation_stats['average_wait_time']:.1f} minutes"
        )
        print(f"Revenue: ${self.simulation_stats['revenue']:.2f}")
        print(f"Costs: ${self.simulation_stats['costs']:.2f}")
        print(
            f"Net Profit: ${self.simulation_stats['revenue'] - self.simulation_stats['costs']:.2f}"
        )
        print(
            f"Patient Satisfaction: {self.simulation_stats['patient_satisfaction']:.1f}%"
        )

    def print_final_report(self):
        """Print final simulation report."""
        print(f"\n🏥 {self.hospital_name} - Final Report")
        print("=" * 50)
        print(
            f"Simulation Duration: {datetime.now() - self.simulation_time}"
        )
        print(
            f"Total Patients: {self.simulation_stats['total_patients']}"
        )
        print(
            f"Patients Treated: {self.simulation_stats['patients_treated']}"
        )
        print(
            f"Average Wait Time: {self.simulation_stats['average_wait_time']:.1f} minutes"
        )
        print(
            f"Total Revenue: ${self.simulation_stats['revenue']:.2f}"
        )
        print(f"Total Costs: ${self.simulation_stats['costs']:.2f}")
        print(
            f"Net Profit: ${self.simulation_stats['revenue'] - self.simulation_stats['costs']:.2f}"
        )
        print(
            f"Patient Satisfaction: {self.simulation_stats['patient_satisfaction']:.1f}%"
        )

        # Staff performance
        print("\nStaff Performance:")
        for staff_name, staff in self.staff.items():
            if staff.performance_metrics["patients_treated"] > 0:
                print(
                    f"  {staff_name}: {staff.performance_metrics['patients_treated']} patients, "
                    f"Avg time: {staff.performance_metrics['average_treatment_time']:.1f} min"
                )

        # EHR statistics
        if CHROMADB_AVAILABLE:
            print(
                f"\nEHR System: ChromaDB with collection '{self.ehr_system.collection_name}'"
            )
        else:
            print(
                f"\nEHR System: In-memory storage with {len(self.ehr_system.memory_storage)} records"
            )
