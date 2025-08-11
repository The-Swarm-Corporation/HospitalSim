#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Hospital Simulation System

This module contains extensive unit tests for all components of the hospital
simulation system using pure functions and loguru for logging.

Features tested:
- Patient class and all its methods
- EHRSystem with ChromaDB and memory storage
- HospitalStaff management and metrics
- PatientQueue priority handling
- HospitalSimulation orchestration
- Edge cases and error handling
- Integration scenarios

Author: AI Assistant
Date: 2024
"""

import sys
import os
import json
import time
from datetime import datetime

# Add hospital_sim to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger
from hospital_sim.main import (
    Patient,
    PatientStatus,
    StaffRole,
    EHRSystem,
    HospitalStaff,
    PatientQueue,
    HospitalSimulation,
    CHROMADB_AVAILABLE,
)
from swarms import Agent

from dotenv import load_dotenv

load_dotenv()

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "errors": [],
    "test_details": [],
}


# Configure logger for tests
logger.add("tests/test_results.log", rotation="10 MB", level="DEBUG")


def log_test_result(
    test_name: str, passed: bool, details: str = "", error: str = ""
):
    """Log test result with details."""
    global test_results

    if passed:
        test_results["passed"] += 1
        logger.success(f"✅ {test_name}: PASSED - {details}")
    else:
        test_results["failed"] += 1
        test_results["errors"].append(f"{test_name}: {error}")
        logger.error(f"❌ {test_name}: FAILED - {error}")

    test_results["test_details"].append(
        {
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
    )


def assert_equal(actual, expected, test_name: str, context: str = ""):
    """Assert that two values are equal."""
    try:
        if actual == expected:
            log_test_result(
                test_name,
                True,
                f"{context} - Expected: {expected}, Got: {actual}",
            )
            return True
        else:
            log_test_result(
                test_name,
                False,
                context,
                f"Expected: {expected}, Got: {actual}",
            )
            return False
    except Exception as e:
        log_test_result(
            test_name,
            False,
            context,
            f"Exception during comparison: {str(e)}",
        )
        return False


def assert_not_none(value, test_name: str, context: str = ""):
    """Assert that value is not None."""
    try:
        if value is not None:
            log_test_result(
                test_name,
                True,
                f"{context} - Value is not None: {type(value)}",
            )
            return True
        else:
            log_test_result(
                test_name, False, context, "Value is None"
            )
            return False
    except Exception as e:
        log_test_result(
            test_name,
            False,
            context,
            f"Exception during check: {str(e)}",
        )
        return False


def assert_true(condition, test_name: str, context: str = ""):
    """Assert that condition is True."""
    try:
        if condition:
            log_test_result(
                test_name, True, f"{context} - Condition is True"
            )
            return True
        else:
            log_test_result(
                test_name, False, context, "Condition is False"
            )
            return False
    except Exception as e:
        log_test_result(
            test_name,
            False,
            context,
            f"Exception during check: {str(e)}",
        )
        return False


def assert_in_range(
    value, min_val, max_val, test_name: str, context: str = ""
):
    """Assert that value is within range."""
    try:
        if min_val <= value <= max_val:
            log_test_result(
                test_name,
                True,
                f"{context} - Value {value} in range [{min_val}, {max_val}]",
            )
            return True
        else:
            log_test_result(
                test_name,
                False,
                context,
                f"Value {value} not in range [{min_val}, {max_val}]",
            )
            return False
    except Exception as e:
        log_test_result(
            test_name,
            False,
            context,
            f"Exception during range check: {str(e)}",
        )
        return False


def assert_contains(
    container, item, test_name: str, context: str = ""
):
    """Assert that container contains item."""
    try:
        if item in container:
            log_test_result(
                test_name,
                True,
                f"{context} - Item found in container",
            )
            return True
        else:
            log_test_result(
                test_name,
                False,
                context,
                f"Item {item} not found in container",
            )
            return False
    except Exception as e:
        log_test_result(
            test_name,
            False,
            context,
            f"Exception during containment check: {str(e)}",
        )
        return False


def assert_isinstance(
    obj, expected_type, test_name: str, context: str = ""
):
    """Assert that object is instance of expected type."""
    try:
        if isinstance(obj, expected_type):
            log_test_result(
                test_name,
                True,
                f"{context} - Object is {expected_type.__name__}",
            )
            return True
        else:
            log_test_result(
                test_name,
                False,
                context,
                f"Object is {type(obj).__name__}, expected {expected_type.__name__}",
            )
            return False
    except Exception as e:
        log_test_result(
            test_name,
            False,
            context,
            f"Exception during type check: {str(e)}",
        )
        return False


# =============================================================================
# PATIENT CLASS TESTS
# =============================================================================


def test_patient_creation_basic():
    """Test basic patient creation with minimal data."""
    logger.info("Testing basic patient creation...")

    try:
        patient = Patient(
            name="John Doe",
            age=35,
            gender="Male",
            chief_complaint="Headache",
        )

        assert_equal(
            patient.name,
            "John Doe",
            "test_patient_name",
            "Basic patient name",
        )
        assert_equal(
            patient.age, 35, "test_patient_age", "Basic patient age"
        )
        assert_equal(
            patient.gender,
            "Male",
            "test_patient_gender",
            "Basic patient gender",
        )
        assert_equal(
            patient.chief_complaint,
            "Headache",
            "test_patient_complaint",
            "Basic patient complaint",
        )
        assert_equal(
            patient.status,
            PatientStatus.WAITING,
            "test_patient_status",
            "Initial patient status",
        )
        assert_not_none(
            patient.patient_id,
            "test_patient_id",
            "Patient ID generation",
        )
        assert_not_none(
            patient.arrival_time,
            "test_patient_arrival",
            "Patient arrival time",
        )
        assert_not_none(
            patient.agent,
            "test_patient_agent",
            "Patient agent creation",
        )

    except Exception as e:
        log_test_result(
            "test_patient_creation_basic",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_patient_creation_full():
    """Test patient creation with complete medical data."""
    logger.info("Testing full patient creation...")

    try:
        symptoms = ["chest pain", "shortness of breath", "sweating"]
        medical_history = ["hypertension", "diabetes"]
        medications = ["metformin", "lisinopril"]
        allergies = ["penicillin", "shellfish"]
        vital_signs = {
            "blood_pressure": {"systolic": 150, "diastolic": 90},
            "heart_rate": 110,
            "temperature": 98.6,
        }

        patient = Patient(
            name="Jane Smith",
            age=45,
            gender="Female",
            chief_complaint="Chest pain",
            symptoms=symptoms,
            medical_history=medical_history,
            current_medications=medications,
            allergies=allergies,
            vital_signs=vital_signs,
        )

        assert_equal(
            patient.symptoms,
            symptoms,
            "test_patient_symptoms",
            "Patient symptoms list",
        )
        assert_equal(
            patient.medical_history,
            medical_history,
            "test_patient_history",
            "Patient medical history",
        )
        assert_equal(
            patient.current_medications,
            medications,
            "test_patient_medications",
            "Patient medications",
        )
        assert_equal(
            patient.allergies,
            allergies,
            "test_patient_allergies",
            "Patient allergies",
        )
        assert_equal(
            patient.vital_signs,
            vital_signs,
            "test_patient_vitals",
            "Patient vital signs",
        )
        assert_true(
            patient.priority_score > 0,
            "test_patient_priority_calculated",
            "Priority score calculation",
        )

    except Exception as e:
        log_test_result(
            "test_patient_creation_full",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_patient_priority_calculation():
    """Test patient priority score calculation with different symptom combinations."""
    logger.info("Testing patient priority calculations...")

    # Test emergency symptoms
    try:
        emergency_patient = Patient(
            name="Emergency Patient",
            age=60,
            symptoms=[
                "chest pain",
                "shortness of breath",
                "unconscious",
            ],
            vital_signs={
                "blood_pressure": {"systolic": 200},
                "heart_rate": 130,
            },
        )
        assert_true(
            emergency_patient.priority_score >= 15,
            "test_emergency_priority",
            "Emergency patient priority",
        )

    except Exception as e:
        log_test_result(
            "test_emergency_priority",
            False,
            "",
            f"Exception: {str(e)}",
        )

    # Test moderate symptoms
    try:
        moderate_patient = Patient(
            name="Moderate Patient",
            age=30,
            symptoms=["pain", "fever"],
            vital_signs={"temperature": 102},
        )
        assert_in_range(
            moderate_patient.priority_score,
            5,
            15,
            "test_moderate_priority",
            "Moderate patient priority",
        )

    except Exception as e:
        log_test_result(
            "test_moderate_priority",
            False,
            "",
            f"Exception: {str(e)}",
        )

    # Test low priority symptoms
    try:
        low_patient = Patient(
            name="Low Priority Patient",
            age=25,
            symptoms=["minor headache"],
            vital_signs={
                "blood_pressure": {"systolic": 120},
                "heart_rate": 70,
            },
        )
        assert_true(
            low_patient.priority_score < 10,
            "test_low_priority",
            "Low priority patient",
        )

    except Exception as e:
        log_test_result(
            "test_low_priority", False, "", f"Exception: {str(e)}"
        )


def test_patient_serialization():
    """Test patient to_dict and from_dict methods."""
    logger.info("Testing patient serialization...")

    try:
        original_patient = Patient(
            name="Serialization Test",
            age=40,
            gender="Female",
            chief_complaint="Test complaint",
            symptoms=["symptom1", "symptom2"],
            medical_history=["history1"],
            current_medications=["med1"],
            allergies=["allergy1"],
        )

        # Test to_dict
        patient_dict = original_patient.to_dict()
        assert_isinstance(
            patient_dict,
            dict,
            "test_patient_to_dict",
            "Patient to_dict returns dictionary",
        )
        assert_contains(
            patient_dict,
            "patient_id",
            "test_dict_contains_id",
            "Dictionary contains patient_id",
        )
        assert_contains(
            patient_dict,
            "name",
            "test_dict_contains_name",
            "Dictionary contains name",
        )
        assert_equal(
            patient_dict["name"],
            "Serialization Test",
            "test_dict_name_value",
            "Dictionary name value",
        )

        # Test from_dict
        recreated_patient = Patient.from_dict(patient_dict)
        assert_equal(
            recreated_patient.name,
            original_patient.name,
            "test_patient_from_dict_name",
            "Recreated patient name",
        )
        assert_equal(
            recreated_patient.age,
            original_patient.age,
            "test_patient_from_dict_age",
            "Recreated patient age",
        )
        assert_equal(
            recreated_patient.symptoms,
            original_patient.symptoms,
            "test_patient_from_dict_symptoms",
            "Recreated patient symptoms",
        )

    except Exception as e:
        log_test_result(
            "test_patient_serialization",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_patient_agent_creation():
    """Test patient agent creation and system prompt generation."""
    logger.info("Testing patient agent creation...")

    try:
        patient = Patient(
            name="Agent Test Patient",
            age=50,
            gender="Male",
            chief_complaint="Testing agent creation",
            symptoms=["test symptom"],
        )

        assert_not_none(
            patient.agent, "test_agent_exists", "Patient agent exists"
        )
        assert_isinstance(
            patient.agent,
            Agent,
            "test_agent_type",
            "Patient agent is Agent instance",
        )
        assert_true(
            len(patient.system_prompt) > 0,
            "test_system_prompt",
            "System prompt generated",
        )
        assert_contains(
            patient.system_prompt,
            patient.name,
            "test_prompt_contains_name",
            "System prompt contains patient name",
        )
        assert_contains(
            patient.system_prompt,
            patient.chief_complaint,
            "test_prompt_contains_complaint",
            "System prompt contains chief complaint",
        )

    except Exception as e:
        log_test_result(
            "test_patient_agent_creation",
            False,
            "",
            f"Exception: {str(e)}",
        )


# =============================================================================
# EHR SYSTEM TESTS
# =============================================================================


def test_ehr_system_initialization():
    """Test EHR system initialization with both ChromaDB and memory storage."""
    logger.info("Testing EHR system initialization...")

    try:
        ehr = EHRSystem(
            collection_name="test_ehr",
            persist_directory="./test_data",
        )

        assert_not_none(
            ehr, "test_ehr_creation", "EHR system created"
        )
        assert_equal(
            ehr.collection_name,
            "test_ehr",
            "test_ehr_collection_name",
            "EHR collection name",
        )
        assert_equal(
            ehr.persist_directory,
            "./test_data",
            "test_ehr_persist_dir",
            "EHR persist directory",
        )

        if CHROMADB_AVAILABLE:
            assert_not_none(
                ehr.client,
                "test_ehr_chromadb_client",
                "ChromaDB client initialized",
            )
            assert_not_none(
                ehr.collection,
                "test_ehr_chromadb_collection",
                "ChromaDB collection initialized",
            )
        else:
            assert_not_none(
                ehr.memory_storage,
                "test_ehr_memory_storage",
                "Memory storage initialized",
            )
            assert_isinstance(
                ehr.memory_storage,
                dict,
                "test_ehr_memory_type",
                "Memory storage is dictionary",
            )

    except Exception as e:
        log_test_result(
            "test_ehr_system_initialization",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_ehr_add_patient_record():
    """Test adding patient records to EHR system."""
    logger.info("Testing EHR patient record addition...")

    try:
        ehr = EHRSystem(collection_name="test_add_record")

        patient = Patient(
            name="EHR Test Patient",
            age=35,
            gender="Female",
            chief_complaint="EHR Testing",
            symptoms=["test symptom"],
            medical_history=["test history"],
        )

        medical_notes = "Test medical notes for EHR system"
        doctor_name = "Dr. Test"

        record_id = ehr.add_patient_record(
            patient, medical_notes, doctor_name
        )

        assert_not_none(
            record_id, "test_ehr_record_id", "EHR record ID returned"
        )
        assert_true(
            record_id.startswith("record_"),
            "test_ehr_record_format",
            "Record ID has correct format",
        )

        # Verify record can be retrieved
        patient_history = ehr.query_patient_history(
            patient.patient_id
        )
        assert_true(
            len(patient_history) > 0,
            "test_ehr_record_retrieval",
            "Patient record can be retrieved",
        )

        if patient_history:
            record = patient_history[0]
            assert_contains(
                record,
                "content",
                "test_ehr_record_content",
                "Record contains content",
            )
            assert_contains(
                record,
                "metadata",
                "test_ehr_record_metadata",
                "Record contains metadata",
            )
            assert_contains(
                record["content"],
                medical_notes,
                "test_ehr_content_notes",
                "Record content contains medical notes",
            )
            assert_contains(
                record["content"],
                patient.name,
                "test_ehr_content_name",
                "Record content contains patient name",
            )

    except Exception as e:
        log_test_result(
            "test_ehr_add_patient_record",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_ehr_query_patient_history():
    """Test querying patient history from EHR system."""
    logger.info("Testing EHR patient history queries...")

    try:
        ehr = EHRSystem(collection_name="test_query_history")

        # Create and add multiple records for same patient
        patient = Patient(
            name="History Test Patient",
            age=40,
            gender="Male",
            chief_complaint="Multiple visits",
        )

        # Add first record
        ehr.add_patient_record(patient, "First visit notes", "Dr. A")
        time.sleep(0.1)  # Small delay to ensure different timestamps

        # Add second record
        ehr.add_patient_record(patient, "Second visit notes", "Dr. B")

        # Query all history
        history = ehr.query_patient_history(patient.patient_id)
        assert_true(
            len(history) >= 2,
            "test_ehr_multiple_records",
            "Multiple records retrieved",
        )

        # Query with specific search term
        search_history = ehr.query_patient_history(
            patient.patient_id, "First visit"
        )
        assert_true(
            len(search_history) >= 1,
            "test_ehr_search_query",
            "Search query returns results",
        )

        if search_history:
            assert_contains(
                search_history[0]["content"],
                "First visit",
                "test_ehr_search_content",
                "Search result contains search term",
            )

    except Exception as e:
        log_test_result(
            "test_ehr_query_patient_history",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_ehr_search_similar_cases():
    """Test searching for similar medical cases in EHR system."""
    logger.info("Testing EHR similar cases search...")

    try:
        ehr = EHRSystem(collection_name="test_similar_cases")

        # Add multiple patients with similar symptoms
        symptoms_list = [
            ["chest pain", "shortness of breath"],
            ["chest pain", "sweating"],
            ["headache", "nausea"],
            ["fever", "cough"],
        ]

        for i, symptoms in enumerate(symptoms_list):
            patient = Patient(
                name=f"Similar Case Patient {i+1}",
                age=30 + i,
                symptoms=symptoms,
                chief_complaint=f"Case {i+1}",
            )
            ehr.add_patient_record(
                patient, f"Medical notes for case {i+1}", f"Dr. {i+1}"
            )

        # Search for similar cases
        similar_cases = ehr.search_similar_cases(
            ["chest pain", "shortness of breath"]
        )
        assert_true(
            len(similar_cases) > 0,
            "test_ehr_similar_found",
            "Similar cases found",
        )

        # Search with diagnosis
        similar_with_diagnosis = ehr.search_similar_cases(
            ["headache"], "migraine"
        )
        assert_isinstance(
            similar_with_diagnosis,
            list,
            "test_ehr_similar_with_diagnosis",
            "Similar cases with diagnosis returns list",
        )

    except Exception as e:
        log_test_result(
            "test_ehr_search_similar_cases",
            False,
            "",
            f"Exception: {str(e)}",
        )


# =============================================================================
# HOSPITAL STAFF TESTS
# =============================================================================


def test_hospital_staff_creation():
    """Test hospital staff creation and initialization."""
    logger.info("Testing hospital staff creation...")

    try:
        # Create a mock agent
        agent = Agent(
            agent_name="Test Agent",
            system_prompt="Test system prompt",
            max_loops=1,
        )

        staff = HospitalStaff("Dr. Test", StaffRole.DOCTOR, agent)

        assert_equal(
            staff.name,
            "Dr. Test",
            "test_staff_name",
            "Staff name assignment",
        )
        assert_equal(
            staff.role,
            StaffRole.DOCTOR,
            "test_staff_role",
            "Staff role assignment",
        )
        assert_equal(
            staff.agent,
            agent,
            "test_staff_agent",
            "Staff agent assignment",
        )
        assert_true(
            staff.is_available,
            "test_staff_availability",
            "Staff initial availability",
        )
        assert_equal(
            staff.current_patient,
            None,
            "test_staff_no_patient",
            "Staff initially has no patient",
        )
        assert_equal(
            len(staff.patients_seen),
            0,
            "test_staff_patients_seen",
            "Staff initially has seen no patients",
        )
        assert_isinstance(
            staff.performance_metrics,
            dict,
            "test_staff_metrics",
            "Staff has performance metrics",
        )

    except Exception as e:
        log_test_result(
            "test_hospital_staff_creation",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_hospital_staff_patient_assignment():
    """Test patient assignment and release for hospital staff."""
    logger.info("Testing hospital staff patient assignment...")

    try:
        agent = Agent(
            agent_name="Test Agent", system_prompt="Test", max_loops=1
        )
        staff = HospitalStaff("Nurse Test", StaffRole.NURSE, agent)
        patient = Patient(
            name="Test Patient", age=30, chief_complaint="Test"
        )

        # Test successful assignment
        result = staff.assign_patient(patient)
        assert_true(
            result,
            "test_staff_assign_success",
            "Patient assignment successful",
        )
        assert_equal(
            staff.current_patient,
            patient,
            "test_staff_current_patient",
            "Current patient set correctly",
        )
        assert_true(
            not staff.is_available,
            "test_staff_unavailable",
            "Staff becomes unavailable",
        )

        # Test assignment when already busy
        patient2 = Patient(
            name="Test Patient 2", age=25, chief_complaint="Test 2"
        )
        result2 = staff.assign_patient(patient2)
        assert_true(
            not result2,
            "test_staff_assign_fail",
            "Second assignment fails when busy",
        )

        # Test patient release
        staff.release_patient()
        assert_equal(
            staff.current_patient,
            None,
            "test_staff_release_patient",
            "Current patient cleared",
        )
        assert_true(
            staff.is_available,
            "test_staff_available_after_release",
            "Staff becomes available",
        )
        assert_equal(
            len(staff.patients_seen),
            1,
            "test_staff_patients_seen_count",
            "Patients seen count updated",
        )
        assert_contains(
            staff.patients_seen,
            patient,
            "test_staff_patients_seen_list",
            "Released patient in seen list",
        )

    except Exception as e:
        log_test_result(
            "test_hospital_staff_patient_assignment",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_hospital_staff_metrics_update():
    """Test hospital staff performance metrics updates."""
    logger.info("Testing hospital staff metrics updates...")

    try:
        agent = Agent(
            agent_name="Test Agent", system_prompt="Test", max_loops=1
        )
        staff = HospitalStaff("Dr. Metrics", StaffRole.DOCTOR, agent)

        # Initial metrics
        assert_equal(
            staff.performance_metrics["patients_treated"],
            0,
            "test_metrics_initial_patients",
            "Initial patients treated",
        )
        assert_equal(
            staff.performance_metrics["average_treatment_time"],
            0.0,
            "test_metrics_initial_time",
            "Initial average time",
        )

        # Update metrics - first patient
        staff.update_metrics(
            30.0, 8.5
        )  # 30 minutes, satisfaction 8.5/10
        assert_equal(
            staff.performance_metrics["patients_treated"],
            1,
            "test_metrics_first_patient",
            "First patient treated count",
        )
        assert_equal(
            staff.performance_metrics["average_treatment_time"],
            30.0,
            "test_metrics_first_time",
            "First patient treatment time",
        )
        assert_equal(
            staff.performance_metrics["patient_satisfaction"],
            8.5,
            "test_metrics_first_satisfaction",
            "First patient satisfaction",
        )

        # Update metrics - second patient
        staff.update_metrics(
            20.0, 9.0
        )  # 20 minutes, satisfaction 9.0/10
        assert_equal(
            staff.performance_metrics["patients_treated"],
            2,
            "test_metrics_second_patient",
            "Second patient treated count",
        )
        assert_equal(
            staff.performance_metrics["average_treatment_time"],
            25.0,
            "test_metrics_average_time",
            "Average treatment time calculation",
        )
        assert_equal(
            staff.performance_metrics["patient_satisfaction"],
            8.75,
            "test_metrics_average_satisfaction",
            "Average satisfaction calculation",
        )

        # Update metrics without satisfaction score
        staff.update_metrics(15.0)  # Only treatment time
        assert_equal(
            staff.performance_metrics["patients_treated"],
            3,
            "test_metrics_third_patient",
            "Third patient treated count",
        )
        expected_avg_time = (30.0 + 20.0 + 15.0) / 3
        assert_equal(
            staff.performance_metrics["average_treatment_time"],
            expected_avg_time,
            "test_metrics_time_only",
            "Treatment time only update",
        )

    except Exception as e:
        log_test_result(
            "test_hospital_staff_metrics_update",
            False,
            "",
            f"Exception: {str(e)}",
        )


# =============================================================================
# PATIENT QUEUE TESTS
# =============================================================================


def test_patient_queue_creation():
    """Test patient queue creation and initialization."""
    logger.info("Testing patient queue creation...")

    try:
        queue = PatientQueue()

        assert_not_none(
            queue.queue,
            "test_queue_priority_queue",
            "Priority queue exists",
        )
        assert_isinstance(
            queue.waiting_patients,
            list,
            "test_queue_waiting_list",
            "Waiting patients list exists",
        )
        assert_isinstance(
            queue.treatment_history,
            list,
            "test_queue_treatment_history",
            "Treatment history list exists",
        )
        assert_equal(
            len(queue.waiting_patients),
            0,
            "test_queue_initial_empty",
            "Queue initially empty",
        )
        assert_true(
            queue.queue.empty(),
            "test_priority_queue_empty",
            "Priority queue initially empty",
        )

    except Exception as e:
        log_test_result(
            "test_patient_queue_creation",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_patient_queue_add_patient():
    """Test adding patients to the queue."""
    logger.info("Testing patient queue add functionality...")

    try:
        queue = PatientQueue()

        # Create patients with different priorities
        emergency_patient = Patient(
            name="Emergency Patient",
            age=50,
            symptoms=["chest pain", "shortness of breath"],
            chief_complaint="Emergency",
        )

        regular_patient = Patient(
            name="Regular Patient",
            age=30,
            symptoms=["headache"],
            chief_complaint="Headache",
        )

        # Add patients to queue
        queue.add_patient(emergency_patient)
        queue.add_patient(regular_patient)

        assert_equal(
            len(queue.waiting_patients),
            2,
            "test_queue_two_patients",
            "Two patients in waiting list",
        )
        assert_equal(
            queue.queue.qsize(),
            2,
            "test_priority_queue_size",
            "Priority queue has two patients",
        )
        assert_contains(
            queue.waiting_patients,
            emergency_patient,
            "test_queue_contains_emergency",
            "Queue contains emergency patient",
        )
        assert_contains(
            queue.waiting_patients,
            regular_patient,
            "test_queue_contains_regular",
            "Queue contains regular patient",
        )

    except Exception as e:
        log_test_result(
            "test_patient_queue_add_patient",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_patient_queue_priority_order():
    """Test that patients are retrieved in priority order."""
    logger.info("Testing patient queue priority ordering...")

    try:
        queue = PatientQueue()

        # Create patients with different priorities (higher priority = more urgent)
        low_priority = Patient(
            name="Low Priority", age=25, symptoms=["minor headache"]
        )
        medium_priority = Patient(
            name="Medium Priority", age=35, symptoms=["pain", "fever"]
        )
        high_priority = Patient(
            name="High Priority",
            age=45,
            symptoms=["chest pain", "shortness of breath"],
        )

        # Add in non-priority order
        queue.add_patient(low_priority)
        queue.add_patient(medium_priority)
        queue.add_patient(high_priority)

        # Retrieve patients - should come out in priority order
        first_patient = queue.get_next_patient()
        assert_equal(
            first_patient.name,
            "High Priority",
            "test_queue_first_priority",
            "Highest priority patient first",
        )

        second_patient = queue.get_next_patient()
        assert_equal(
            second_patient.name,
            "Medium Priority",
            "test_queue_second_priority",
            "Medium priority patient second",
        )

        third_patient = queue.get_next_patient()
        assert_equal(
            third_patient.name,
            "Low Priority",
            "test_queue_third_priority",
            "Low priority patient third",
        )

        # Queue should be empty now
        fourth_patient = queue.get_next_patient()
        assert_equal(
            fourth_patient,
            None,
            "test_queue_empty_after",
            "Queue empty after all patients retrieved",
        )

    except Exception as e:
        log_test_result(
            "test_patient_queue_priority_order",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_patient_queue_status():
    """Test patient queue status reporting."""
    logger.info("Testing patient queue status...")

    try:
        queue = PatientQueue()

        # Add some patients
        patients = [
            Patient(name="Patient 1", age=30, symptoms=["headache"]),
            Patient(
                name="Patient 2", age=40, symptoms=["chest pain"]
            ),
            Patient(name="Patient 3", age=35, symptoms=["fever"]),
        ]

        for patient in patients:
            queue.add_patient(patient)

        status = queue.get_queue_status()

        assert_isinstance(
            status,
            dict,
            "test_queue_status_type",
            "Queue status is dictionary",
        )
        assert_contains(
            status,
            "total_waiting",
            "test_queue_status_total",
            "Status contains total waiting",
        )
        assert_contains(
            status,
            "queue_length",
            "test_queue_status_length",
            "Status contains queue length",
        )
        assert_contains(
            status,
            "waiting_patients",
            "test_queue_status_patients",
            "Status contains waiting patients",
        )
        assert_contains(
            status,
            "estimated_wait_times",
            "test_queue_status_wait_times",
            "Status contains wait times",
        )

        assert_equal(
            status["total_waiting"],
            3,
            "test_queue_status_count",
            "Correct waiting count",
        )
        assert_equal(
            len(status["waiting_patients"]),
            3,
            "test_queue_status_patient_list",
            "Correct patient list length",
        )
        assert_isinstance(
            status["estimated_wait_times"],
            dict,
            "test_queue_wait_times_type",
            "Wait times is dictionary",
        )

    except Exception as e:
        log_test_result(
            "test_patient_queue_status",
            False,
            "",
            f"Exception: {str(e)}",
        )


# =============================================================================
# HOSPITAL SIMULATION TESTS
# =============================================================================


def test_hospital_simulation_creation():
    """Test hospital simulation creation and initialization."""
    logger.info("Testing hospital simulation creation...")

    try:
        hospital = HospitalSimulation(
            hospital_name="Test Hospital",
            description="Test hospital for unit tests",
            random_models_on=False,
        )

        assert_equal(
            hospital.hospital_name,
            "Test Hospital",
            "test_hospital_name",
            "Hospital name set correctly",
        )
        assert_equal(
            hospital.description,
            "Test hospital for unit tests",
            "test_hospital_description",
            "Hospital description set",
        )
        assert_true(
            not hospital.random_models_on,
            "test_hospital_random_models",
            "Random models setting",
        )

        # Check EHR system
        assert_not_none(
            hospital.ehr_system,
            "test_hospital_ehr",
            "EHR system initialized",
        )
        assert_isinstance(
            hospital.ehr_system,
            EHRSystem,
            "test_hospital_ehr_type",
            "EHR system is correct type",
        )

        # Check patient queue
        assert_not_none(
            hospital.patient_queue,
            "test_hospital_queue",
            "Patient queue initialized",
        )
        assert_isinstance(
            hospital.patient_queue,
            PatientQueue,
            "test_hospital_queue_type",
            "Patient queue is correct type",
        )

        # Check staff initialization
        assert_isinstance(
            hospital.staff,
            dict,
            "test_hospital_staff_dict",
            "Staff dictionary exists",
        )
        assert_isinstance(
            hospital.executives,
            list,
            "test_hospital_executives",
            "Executives list exists",
        )
        assert_isinstance(
            hospital.doctors,
            list,
            "test_hospital_doctors",
            "Doctors list exists",
        )
        assert_isinstance(
            hospital.nurses,
            list,
            "test_hospital_nurses",
            "Nurses list exists",
        )
        assert_isinstance(
            hospital.receptionists,
            list,
            "test_hospital_receptionists",
            "Receptionists list exists",
        )

        # Check simulation state
        assert_true(
            not hospital.is_running,
            "test_hospital_not_running",
            "Hospital initially not running",
        )
        assert_isinstance(
            hospital.simulation_stats,
            dict,
            "test_hospital_stats",
            "Simulation stats initialized",
        )

    except Exception as e:
        log_test_result(
            "test_hospital_simulation_creation",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_hospital_staff_initialization():
    """Test that hospital staff are properly initialized."""
    logger.info("Testing hospital staff initialization...")

    try:
        hospital = HospitalSimulation()

        # Check that staff are created
        assert_true(
            len(hospital.executives) > 0,
            "test_executives_created",
            "Executives created",
        )
        assert_true(
            len(hospital.doctors) > 0,
            "test_doctors_created",
            "Doctors created",
        )
        assert_true(
            len(hospital.nurses) > 0,
            "test_nurses_created",
            "Nurses created",
        )
        assert_true(
            len(hospital.receptionists) > 0,
            "test_receptionists_created",
            "Receptionists created",
        )

        # Check staff types and roles
        for executive in hospital.executives:
            assert_isinstance(
                executive,
                HospitalStaff,
                "test_executive_type",
                "Executive is HospitalStaff",
            )
            assert_equal(
                executive.role,
                StaffRole.EXECUTIVE,
                "test_executive_role",
                "Executive has correct role",
            )
            assert_not_none(
                executive.agent,
                "test_executive_agent",
                "Executive has agent",
            )

        for doctor in hospital.doctors:
            assert_isinstance(
                doctor,
                HospitalStaff,
                "test_doctor_type",
                "Doctor is HospitalStaff",
            )
            assert_equal(
                doctor.role,
                StaffRole.DOCTOR,
                "test_doctor_role",
                "Doctor has correct role",
            )
            assert_not_none(
                doctor.agent, "test_doctor_agent", "Doctor has agent"
            )

        for nurse in hospital.nurses:
            assert_isinstance(
                nurse,
                HospitalStaff,
                "test_nurse_type",
                "Nurse is HospitalStaff",
            )
            assert_equal(
                nurse.role,
                StaffRole.NURSE,
                "test_nurse_role",
                "Nurse has correct role",
            )
            assert_not_none(
                nurse.agent, "test_nurse_agent", "Nurse has agent"
            )

        for receptionist in hospital.receptionists:
            assert_isinstance(
                receptionist,
                HospitalStaff,
                "test_receptionist_type",
                "Receptionist is HospitalStaff",
            )
            assert_equal(
                receptionist.role,
                StaffRole.RECEPTIONIST,
                "test_receptionist_role",
                "Receptionist has correct role",
            )
            assert_not_none(
                receptionist.agent,
                "test_receptionist_agent",
                "Receptionist has agent",
            )

        # Check staff availability
        for staff_member in hospital.staff.values():
            assert_true(
                staff_member.is_available,
                "test_staff_initial_availability",
                "Staff initially available",
            )
            assert_equal(
                staff_member.current_patient,
                None,
                "test_staff_no_initial_patient",
                "Staff has no initial patient",
            )

    except Exception as e:
        log_test_result(
            "test_hospital_staff_initialization",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_hospital_add_patient():
    """Test adding patients to the hospital."""
    logger.info("Testing hospital add patient functionality...")

    try:
        hospital = HospitalSimulation()

        patient = Patient(
            name="Hospital Test Patient",
            age=45,
            gender="Male",
            chief_complaint="Testing hospital admission",
            symptoms=["test symptom"],
        )

        initial_patient_count = hospital.simulation_stats[
            "total_patients"
        ]
        initial_queue_size = len(
            hospital.patient_queue.waiting_patients
        )

        # Add patient to hospital
        hospital.add_patient(patient)

        # Check that patient was added to queue
        assert_equal(
            len(hospital.patient_queue.waiting_patients),
            initial_queue_size + 1,
            "test_patient_added_to_queue",
            "Patient added to queue",
        )

        # Check that patient count increased
        assert_equal(
            hospital.simulation_stats["total_patients"],
            initial_patient_count + 1,
            "test_patient_count_increased",
            "Patient count increased",
        )

        # Check that patient is in queue
        assert_contains(
            hospital.patient_queue.waiting_patients,
            patient,
            "test_patient_in_queue",
            "Patient is in waiting queue",
        )

    except Exception as e:
        log_test_result(
            "test_hospital_add_patient",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_hospital_run_method():
    """Test hospital run method with patient record."""
    logger.info("Testing hospital run method...")

    try:
        hospital = HospitalSimulation()

        patient_record = {
            "name": "Run Test Patient",
            "age": 35,
            "gender": "Female",
            "chief_complaint": "Testing run method",
            "symptoms": ["test symptom"],
            "medical_history": ["test history"],
            "current_medications": ["test medication"],
            "allergies": ["test allergy"],
        }

        # Run hospital pipeline for patient
        result = hospital.run(patient_record)

        # Check result structure
        assert_isinstance(
            result,
            dict,
            "test_run_result_type",
            "Run result is dictionary",
        )
        assert_contains(
            result,
            "patient_id",
            "test_run_result_id",
            "Result contains patient ID",
        )
        assert_contains(
            result,
            "patient_name",
            "test_run_result_name",
            "Result contains patient name",
        )
        assert_contains(
            result,
            "status",
            "test_run_result_status",
            "Result contains status",
        )

        # Check patient name in result
        assert_equal(
            result["patient_name"],
            "Run Test Patient",
            "test_run_patient_name",
            "Correct patient name in result",
        )

        # Check that result has required fields
        expected_fields = [
            "reception",
            "triage",
            "consultation",
            "treatment",
            "ehr_record_id",
        ]
        for field in expected_fields:
            assert_contains(
                result,
                field,
                f"test_run_result_{field}",
                f"Result contains {field}",
            )

    except Exception as e:
        log_test_result(
            "test_hospital_run_method",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_hospital_generate_patients():
    """Test hospital patient generation functionality."""
    logger.info("Testing hospital patient generation...")

    try:
        hospital = HospitalSimulation()

        initial_count = hospital.simulation_stats["total_patients"]
        initial_queue_size = len(
            hospital.patient_queue.waiting_patients
        )

        # Generate 3 patients
        hospital.generate_patients(3)

        # Check that patients were added
        assert_equal(
            hospital.simulation_stats["total_patients"],
            initial_count + 3,
            "test_generate_patient_count",
            "Correct number of patients generated",
        )

        assert_equal(
            len(hospital.patient_queue.waiting_patients),
            initial_queue_size + 3,
            "test_generate_queue_size",
            "Patients added to queue",
        )

        # Check that generated patients have proper data
        if len(hospital.patient_queue.waiting_patients) > 0:
            sample_patient = hospital.patient_queue.waiting_patients[
                0
            ]
            assert_not_none(
                sample_patient.name,
                "test_generated_patient_name",
                "Generated patient has name",
            )
            assert_true(
                sample_patient.age > 0,
                "test_generated_patient_age",
                "Generated patient has valid age",
            )
            assert_true(
                len(sample_patient.symptoms) > 0,
                "test_generated_patient_symptoms",
                "Generated patient has symptoms",
            )

    except Exception as e:
        log_test_result(
            "test_hospital_generate_patients",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_hospital_statistics_update():
    """Test hospital statistics updating."""
    logger.info("Testing hospital statistics update...")

    try:
        hospital = HospitalSimulation()

        # Set some test data
        hospital.simulation_stats["patients_treated"] = 5
        hospital.total_wait_time = 150.0  # 150 minutes total

        # Update statistics
        hospital.update_statistics()

        # Check calculated values
        expected_avg_wait = 150.0 / 5  # 30 minutes average
        assert_equal(
            hospital.simulation_stats["average_wait_time"],
            expected_avg_wait,
            "test_stats_average_wait",
            "Average wait time calculated correctly",
        )

        expected_revenue = 5 * 150  # $150 per patient
        assert_equal(
            hospital.simulation_stats["revenue"],
            expected_revenue,
            "test_stats_revenue",
            "Revenue calculated correctly",
        )

        expected_costs = 5 * 100  # $100 per patient
        assert_equal(
            hospital.simulation_stats["costs"],
            expected_costs,
            "test_stats_costs",
            "Costs calculated correctly",
        )

        # Check patient satisfaction calculation
        satisfaction = hospital.simulation_stats[
            "patient_satisfaction"
        ]
        assert_true(
            satisfaction >= 0,
            "test_stats_satisfaction_min",
            "Patient satisfaction >= 0",
        )
        assert_true(
            satisfaction <= 100,
            "test_stats_satisfaction_max",
            "Patient satisfaction <= 100",
        )

    except Exception as e:
        log_test_result(
            "test_hospital_statistics_update",
            False,
            "",
            f"Exception: {str(e)}",
        )


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================


def test_patient_invalid_data():
    """Test patient creation with invalid or edge case data."""
    logger.info("Testing patient creation with invalid data...")

    try:
        # Test with negative age
        patient_neg_age = Patient(
            name="Negative Age", age=-5, chief_complaint="Test"
        )
        assert_equal(
            patient_neg_age.age,
            -5,
            "test_patient_negative_age",
            "Negative age handled",
        )

        # Test with very high age
        patient_old = Patient(
            name="Very Old", age=150, chief_complaint="Test"
        )
        assert_equal(
            patient_old.age,
            150,
            "test_patient_old_age",
            "Very high age handled",
        )

        # Test with empty name
        patient_no_name = Patient(
            name="", age=30, chief_complaint="Test"
        )
        assert_equal(
            patient_no_name.name,
            "",
            "test_patient_empty_name",
            "Empty name handled",
        )

        # Test with empty symptoms list
        patient_no_symptoms = Patient(
            name="No Symptoms", age=30, symptoms=[]
        )
        assert_equal(
            len(patient_no_symptoms.symptoms),
            0,
            "test_patient_no_symptoms",
            "Empty symptoms list handled",
        )

        # Test with very long lists
        long_list = ["item"] * 100
        patient_long_lists = Patient(
            name="Long Lists",
            age=30,
            symptoms=long_list,
            medical_history=long_list,
            current_medications=long_list,
            allergies=long_list,
        )
        assert_equal(
            len(patient_long_lists.symptoms),
            100,
            "test_patient_long_symptoms",
            "Long symptoms list handled",
        )

    except Exception as e:
        log_test_result(
            "test_patient_invalid_data",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_ehr_system_edge_cases():
    """Test EHR system with edge cases."""
    logger.info("Testing EHR system edge cases...")

    try:
        ehr = EHRSystem(collection_name="test_edge_cases")

        # Test with patient with minimal data
        minimal_patient = Patient(name="Minimal", age=0)
        minimal_record_id = ehr.add_patient_record(
            minimal_patient, "", ""
        )
        assert_not_none(
            minimal_record_id,
            "test_ehr_minimal_patient",
            "EHR handles minimal patient data",
        )

        # Test querying non-existent patient
        empty_history = ehr.query_patient_history("non-existent-id")
        assert_isinstance(
            empty_history,
            list,
            "test_ehr_nonexistent_patient",
            "Non-existent patient query returns list",
        )
        assert_equal(
            len(empty_history),
            0,
            "test_ehr_empty_history",
            "Non-existent patient has empty history",
        )

        # Test searching with empty symptoms
        empty_search = ehr.search_similar_cases([])
        assert_isinstance(
            empty_search,
            list,
            "test_ehr_empty_symptoms_search",
            "Empty symptoms search returns list",
        )

        # Test with very long medical notes
        long_notes = "Very long medical notes. " * 1000
        long_record_id = ehr.add_patient_record(
            minimal_patient, long_notes, "Dr. Long"
        )
        assert_not_none(
            long_record_id,
            "test_ehr_long_notes",
            "EHR handles very long medical notes",
        )

    except Exception as e:
        log_test_result(
            "test_ehr_system_edge_cases",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_hospital_staff_edge_cases():
    """Test hospital staff with edge cases."""
    logger.info("Testing hospital staff edge cases...")

    try:
        agent = Agent(
            agent_name="Test Agent", system_prompt="Test", max_loops=1
        )
        staff = HospitalStaff(
            "Edge Case Staff", StaffRole.DOCTOR, agent
        )

        # Test releasing patient when none assigned
        staff.release_patient()  # Should not cause error
        assert_equal(
            staff.current_patient,
            None,
            "test_staff_release_none",
            "Release when no patient assigned",
        )

        # Test updating metrics with zero values
        staff.update_metrics(0.0, 0.0)
        assert_equal(
            staff.performance_metrics["patients_treated"],
            1,
            "test_staff_zero_metrics",
            "Zero metrics handled",
        )

        # Test updating metrics with very large values
        staff.update_metrics(10000.0, 10.0)
        assert_true(
            staff.performance_metrics["patients_treated"] == 2,
            "test_staff_large_metrics",
            "Large metrics handled",
        )

        # Test updating metrics with negative values
        staff.update_metrics(-5.0, -1.0)
        assert_equal(
            staff.performance_metrics["patients_treated"],
            3,
            "test_staff_negative_metrics",
            "Negative metrics handled",
        )

    except Exception as e:
        log_test_result(
            "test_hospital_staff_edge_cases",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_patient_queue_edge_cases():
    """Test patient queue with edge cases."""
    logger.info("Testing patient queue edge cases...")

    try:
        queue = PatientQueue()

        # Test getting patient from empty queue
        empty_result = queue.get_next_patient()
        assert_equal(
            empty_result,
            None,
            "test_queue_empty_get",
            "Getting from empty queue returns None",
        )

        # Test status of empty queue
        empty_status = queue.get_queue_status()
        assert_equal(
            empty_status["total_waiting"],
            0,
            "test_queue_empty_status",
            "Empty queue status correct",
        )

        # Test with patients having same priority
        same_priority_patients = [
            Patient(
                name=f"Same Priority {i}",
                age=30,
                symptoms=["headache"],
            )
            for i in range(3)
        ]

        for patient in same_priority_patients:
            queue.add_patient(patient)

        # All should have same priority, order should be maintained by timestamp
        first = queue.get_next_patient()
        second = queue.get_next_patient()
        third = queue.get_next_patient()

        assert_not_none(
            first,
            "test_queue_same_priority_first",
            "First patient with same priority retrieved",
        )
        assert_not_none(
            second,
            "test_queue_same_priority_second",
            "Second patient with same priority retrieved",
        )
        assert_not_none(
            third,
            "test_queue_same_priority_third",
            "Third patient with same priority retrieved",
        )

    except Exception as e:
        log_test_result(
            "test_patient_queue_edge_cases",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_hospital_simulation_error_handling():
    """Test hospital simulation error handling."""
    logger.info("Testing hospital simulation error handling...")

    try:
        hospital = HospitalSimulation()

        # Test run method with invalid patient record
        invalid_record = {}  # Empty record
        result = hospital.run(invalid_record)

        # Should handle gracefully and return error result
        assert_isinstance(
            result,
            dict,
            "test_hospital_invalid_record_type",
            "Invalid record returns dict",
        )
        # Note: The current implementation might still create a patient with empty fields

        # Test run method with malformed patient record
        malformed_record = {
            "name": None,
            "age": "not_a_number",
            "symptoms": "not_a_list",
        }

        try:
            malformed_result = hospital.run(malformed_record)
            assert_isinstance(
                malformed_result,
                dict,
                "test_hospital_malformed_record",
                "Malformed record handled",
            )
        except Exception:
            # If exception occurs, that's also acceptable for malformed data
            log_test_result(
                "test_hospital_malformed_record",
                True,
                "Exception raised for malformed data as expected",
            )

        # Test with patient record containing very long strings
        long_string = "x" * 10000
        long_record = {
            "name": long_string,
            "chief_complaint": long_string,
            "symptoms": [long_string] * 100,
        }

        long_result = hospital.run(long_record)
        assert_isinstance(
            long_result,
            dict,
            "test_hospital_long_strings",
            "Very long strings handled",
        )

    except Exception as e:
        log_test_result(
            "test_hospital_simulation_error_handling",
            False,
            "",
            f"Exception: {str(e)}",
        )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_end_to_end_patient_flow():
    """Test complete patient flow from admission to discharge."""
    logger.info("Testing end-to-end patient flow...")

    try:
        hospital = HospitalSimulation()

        # Create a comprehensive patient
        patient_record = {
            "name": "Integration Test Patient",
            "age": 45,
            "gender": "Female",
            "chief_complaint": "Chest pain and shortness of breath",
            "symptoms": [
                "chest pain",
                "shortness of breath",
                "sweating",
            ],
            "medical_history": ["hypertension"],
            "current_medications": ["lisinopril"],
            "allergies": ["penicillin"],
        }

        # Process patient through complete pipeline
        result = hospital.run(patient_record)

        # Verify complete flow
        assert_equal(
            result["status"],
            "completed",
            "test_e2e_status",
            "Patient flow completed successfully",
        )
        assert_not_none(
            result["patient_id"],
            "test_e2e_patient_id",
            "Patient ID assigned",
        )
        assert_not_none(
            result["ehr_record_id"],
            "test_e2e_ehr_record",
            "EHR record created",
        )
        assert_not_none(
            result["assigned_doctor"],
            "test_e2e_doctor_assigned",
            "Doctor assigned",
        )
        assert_not_none(
            result["assigned_nurse"],
            "test_e2e_nurse_assigned",
            "Nurse assigned",
        )

        # Verify priority was calculated correctly
        assert_true(
            result["priority_score"] > 0,
            "test_e2e_priority",
            "Priority score calculated",
        )

        # Verify treatment time is reasonable
        assert_true(
            result["total_time_minutes"] > 0,
            "test_e2e_treatment_time",
            "Treatment time recorded",
        )

        # Verify EHR record exists
        patient_history = hospital.ehr_system.query_patient_history(
            result["patient_id"]
        )
        assert_true(
            len(patient_history) > 0,
            "test_e2e_ehr_exists",
            "EHR record exists",
        )

        if patient_history:
            record_content = patient_history[0]["content"]
            assert_contains(
                record_content,
                "Integration Test Patient",
                "test_e2e_ehr_name",
                "EHR contains patient name",
            )
            assert_contains(
                record_content,
                "chest pain",
                "test_e2e_ehr_symptoms",
                "EHR contains symptoms",
            )

    except Exception as e:
        log_test_result(
            "test_end_to_end_patient_flow",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_multiple_patients_concurrency():
    """Test handling multiple patients simultaneously."""
    logger.info("Testing multiple patients handling...")

    try:
        hospital = HospitalSimulation()

        # Create multiple patients with different priorities
        patients = [
            {
                "name": "Emergency Patient 1",
                "age": 60,
                "chief_complaint": "Heart attack symptoms",
                "symptoms": ["chest pain", "shortness of breath"],
            },
            {
                "name": "Regular Patient 1",
                "age": 30,
                "chief_complaint": "Headache",
                "symptoms": ["headache"],
            },
            {
                "name": "Emergency Patient 2",
                "age": 45,
                "chief_complaint": "Severe bleeding",
                "symptoms": ["severe bleeding", "trauma"],
            },
            {
                "name": "Regular Patient 2",
                "age": 25,
                "chief_complaint": "Fever",
                "symptoms": ["fever", "cough"],
            },
        ]

        results = []
        for patient_record in patients:
            result = hospital.run(patient_record)
            results.append(result)

        # Verify all patients were processed
        assert_equal(
            len(results),
            4,
            "test_multi_patient_count",
            "All patients processed",
        )

        # Verify all have unique patient IDs
        patient_ids = [
            r["patient_id"] for r in results if "patient_id" in r
        ]
        unique_ids = set(patient_ids)
        assert_equal(
            len(unique_ids),
            len(patient_ids),
            "test_multi_patient_unique_ids",
            "All patient IDs are unique",
        )

        # Verify hospital statistics updated
        assert_true(
            hospital.simulation_stats["total_patients"] >= 4,
            "test_multi_patient_stats",
            "Hospital stats updated",
        )

        # Verify EHR records for all patients
        for result in results:
            if "patient_id" in result:
                history = hospital.ehr_system.query_patient_history(
                    result["patient_id"]
                )
                assert_true(
                    len(history) > 0,
                    f"test_multi_patient_ehr_{result.get('patient_name', 'unknown')}",
                    "EHR record exists for patient",
                )

    except Exception as e:
        log_test_result(
            "test_multiple_patients_concurrency",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_staff_workload_distribution():
    """Test that staff workload is distributed properly."""
    logger.info("Testing staff workload distribution...")

    try:
        hospital = HospitalSimulation()

        # Process multiple patients to see staff distribution
        patient_records = [
            {
                "name": f"Workload Patient {i}",
                "age": 30 + i,
                "chief_complaint": f"Complaint {i}",
            }
            for i in range(
                6
            )  # More patients than doctors to test distribution
        ]

        results = []
        for patient_record in patient_records:
            result = hospital.run(patient_record)
            results.append(result)

        # Collect assigned doctors
        assigned_doctors = [
            r.get("assigned_doctor")
            for r in results
            if r.get("assigned_doctor")
        ]
        doctor_counts = {}
        for doctor in assigned_doctors:
            doctor_counts[doctor] = doctor_counts.get(doctor, 0) + 1

        # Verify doctors were assigned
        assert_true(
            len(assigned_doctors) > 0,
            "test_workload_doctors_assigned",
            "Doctors were assigned to patients",
        )

        # Verify multiple doctors were used if available
        if len(hospital.doctors) > 1 and len(assigned_doctors) > 1:
            assert_true(
                len(doctor_counts) > 1,
                "test_workload_distribution",
                "Multiple doctors used for distribution",
            )

        # Check that all doctors have reasonable patient counts
        for doctor_name, count in doctor_counts.items():
            assert_true(
                count > 0,
                f"test_workload_{doctor_name}_positive",
                f"Doctor {doctor_name} has positive patient count",
            )
            assert_true(
                count <= len(patient_records),
                f"test_workload_{doctor_name}_reasonable",
                f"Doctor {doctor_name} has reasonable patient count",
            )

    except Exception as e:
        log_test_result(
            "test_staff_workload_distribution",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_ehr_system_data_persistence():
    """Test EHR system data persistence and retrieval."""
    logger.info("Testing EHR system data persistence...")

    try:
        # Create first EHR instance and add data
        ehr1 = EHRSystem(
            collection_name="test_persistence",
            persist_directory="./test_persistence_data",
        )

        patient = Patient(
            name="Persistence Test Patient",
            age=40,
            chief_complaint="Testing data persistence",
            symptoms=["test symptom"],
        )

        record_id = ehr1.add_patient_record(
            patient,
            "Test medical notes for persistence",
            "Dr. Persistence",
        )
        assert_not_none(
            record_id,
            "test_persistence_record_added",
            "Record added to first EHR instance",
        )

        # Query data from first instance
        history1 = ehr1.query_patient_history(patient.patient_id)
        assert_true(
            len(history1) > 0,
            "test_persistence_query_first",
            "Data retrieved from first instance",
        )

        # Create second EHR instance with same configuration
        ehr2 = EHRSystem(
            collection_name="test_persistence",
            persist_directory="./test_persistence_data",
        )

        # Query data from second instance
        history2 = ehr2.query_patient_history(patient.patient_id)

        if CHROMADB_AVAILABLE:
            # With ChromaDB, data should persist
            assert_true(
                len(history2) > 0,
                "test_persistence_chromadb",
                "Data persisted with ChromaDB",
            )
        else:
            # With memory storage, data won't persist between instances
            assert_equal(
                len(history2),
                0,
                "test_persistence_memory",
                "Memory storage doesn't persist between instances",
            )

    except Exception as e:
        log_test_result(
            "test_ehr_system_data_persistence",
            False,
            "",
            f"Exception: {str(e)}",
        )


# =============================================================================
# PERFORMANCE AND STRESS TESTS
# =============================================================================


def test_large_patient_queue_performance():
    """Test performance with large number of patients in queue."""
    logger.info("Testing large patient queue performance...")

    try:
        queue = PatientQueue()

        # Add many patients to queue
        start_time = time.time()
        num_patients = 100

        for i in range(num_patients):
            patient = Patient(
                name=f"Performance Patient {i}",
                age=20 + (i % 60),
                symptoms=(
                    ["test symptom"]
                    if i % 2 == 0
                    else ["emergency symptom", "chest pain"]
                ),
            )
            queue.add_patient(patient)

        add_time = time.time() - start_time

        # Test queue operations
        assert_equal(
            len(queue.waiting_patients),
            num_patients,
            "test_large_queue_size",
            "All patients added to queue",
        )

        # Test retrieving all patients
        start_time = time.time()
        retrieved_count = 0
        while True:
            patient = queue.get_next_patient()
            if patient is None:
                break
            retrieved_count += 1

        retrieve_time = time.time() - start_time

        assert_equal(
            retrieved_count,
            num_patients,
            "test_large_queue_retrieval",
            "All patients retrieved from queue",
        )

        # Performance should be reasonable (less than 1 second for 100 patients)
        assert_true(
            add_time < 5.0,
            "test_large_queue_add_performance",
            f"Add performance reasonable: {add_time:.2f}s",
        )
        assert_true(
            retrieve_time < 5.0,
            "test_large_queue_retrieve_performance",
            f"Retrieve performance reasonable: {retrieve_time:.2f}s",
        )

    except Exception as e:
        log_test_result(
            "test_large_patient_queue_performance",
            False,
            "",
            f"Exception: {str(e)}",
        )


def test_ehr_system_many_records():
    """Test EHR system with many records."""
    logger.info("Testing EHR system with many records...")

    try:
        ehr = EHRSystem(collection_name="test_many_records")

        # Add many patient records
        start_time = time.time()
        num_records = 50  # Reduced for reasonable test time
        patient_ids = []

        for i in range(num_records):
            patient = Patient(
                name=f"Many Records Patient {i}",
                age=25 + (i % 50),
                chief_complaint=f"Complaint {i}",
                symptoms=[f"symptom_{i}"],
            )

            record_id = ehr.add_patient_record(
                patient,
                f"Medical notes for patient {i}",
                f"Dr. {i % 5}",
            )
            patient_ids.append(patient.patient_id)

        add_time = time.time() - start_time

        # Test querying records
        start_time = time.time()
        retrieved_count = 0

        for patient_id in patient_ids[:10]:  # Test first 10
            history = ehr.query_patient_history(patient_id)
            if len(history) > 0:
                retrieved_count += 1

        query_time = time.time() - start_time

        assert_equal(
            retrieved_count,
            10,
            "test_many_records_retrieval",
            "Records can be retrieved",
        )
        assert_true(
            add_time < 30.0,
            "test_many_records_add_performance",
            f"Add performance reasonable: {add_time:.2f}s",
        )
        assert_true(
            query_time < 10.0,
            "test_many_records_query_performance",
            f"Query performance reasonable: {query_time:.2f}s",
        )

    except Exception as e:
        log_test_result(
            "test_ehr_system_many_records",
            False,
            "",
            f"Exception: {str(e)}",
        )


# =============================================================================
# TEST RUNNER AND REPORTING
# =============================================================================


def run_all_tests():
    """Run all test functions and generate comprehensive report."""
    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE HOSPITAL SIMULATION TESTS")
    logger.info("=" * 80)

    # Test categories and their functions
    test_categories = {
        "Patient Class Tests": [
            test_patient_creation_basic,
            test_patient_creation_full,
            test_patient_priority_calculation,
            test_patient_serialization,
            test_patient_agent_creation,
        ],
        "EHR System Tests": [
            test_ehr_system_initialization,
            test_ehr_add_patient_record,
            test_ehr_query_patient_history,
            test_ehr_search_similar_cases,
        ],
        "Hospital Staff Tests": [
            test_hospital_staff_creation,
            test_hospital_staff_patient_assignment,
            test_hospital_staff_metrics_update,
        ],
        "Patient Queue Tests": [
            test_patient_queue_creation,
            test_patient_queue_add_patient,
            test_patient_queue_priority_order,
            test_patient_queue_status,
        ],
        "Hospital Simulation Tests": [
            test_hospital_simulation_creation,
            test_hospital_staff_initialization,
            test_hospital_add_patient,
            test_hospital_run_method,
            test_hospital_generate_patients,
            test_hospital_statistics_update,
        ],
        "Edge Cases and Error Handling": [
            test_patient_invalid_data,
            test_ehr_system_edge_cases,
            test_hospital_staff_edge_cases,
            test_patient_queue_edge_cases,
            test_hospital_simulation_error_handling,
        ],
        "Integration Tests": [
            test_end_to_end_patient_flow,
            test_multiple_patients_concurrency,
            test_staff_workload_distribution,
            test_ehr_system_data_persistence,
        ],
        "Performance Tests": [
            test_large_patient_queue_performance,
            test_ehr_system_many_records,
        ],
    }

    # Run all test categories
    for category, test_functions in test_categories.items():
        logger.info(f"\n{'='*20} {category} {'='*20}")
        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                log_test_result(
                    test_func.__name__,
                    False,
                    "",
                    f"Unexpected exception: {str(e)}",
                )

    # Generate final report
    generate_test_report()


def generate_test_report():
    """Generate comprehensive test report."""
    logger.info("\n" + "=" * 80)
    logger.info("HOSPITAL SIMULATION TEST RESULTS SUMMARY")
    logger.info("=" * 80)

    total_tests = test_results["passed"] + test_results["failed"]
    pass_rate = (
        (test_results["passed"] / total_tests * 100)
        if total_tests > 0
        else 0
    )

    logger.info(f"Total Tests Run: {total_tests}")
    logger.info(f"Tests Passed: {test_results['passed']}")
    logger.info(f"Tests Failed: {test_results['failed']}")
    logger.info(f"Pass Rate: {pass_rate:.1f}%")

    if test_results["failed"] > 0:
        logger.error("\nFAILED TESTS:")
        for error in test_results["errors"]:
            logger.error(f"  - {error}")

    # Component-wise summary
    component_stats = {}
    for test_detail in test_results["test_details"]:
        test_name = test_detail["test_name"]
        component = "Unknown"

        if "patient" in test_name.lower():
            component = "Patient System"
        elif "ehr" in test_name.lower():
            component = "EHR System"
        elif "staff" in test_name.lower():
            component = "Staff Management"
        elif "queue" in test_name.lower():
            component = "Patient Queue"
        elif "hospital" in test_name.lower():
            component = "Hospital Simulation"
        elif (
            "e2e" in test_name.lower()
            or "multi" in test_name.lower()
            or "workload" in test_name.lower()
        ):
            component = "Integration"
        elif (
            "large" in test_name.lower()
            or "many" in test_name.lower()
            or "performance" in test_name.lower()
        ):
            component = "Performance"

        if component not in component_stats:
            component_stats[component] = {"passed": 0, "failed": 0}

        if test_detail["passed"]:
            component_stats[component]["passed"] += 1
        else:
            component_stats[component]["failed"] += 1

    logger.info("\nCOMPONENT-WISE RESULTS:")
    for component, stats in component_stats.items():
        total = stats["passed"] + stats["failed"]
        rate = (stats["passed"] / total * 100) if total > 0 else 0
        logger.info(
            f"  {component}: {stats['passed']}/{total} passed ({rate:.1f}%)"
        )

    # Save detailed results to JSON
    with open("tests/detailed_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info(
        "\nDetailed test results saved to: tests/detailed_test_results.json"
    )
    logger.info("Test logs saved to: tests/test_results.log")

    if pass_rate >= 90:
        logger.success(f"\n🎉 EXCELLENT! Pass rate: {pass_rate:.1f}%")
    elif pass_rate >= 80:
        logger.info(f"\n✅ GOOD! Pass rate: {pass_rate:.1f}%")
    elif pass_rate >= 70:
        logger.warning(
            f"\n⚠️  NEEDS IMPROVEMENT! Pass rate: {pass_rate:.1f}%"
        )
    else:
        logger.error(f"\n❌ POOR! Pass rate: {pass_rate:.1f}%")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Run all tests when executed directly."""
    print("🏥 Starting Hospital Simulation Comprehensive Unit Tests")
    print("=" * 60)

    # Create tests directory if it doesn't exist
    os.makedirs("tests", exist_ok=True)

    # Run all tests
    run_all_tests()

    print("\n" + "=" * 60)
    print("🏁 Hospital Simulation Unit Tests Complete!")
    print("Check tests/test_results.log for detailed logs")
    print(
        "Check tests/detailed_test_results.json for detailed results"
    )
