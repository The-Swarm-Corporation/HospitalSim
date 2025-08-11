#!/usr/bin/env python3
"""
Test Hospital Simulation System

Simple tests to verify the hospital simulation works correctly.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to the path so we can import hospital_sim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hospital_sim import (
    Patient,
    PatientStatus,
    StaffRole,
    HospitalStaff,
    PatientQueue,
    EHRSystem,
    HospitalSimulation,
)


class TestPatient(unittest.TestCase):
    """Test the Patient class."""

    def test_patient_creation(self):
        """Test basic patient creation."""
        patient = Patient(
            name="Test Patient",
            age=30,
            gender="Female",
            chief_complaint="Headache",
            symptoms=["headache", "nausea"],
        )

        self.assertEqual(patient.name, "Test Patient")
        self.assertEqual(patient.age, 30)
        self.assertEqual(patient.gender, "Female")
        self.assertEqual(patient.chief_complaint, "Headache")
        self.assertEqual(patient.symptoms, ["headache", "nausea"])
        self.assertEqual(patient.status, PatientStatus.WAITING)
        self.assertIsNotNone(patient.patient_id)

    def test_priority_calculation(self):
        """Test patient priority calculation."""
        # Emergency symptoms should get high priority
        emergency_patient = Patient(
            name="Emergency Patient",
            symptoms=["chest pain", "shortness of breath"],
        )
        self.assertGreater(emergency_patient.priority_score, 10)

        # Regular symptoms should get lower priority
        regular_patient = Patient(
            name="Regular Patient", symptoms=["headache", "fever"]
        )
        self.assertLess(regular_patient.priority_score, 15)

    def test_patient_serialization(self):
        """Test patient to_dict and from_dict methods."""
        original_patient = Patient(
            name="Serial Patient",
            age=25,
            gender="Male",
            symptoms=["cough"],
        )

        # Convert to dict
        patient_dict = original_patient.to_dict()

        # Convert back to patient
        restored_patient = Patient.from_dict(patient_dict)

        self.assertEqual(original_patient.name, restored_patient.name)
        self.assertEqual(original_patient.age, restored_patient.age)
        self.assertEqual(
            original_patient.gender, restored_patient.gender
        )


class TestPatientQueue(unittest.TestCase):
    """Test the PatientQueue class."""

    def test_queue_operations(self):
        """Test basic queue operations."""
        queue = PatientQueue()

        # Create test patients
        patient1 = Patient(name="Patient 1", symptoms=["headache"])
        patient2 = Patient(name="Patient 2", symptoms=["chest pain"])

        # Add patients
        queue.add_patient(patient1)
        queue.add_patient(patient2)

        # Check queue status
        status = queue.get_queue_status()
        self.assertEqual(status["total_waiting"], 2)

        # Get next patient (should be higher priority)
        next_patient = queue.get_next_patient()
        self.assertIsNotNone(next_patient)

        # Check updated status
        status = queue.get_queue_status()
        self.assertEqual(status["total_waiting"], 1)


class TestEHRSystem(unittest.TestCase):
    """Test the EHR system."""

    def test_ehr_initialization(self):
        """Test EHR system initialization."""
        ehr = EHRSystem()
        self.assertIsNotNone(ehr)

    def test_patient_record_creation(self):
        """Test adding patient records."""
        ehr = EHRSystem()
        patient = Patient(name="Test Patient", age=40)

        record_id = ehr.add_patient_record(
            patient, "Patient has headache", "Dr. Test"
        )

        self.assertIsNotNone(record_id)

    def test_patient_history_query(self):
        """Test querying patient history."""
        ehr = EHRSystem()
        patient = Patient(name="Query Patient", age=35)

        # Add a record
        ehr.add_patient_record(patient, "Test notes", "Dr. Test")

        # Query history
        history = ehr.query_patient_history(patient.patient_id)
        self.assertGreater(len(history), 0)


class TestHospitalStaff(unittest.TestCase):
    """Test the HospitalStaff class."""

    def test_staff_creation(self):
        """Test staff member creation."""
        mock_agent = Mock()
        staff = HospitalStaff(
            "Dr. Test", StaffRole.DOCTOR, mock_agent
        )

        self.assertEqual(staff.name, "Dr. Test")
        self.assertEqual(staff.role, StaffRole.DOCTOR)
        self.assertTrue(staff.is_available)
        self.assertIsNone(staff.current_patient)

    def test_patient_assignment(self):
        """Test patient assignment to staff."""
        mock_agent = Mock()
        staff = HospitalStaff(
            "Nurse Test", StaffRole.NURSE, mock_agent
        )
        patient = Patient(name="Test Patient")

        # Assign patient
        success = staff.assign_patient(patient)
        self.assertTrue(success)
        self.assertEqual(staff.current_patient, patient)
        self.assertFalse(staff.is_available)

        # Try to assign another patient (should fail)
        patient2 = Patient(name="Patient 2")
        success2 = staff.assign_patient(patient2)
        self.assertFalse(success2)

    def test_patient_release(self):
        """Test releasing patients from staff."""
        mock_agent = Mock()
        staff = HospitalStaff(
            "Doctor Test", StaffRole.DOCTOR, mock_agent
        )
        patient = Patient(name="Test Patient")

        # Assign and then release
        staff.assign_patient(patient)
        staff.release_patient()

        self.assertIsNone(staff.current_patient)
        self.assertTrue(staff.is_available)
        self.assertIn(patient, staff.patients_seen)


class TestHospitalSimulation(unittest.TestCase):
    """Test the HospitalSimulation class."""

    @patch("hospital_sim.Agent")
    def test_simulation_creation(self, mock_agent):
        """Test hospital simulation creation."""
        # Mock the Agent class to avoid LLM calls
        mock_agent.return_value = Mock()

        hospital = HospitalSimulation("Test Hospital")

        self.assertEqual(hospital.hospital_name, "Test Hospital")
        self.assertIsNotNone(hospital.ehr_system)
        self.assertIsNotNone(hospital.patient_queue)
        self.assertGreater(len(hospital.staff), 0)

    @patch("hospital_sim.Agent")
    def test_patient_generation(self, mock_agent):
        """Test patient generation."""
        mock_agent.return_value = Mock()
        hospital = HospitalSimulation("Test Hospital")

        # Generate patients
        hospital.generate_patients(3)

        # Check that patients were added
        self.assertEqual(
            hospital.simulation_stats["total_patients"], 3
        )


def run_tests():
    """Run all tests."""
    print("🏥 Running Hospital Simulation Tests")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestPatient,
        TestPatientQueue,
        TestEHRSystem,
        TestHospitalStaff,
        TestHospitalSimulation,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(
            test_class
        )
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} tests failed")
        print(f"❌ {len(result.errors)} tests had errors")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
