# 🏥 Hospital Simulation System

A comprehensive multi-agent simulation of a small hospital with intelligent patient management, medical staff coordination, and an Electronic Health Record (EHR) system powered by ChromaDB for advanced RAG capabilities.

## 🌟 Features

### 🏗️ **Multi-Agent Architecture**
- **Executive Team**: CEO, CFO, CMO for strategic planning and hospital management
- **Medical Staff**: Specialized doctors and nurses with role-specific expertise
- **Support Staff**: Receptionists for patient check-in and queue management
- **Intelligent Coordination**: Agents communicate and collaborate automatically

### 👥 **Patient Management**
- **Priority-Based Queue**: Intelligent triage system with emergency prioritization
- **Comprehensive Patient Model**: Name, age, symptoms, medical history, allergies, vital signs
- **Dynamic Priority Scoring**: Automatic calculation based on symptoms and vital signs
- **Stateful Tracking**: Complete patient journey from arrival to discharge

### 🧠 **Advanced RAG System**
- **ChromaDB Integration**: Vector database for medical record storage and retrieval
- **Semantic Search**: Find similar medical cases and patient history
- **Medical Knowledge Base**: Persistent storage of all patient interactions
- **Fallback Storage**: In-memory storage when ChromaDB unavailable

### 📊 **Hospital Operations**
- **Automated Patient Flow**: Reception → Triage → Doctor Consultation → EHR
- **Performance Metrics**: Track wait times, patient satisfaction, revenue, costs
- **Executive Meetings**: Regular strategic planning sessions
- **Real-time Monitoring**: Live status updates and statistics

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd hospital-simulation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the simulation**:
```bash
python hospital_sim.py
```

### Basic Usage

```python
from hospital_sim import HospitalSimulation, Patient

# Create hospital
hospital = HospitalSimulation("General Hospital")

# Add patients
patient = Patient(
    name="John Doe",
    age=45,
    chief_complaint="Chest pain",
    symptoms=["chest pain", "shortness of breath"]
)
hospital.add_patient(patient)

# Run simulation
hospital.run_simulation(duration_minutes=60)
```

## 🏥 System Architecture

### Patient Flow
```
Patient Arrival → Reception Check-in → Queue Management → 
Triage (Nurse) → Doctor Consultation → EHR Documentation → Discharge
```

### Staff Hierarchy
```
Executive Team (CEO, CFO, CMO)
    ↓
Medical Staff (Doctors, Nurses)
    ↓
Support Staff (Receptionists)
```

### Data Flow
```
Patient Data → EHR System (ChromaDB) → RAG Queries → 
Medical History → Diagnosis Support → Treatment Planning
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set your OpenAI API key for enhanced LLM capabilities
export OPENAI_API_KEY="your-api-key-here"

# Optional: ChromaDB configuration
export CHROMADB_HOST="localhost"
export CHROMADB_PORT="8000"
```

### Simulation Parameters
```python
hospital.run_simulation(
    duration_minutes=60,        # Simulation duration
    patient_arrival_rate=0.2    # Patients per minute
)
```

## 📋 Patient Model

### Core Fields
- **Basic Info**: Name, age, gender, patient ID
- **Medical**: Chief complaint, symptoms, medical history
- **Medications**: Current medications, allergies
- **Vital Signs**: Blood pressure, heart rate, temperature, oxygen
- **Status**: Waiting, in triage, with doctor, treated, discharged
- **Priority Score**: 0-20 scale for queue management

### Priority Calculation
```python
# Emergency symptoms: +10 points
# Pain symptoms: +5 points  
# Fever symptoms: +3 points
# Abnormal vitals: +6-8 points
# Maximum priority: 20 points
```

## 🧪 EHR System

### ChromaDB Features
- **Document Storage**: Comprehensive medical records
- **Semantic Search**: Find relevant medical information
- **Metadata Filtering**: Search by patient, doctor, date
- **Similar Case Search**: Find comparable medical cases

### Fallback Storage
- **In-Memory**: When ChromaDB unavailable
- **JSON Format**: Structured data storage
- **Search Capabilities**: Basic text matching

## 🤖 Agent Specializations

### Executive Agents
- **CEO**: Strategic planning, growth, quality oversight
- **CFO**: Financial management, cost optimization, revenue
- **CMO**: Medical quality, protocols, safety standards

### Medical Agents
- **Emergency Doctor**: Rapid assessment, critical care, stabilization
- **General Doctor**: Comprehensive evaluation, diagnosis, treatment
- **Triage Nurse**: Initial assessment, vital signs, priority assignment
- **Floor Nurse**: Patient care, medication, monitoring

### Support Agents
- **Receptionist**: Check-in, scheduling, patient flow management

## 📊 Performance Metrics

### Hospital Metrics
- **Patient Volume**: Total patients, treated patients
- **Efficiency**: Average wait time, treatment time
- **Financial**: Revenue, costs, net profit
- **Quality**: Patient satisfaction scores

### Staff Metrics
- **Productivity**: Patients treated, average treatment time
- **Quality**: Patient satisfaction, error rates
- **Efficiency**: Time management, resource utilization

## 🔄 Executive Meetings

### Regular Sessions
- **Frequency**: Every 15 minutes during simulation
- **Participants**: CEO, CFO, CMO
- **Topics**: Performance review, strategic planning, optimization

### Decision Implementation
- **Marketing**: Patient volume strategies
- **Cost Control**: Operational efficiency measures
- **Quality**: Care standards and protocols

## 🚨 Emergency Handling

### Priority System
- **Emergency (15-20)**: Immediate attention (chest pain, trauma)
- **Urgent (10-14)**: High priority (severe pain, high fever)
- **Standard (5-9)**: Normal priority (routine complaints)
- **Low (0-4)**: Non-urgent (minor symptoms)

### Triage Process
1. **Vital Signs**: Blood pressure, heart rate, temperature
2. **Symptom Assessment**: Chief complaint and associated symptoms
3. **Priority Assignment**: Calculate priority score
4. **Queue Placement**: Add to priority queue

## 🧪 Testing and Development

### Running Tests
```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Code formatting
black hospital_sim.py

# Linting
flake8 hospital_sim.py
```

## 🔮 Future Enhancements

### Planned Features
- **Specialist Agents**: Cardiologists, neurologists, surgeons
- **Advanced Diagnostics**: Lab results, imaging, pathology
- **Treatment Protocols**: Evidence-based medicine guidelines
- **Insurance Integration**: Billing and claims processing
- **Telemedicine**: Remote consultation capabilities
- **AI Diagnostics**: Machine learning for diagnosis support

### Scalability Improvements
- **Distributed Processing**: Multi-hospital simulations
- **Real-time Analytics**: Live dashboards and reporting
- **Integration APIs**: Connect with external medical systems
- **Mobile Support**: Patient and staff mobile applications

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- **Medical Protocols**: Evidence-based treatment guidelines
- **Agent Specializations**: New medical specialties
- **Data Models**: Enhanced patient and medical record structures
- **Performance Optimization**: Simulation speed and efficiency
- **Testing**: Comprehensive test coverage
- **Documentation**: User guides and API documentation

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Swarms Framework**: Multi-agent orchestration platform
- **ChromaDB**: Vector database for RAG systems
- **Medical Community**: For domain expertise and validation
- **Open Source Community**: For tools and libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**🏥 Built with ❤️ for the future of healthcare simulation**
