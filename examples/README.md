# Hospital Simulation Examples

This directory contains example scripts demonstrating different aspects of the Hospital Simulation system. Each example is designed to showcase specific features and use cases.

## Prerequisites

Before running any examples, ensure you have:

1. **Python 3.7+** installed
2. **Required dependencies** installed:
   ```bash
   pip install -r ../requirements.txt
   ```
3. **Environment variables** set up (if using `.env` file)

## Example Files

### 1. Basic Simulation ([`basic_simulation.py`](./basic_simulation.py))

**Purpose**: Demonstrates the most basic hospital simulation setup.

**Features**:
- Creates a hospital instance
- Generates 3 random patients
- Runs a 10-minute simulation with low patient arrival rate

**Usage**:
```bash
python examples/basic_simulation.py
```

**What it demonstrates**:
- Basic hospital initialization
- Patient generation
- Simulation execution
- Minimal configuration

---

### 2. Custom Patients ([`custom_patients.py`](./custom_patients.py))

**Purpose**: Shows how to create and add custom patients with specific medical profiles.

**Features**:
- Creates 3 custom patients with detailed medical information
- Demonstrates patient attributes (symptoms, medical history, medications, allergies)
- Runs a 15-minute simulation

**Custom Patients**:
- **Aurora Dreamweaver**: 35-year-old female with severe migraine
- **Jasper Stormcloud**: 52-year-old male with chest tightness
- **Nova Celestine**: 28-year-old female with fever and sore throat

**Usage**:
```bash
python examples/custom_patients.py
```

**What it demonstrates**:
- Custom patient creation
- Detailed medical profiles
- Patient data structure
- Medical history and medication tracking

---

### 3. Emergency Scenario ([`emergency_scenario.py`](./emergency_scenario.py))

**Purpose**: Focuses on emergency patient handling and critical care scenarios.

**Features**:
- Creates 3 emergency patients with urgent conditions
- Higher patient arrival rate (0.2 per minute)
- 20-minute simulation duration
- Emergency-specific medical profiles

**Emergency Patients**:
- **Phoenix Razorcrest**: 45-year-old male with chest pain (potential heart attack)
- **Sage Whisperwind**: 67-year-old female with sudden severe headache (potential stroke)
- **Titan Shadowmere**: 38-year-old male with severe abdominal pain (potential appendicitis)

**Usage**:
```bash
python examples/emergency_scenario.py
```

**What it demonstrates**:
- Emergency patient handling
- Critical care scenarios
- High-priority patient processing
- Time-sensitive medical conditions

---

### 4. EHR Demo ([`ehr_demo.py`](./ehr_demo.py))

**Purpose**: Demonstrates Electronic Health Record (EHR) system capabilities.

**Features**:
- Creates a complex patient with extensive medical history
- Demonstrates EHR search functionality
- Shows similar case matching
- Generates comprehensive medical data

**Complex Patient**:
- **Nebula Starforge**: 58-year-old female with chronic back pain
- Multiple medical conditions and medications
- Complex allergy profile
- Demonstrates EHR data integration

**Usage**:
```bash
python examples/ehr_demo.py
```

**What it demonstrates**:
- EHR system integration
- Medical data search and matching
- Complex patient profiles
- Healthcare data analytics

---

### 5. Interactive Simulation ([`interactive_simulation.py`](./interactive_simulation.py))

**Purpose**: Provides an interactive experience where users can customize simulation parameters.

**Features**:
- User input for hospital name
- Customizable simulation duration
- Adjustable patient arrival rate
- Interactive parameter setting

**Usage**:
```bash
python examples/interactive_simulation.py
```

**Interactive Prompts**:
- Hospital name (default: "Interactive Hospital")
- Simulation duration in minutes (default: 20)
- Patient arrival rate per minute (default: 0.15)

**What it demonstrates**:
- User interaction
- Parameter customization
- Dynamic simulation configuration
- Interactive healthcare simulation

---

## Running Examples

### From Project Root
```bash
# Run any example from the project root
python examples/basic_simulation.py
python examples/custom_patients.py
python examples/emergency_scenario.py
python examples/ehr_demo.py
python examples/interactive_simulation.py
```

### From Examples Directory
```bash
# Navigate to examples directory first
cd examples
python basic_simulation.py
python custom_patients.py
python emergency_scenario.py
python ehr_demo.py
python interactive_simulation.py
```

## Expected Outputs

Each example will:
1. **Initialize** the hospital simulation system
2. **Create/Generate** patients as specified
3. **Run** the simulation for the configured duration
4. **Process** patients through the hospital workflow
5. **Generate** simulation results and statistics

## Customization

You can modify any example to:
- Change patient data and medical profiles
- Adjust simulation parameters (duration, arrival rates)
- Add additional patients or scenarios
- Modify hospital configurations
- Test different medical conditions and treatments

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root or have the hospital_sim module in your Python path
2. **Missing Dependencies**: Install required packages with `pip install -r requirements.txt`
3. **Environment Variables**: Check that your `.env` file is properly configured if using environment-specific settings

### Getting Help

- Check the main project README for detailed setup instructions
- Review the hospital_sim module documentation
- Examine the source code in the main hospital_sim directory

## Next Steps

After running these examples, you can:
- Create your own custom scenarios
- Integrate the simulation into larger healthcare applications
- Extend the patient data models
- Add new medical conditions and treatments
- Build custom reporting and analytics

---

*For more information about the Hospital Simulation system, see the main project documentation.*
