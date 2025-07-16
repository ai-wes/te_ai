To run the refactored living therapeutics system that now properly uses the core TE-AI system, you have several options:

Option 1: Run via Command Line (Recommended)

# Navigate to the scripts directory

cd C:\Users\wes\Desktop\te_ai\scripts

# Run with different modes:

# Basic therapeutic simulation

python -m domains.living_therapeutics_system.living_therapeutics_system_run --mode basic

# Production mode with full monitoring

python -m domains.living_therapeutics_system.living_therapeutics_system_run --mode production --hours 48 --save-results

# Enhanced mode with stem cell features

python -m domains.living_therapeutics_system.living_therapeutics_system_run --mode enhanced --patient-type cancer --severity 0.9

# With live visualization (requires visualization server running)

python -m domains.living_therapeutics_system.living_therapeutics_system_run --mode visualization

Option 2: Run with Core System Visualization

Since it now uses the core TE-AI system, you can also run it with the main visualization:

# This will run the core system with visualization

python run_with_visualization.py

# Then in another terminal, run the therapeutic system

python -m domains.living_therapeutics_system.living_therapeutics_system_run --mode visualization

Option 3: Direct Python Script

Create a simple run script or use Python directly:

from domains.living_therapeutics_system import LivingTherapeuticSystem, create_patient_profile

# Create patient

patient = create_patient_profile('autoimmune', severity=0.7)

# Initialize system (this now uses full core TE-AI)

system = LivingTherapeuticSystem(patient)

# Run treatment cycles

for i in range(10):
result = system.run_treatment_cycle()
print(f"Cycle {i}: Efficacy={result['response']['efficacy_score']:.3f}")

Command Line Options:

- --mode: Choose simulation mode (basic, production, enhanced, visualization, validate)
- --patient-type: Disease type (autoimmune, cancer, metabolic)
- --severity: Disease severity (0.0-1.0)
- --hours: Treatment duration in hours
- --save-results: Save results to JSON file
- --report-interval: Hours between progress reports (for production mode)

What to Expect:

Now that it uses the core system, you'll see:

1. Full TE-AI Evolution Messages:


    - Transposition events
    - Dream consolidation cycles
    - Phase transition detection
    - Quantum gene emergence

2. Therapeutic-Specific Output:


    - Patient assessment
    - Treatment efficacy scores
    - Disease severity tracking
    - Therapeutic gene differentiation

3. Core System Features:


    - GPU batch processing
    - Population fitness evolution
    - Stress-induced transposition
    - Memory consolidation
