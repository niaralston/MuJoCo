# Baymax Humanoid Robot Simulation

Hi Professor Li,

I am currently working on a humanoid robot simulation using MuJoCo. The main development is focused on two key files:

## Main Files
1. [`Baymax.xml`](./Baymax/Baymax.xml) - The robot model definition file which includes:
   - Robot structure and joint configurations
   - Physical properties (mass, friction, etc.)
   - Actuator specifications
   - Visual properties and sensor configurations
   - Joint stiffness and damping parameters

2. [`BaymaxPIDControl.py`](./Baymax/BaymaxPIDControl.py) - The control implementation which includes:
   - Currently implemented P control system for joint angles
   - In progress: Implementation of I (Integral) and D (Derivative) control components
   - Zero Moment Point (ZMP) calculation and visualization
   - Center of Mass (COM) tracking
   - The Start of stability analysis 
   - Data collection and plotting functionality
   - Simulation visualization and recording capabilities

## Current Features
- Full humanoid leg model with torso
- Proportional (P) control with configurable gains
- Real-time stability monitoring
- Performance analysis and visualization
- Data logging and plotting

## In Development
- Integral (I) control for reducing steady-state error
- Derivative (D) control for improved dynamic response
- Fine-tuning of control parameters for optimal performance

The simulation currently focuses on achieving stable control of the humanoid robot's legs using proportional control, with ongoing work to implement and tune the integral and derivative components for enhanced performance. 