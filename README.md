# Baymax Humanoid Robot Simulation

Hi Professor Li,

I am currently working on a humanoid robot simulation using MuJoCo. The main development is now focused on implementing preview control for walking.

## Current Focus
[`PreviewControlVisualizer.py`](./Baymax/PreviewControlVisualizer.py) - A new implementation that uses preview control for walking:
- Preview control algorithm for walking pattern generation
- ZMP and CoM trajectory visualization
- Basic inverse kinematics for leg control
- Data collection and plotting
- Step pattern generation and tracking
- Error integrator visualization

## Previous Development Files
1. [`Baymax.xml`](./Baymax/Baymax.xml) - The robot model definition file which includes:
   - Robot structure and joint configurations
   - Physical properties (mass, friction, etc.)
   - Actuator specifications
   - Visual properties and sensor configurations
   - Joint stiffness and damping parameters

2. [`BaymaxPIDControl.py`](./Baymax/BaymaxPIDControl.py) - The initial control implementation which includes:
   - Basic P (Proportional) control system for joint angles
   - Simple ZMP calculation
   - Basic COM tracking
   - Data collection and plotting functionality
   - Simulation visualization

## Current Features
- Full humanoid leg model with torso
- Preview control for walking pattern generation
- ZMP and CoM tracking
- Performance analysis and visualization
- Data logging and plotting
- Step pattern generation
- Error integrator tracking

## In Development
- Fine-tuning of preview control parameters
- Improving walking pattern generation
- Testing different walking patterns
- Working on walking stability

The simulation has progressed from basic P control to preview control for walking pattern generation. The current focus is on implementing preview control for bipedal walking, with ongoing work to develop stable walking patterns. 