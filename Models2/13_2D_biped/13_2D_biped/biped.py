#Built off of example code from https://pab47.github.io/mujoco.html
# MuJoCo 2D Biped Walking Simulation
# This code implements a walking controller for a 2D biped robot using a state machine approach
# The robot uses telescoping (sliding) legs instead of bending knees for simplified control

# --- User Settings ---
RECORD_VIDEO = False  # Set to True to record the simulation, False to just view it

# --- Required Libraries ---
import mujoco as mj                # Main MuJoCo physics engine for robot simulation
from numpy.linalg import inv       # Matrix inverse operations for calculations
from scipy.spatial.transform import Rotation as R  # Utilities for handling 3D rotations
import numpy as np                 # Numerical operations and array handling
import os                         # File and path operations
import mujoco.viewer             # MuJoCo visualization tools
import time                      # Time control and delays
if RECORD_VIDEO:
    import cv2                   # For video recording (only import if needed)
import platform                  # For OS detection
import subprocess                # For running system commands

# Clear terminal
if platform.system() == 'Windows':
    subprocess.run('cls', shell=True)
else:
    subprocess.run('clear', shell=True)

# --- Simulation Parameters ---
xml_path = 'biped.xml'  # XML file containing robot's physical description and properties
simend = 120            # Total simulation duration in seconds

# Counter to track simulation steps
step_no = 0

# --- Finite State Machine (FSM) Definitions ---
# The walking controller uses a state machine to coordinate leg movements

# States for leg (hip) control
FSM_LEG1_SWING = 0    # State when leg 1 is in swing phase (moving through air)
FSM_LEG2_SWING = 1    # State when leg 2 is in swing phase

# States for leg 1 telescoping control during walking cycle
FSM_KNEE1_STANCE = 0  # Leg 1 is fully extended, supporting body weight
FSM_KNEE1_RETRACT = 1 # Leg 1 is retracted (shortened by sliding up)

# States for leg 2 telescoping control
FSM_KNEE2_STANCE = 0  # Leg 2 is fully extended, supporting body weight
FSM_KNEE2_RETRACT = 1 # Leg 2 is retracted (shortened by sliding up)

# Initialize FSM states for walking
fsm_hip = FSM_LEG2_SWING      # Begin with leg 2 in swing phase
fsm_knee1 = FSM_KNEE1_STANCE  # Begin with leg 1 fully extended
fsm_knee2 = FSM_KNEE2_STANCE  # Begin with leg 2 fully extended


def controller(model, data):
    """
    Main walking controller function - Called every simulation step
    
    This function implements the walking control logic using a state machine approach.
    It monitors leg positions and angles to coordinate the walking gait, switching
    between swing and stance phases for each leg while controlling leg extension/retraction.
    The legs use sliding joints to extend and retract (like telescopes) rather than bending knees.
    
    Coordinate System (Right-Hand Rule):
    - X-axis: Points forward (walking direction, to the right in visualization)
    - Y-axis: Points into the screen (rotation axis for leg pitch)
    - Z-axis: Points up

    Angle Conventions:
    - Leg angles follow the right-hand rule around Y-axis:
      * When thumb points into screen (+Y)
      * Fingers curl from +Z (up) toward -X (backward)
      * Therefore, positive rotation = leg leans backward
      * Negative rotation = leg leans forward
    - MuJoCo returns pitch following this convention, but we negate it for intuitive control:
      * abs_leg = -euler_leg[1] means:
        - Positive abs_leg = leg leans backward (ready to push)
        - Negative abs_leg = leg leans forward (ready to catch)
    - Relative angle between legs:
      * Positive when leg 1 is in front of leg 2
      * Negative when leg 1 is behind leg 2

    Args:
        model: MuJoCo model containing robot's physical properties
        data: Current simulation state data
    """
    global fsm_hip, fsm_knee1, fsm_knee2, step_no

    # --- State Estimation - Get current leg positions and orientations ---
    
    # Get leg 1 state
    quat_leg1 = data.xquat[1, :]  # Quaternion orientation of leg 1
    euler_leg1 = quat2euler(quat_leg1)  # Convert to euler angles for easier use
    abs_leg1 = -euler_leg1[1]  # Negate pitch for intuitive control (positive = lean back, negative = lean forward)
    pos_foot1 = data.xpos[2, :]  # 3D position of foot 1

    # Get leg 2 state
    quat_leg2 = data.xquat[3, :]  # Quaternion orientation of leg 2
    euler_leg2 = quat2euler(quat_leg2)  # Convert to euler angles
    abs_leg2 = -euler_leg2[1]  # Negate pitch for intuitive control (positive = lean back, negative = lean forward)
    pos_foot2 = data.xpos[4, :]  # 3D position of foot 2

    # --- State Transitions - Determine when to switch between walking phases ---
    
    # Switch from leg 2 swing to leg 1 swing when:
    # 1. Leg 2 is currently swinging
    # 2. Foot 2 is close to ground (height < 0.05)
    # 3. Leg 1 is behind (angle < 0) to prepare for ground contact
    if fsm_hip == FSM_LEG2_SWING and pos_foot2[2] < 0.05 and abs_leg1 < 0.0:
        fsm_hip = FSM_LEG1_SWING  # Begin swinging leg 1

    # Similar transition for switching from leg 1 swing to leg 2 swing
    # Switch when leg 1 is down and leg 2 is leaning forward to catch
    if fsm_hip == FSM_LEG1_SWING and pos_foot1[2] < 0.05 and abs_leg2 < 0.0:
        fsm_hip = FSM_LEG2_SWING  # Begin swinging leg 2

    # --- Leg Length Control State Transitions ---
    
    # Control leg 1 extension/retraction timing
    # Retract when other leg (leg 2) is down and this leg 1 is leaning backwards
    if fsm_knee1 == FSM_KNEE1_STANCE and pos_foot2[2] < 0.05 and abs_leg1 < 0.0:
        fsm_knee1 = FSM_KNEE1_RETRACT  # Start retracting leg 1 (sliding up)
    # Extend when leg has swung back enough (positive angle)
    if fsm_knee1 == FSM_KNEE1_RETRACT and abs_leg1 > 0.1:
        fsm_knee1 = FSM_KNEE1_STANCE   # Extend leg 1 fully

    # Control leg 2 extension/retraction timing
    # Similar logic as leg 1 but for leg 2
    if fsm_knee2 == FSM_KNEE2_STANCE and pos_foot1[2] < 0.05 and abs_leg2 < 0.0:
        fsm_knee2 = FSM_KNEE2_RETRACT  # Start retracting leg 2 (sliding up)
    if fsm_knee2 == FSM_KNEE2_RETRACT and abs_leg2 > 0.1:
        fsm_knee2 = FSM_KNEE2_STANCE   # Extend leg 2 fully

    # --- Apply Control Actions ---
    
    # Control hip joint for leg swinging
    if fsm_hip == FSM_LEG1_SWING:
        data.ctrl[0] = -0.5  # Apply torque to swing leg 1 forward
    if fsm_hip == FSM_LEG2_SWING:
        data.ctrl[0] = 0.5   # Apply torque to swing leg 2 forward

    # Control leg extension (sliding joint positions)
    if fsm_knee1 == FSM_KNEE1_STANCE:
        data.ctrl[2] = 0.0   # Keep leg 1 fully extended
    if fsm_knee1 == FSM_KNEE1_RETRACT:
        data.ctrl[2] = -0.25 # Retract leg 1 by sliding up

    if fsm_knee2 == FSM_KNEE2_STANCE:
        data.ctrl[4] = 0.0   # Keep leg 2 fully extended
    if fsm_knee2 == FSM_KNEE2_RETRACT:
        data.ctrl[4] = -0.25 # Retract leg 2 by sliding up

def init_controller(model, data):
    """
    Initialize the controller by setting starting positions
    
    Args:
        model: MuJoCo model
        data: Simulation state data
    """
    data.qpos[4] = 0.5  # Set initial joint position
    data.ctrl[0] = data.qpos[4]  # Set initial control signal to 0.5

def quat2euler(quat):
    """
    Convert quaternion to euler angles
    
    MuJoCo uses quaternions for rotation but euler angles are more intuitive for control
    
    Args:
        quat: Quaternion array [w, x, y, z]
    
    Returns:
        euler: Euler angles [roll, pitch, yaw] in radians
    """
    # Convert between different quaternion conventions
    _quat = np.concatenate([quat[1:], quat[:1]])
    r = R.from_quat(_quat)
    # Get euler angles in xyz order (roll, pitch, yaw)
    euler = r.as_euler('xyz', degrees=False)
    return euler


# --- Main Simulation Setup ---

# Get the full path to the XML file
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# Initialize MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # Load robot model from XML
data = mj.MjData(model)                # Create data instance for simulation

# Set gravity for walking on slight incline
model.opt.gravity[0] = 9.81 * np.sin(0.1)  # Small x component for incline
model.opt.gravity[2] = -9.81 * np.cos(0.1) # Mostly downward gravity

# Initialize controller
init_controller(model, data)

# --- Main Simulation Loop ---
with mj.viewer.launch_passive(model, data) as viewer:
    # Set up camera view
    viewer.cam.azimuth = 90       # Side view
    viewer.cam.elevation = -20    # Look slightly down
    viewer.cam.distance = 40.0    # Distance from robot
    viewer.cam.lookat[:] = [10.0, 0.0, 0.1]  # Look at robot's center
    
    # Set up simulation parameters
    fps = 60  # Frame rate for both viewing and recording
    
    # Initialize video recording if enabled
    if RECORD_VIDEO:
        try:
            # Try high quality first
            width, height = 1920, 1080  # Full HD resolution
            
            # Set up video file path
            video_dir = os.path.dirname(os.path.abspath(__file__))
            video_path = os.path.join(video_dir, 'Biped_Simulation.avi')
            print(f"\nVideo will be saved to: {video_path}")
            
            # Create renderer for high quality video
            renderer = mj.Renderer(model, height=height, width=width)
            
            # Set initial camera parameters for renderer
            renderer.update_scene(data, camera=viewer.cam)
            
        except ValueError:
            # Fallback to a lower resolution if high quality fails
            print("\nFalling back to 1280x720 resolution...")
            width, height = 1280, 720
            renderer = mj.Renderer(model, height=height, width=width)
            renderer.update_scene(data, camera=viewer.cam)
        
        # Use XVID codec which is more widely supported
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)
        
        if not out.isOpened():
            raise RuntimeError("Failed to create video writer. Check if you have write permissions in this directory.")
        
        print("Recording simulation... (Move camera with mouse to adjust view)")
        frames_written = 0  # Counter to track number of frames written
    else:
        print("Running simulation... (Move camera with mouse to adjust view)")
    
    # Variables for timing
    sim_start = time.time()
    last_update = sim_start
    
    # Run simulation until end time or window closes
    while viewer.is_running() and data.time < simend:
        # Maintain consistent simulation speed
        now = time.time()
        sim_time = now - sim_start
        
        # Run physics steps to catch up to wall clock
        while data.time < sim_time and data.time < simend:
            mj.mj_step(model, data)
            controller(model, data)
        
        # Update visualization
        if now - last_update >= 1.0/fps:  # Ensure consistent frame rate
            viewer.sync()
            
            if RECORD_VIDEO:
                # Update renderer with current camera view
                renderer.update_scene(data, camera=viewer.cam)
                
                # Render and save frame
                pixels = renderer.render()
                out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                frames_written += 1
            
            last_update = now

    # Clean up
    print(f"\nSimulation finished. Total time: {time.time() - sim_start:.2f} seconds")
    if RECORD_VIDEO:
        print(f"Frames written: {frames_written}")
        print("Saving high-quality video...")
        out.release()
        
        # Verify the video file was created
        if os.path.exists(video_path):
            print(f"Video saved successfully to: {video_path}")
            print(f"File size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
        else:
            print("Warning: Video file was not created successfully!")
