import mujoco
import mujoco.viewer
import time # needed for screenshots, joint timings, and ending the sim after 10s
import os # needed for file paths
import cv2  # For video recording
import numpy as np
from scipy.spatial.transform import Rotation as R # needed for rotation calculations
import xml.etree.ElementTree as ET # needed for modifying the XML file
import matplotlib.pyplot as plt  # For plotting

# Clear terminal before running
os.system('cls')

# === Simulation and Recording Settings ===
RECORD_VIDEO = False  # Set to True to record the simulation as a video
TARGET_FPS = 30       # Target frames per second for simulation and video
PLAYBACK_SPEED = 4.0  # Playback speed for video - higher values make playback slower

# Individual ramp durations for each joint (seconds)
RIGHT_HIP_RAMP_DURATION = 0.5
LEFT_HIP_RAMP_DURATION = 0.5
RIGHT_KNEE_RAMP_DURATION = 0.4
LEFT_KNEE_RAMP_DURATION = 0.4

# === Foot Ground Detection Settings ===
# FOOT_GROUND_THRESHOLD = 0.25  # meters - distance from ground to trigger leg switch
# LEG_SWITCH_RAMP_DURATION = 0.5  # seconds for leg switch to complete

CYLINDER_LEGS = False  # Set to True for flat-bottomed (cylinder) legs, False for rounded (capsule) legs

# === Actuation Target Angles (degrees) ===
RIGHT_KNEE_TARGET_ANGLE = 0  # Target angle for right knee actuator
RIGHT_HIP_TARGET_ANGLE = 90   # Target angle for right hip actuator Y
LEFT_HIP_TARGET_ANGLE = 0   # Target angle for left hip actuator Y
LEFT_KNEE_TARGET_ANGLE = 0     # Target angle for left knee actuator
LEAN_ANGLE = 0  # degrees, adjust as needed # x hip rotation

# === Hip Z (Rotation) Target Angles ===
RIGHT_HIP_Z_TARGET = 0  # degrees - positive rotates leg outward (axis="0 0 1")
LEFT_HIP_Z_TARGET = 0   # degrees - keep left leg straight (axis="0 0 -1")

# === Switched Leg Target Angles (degrees) ===
# SWITCHED_RIGHT_KNEE_TARGET_ANGLE = 0    # Right leg straightens
# SWITCHED_RIGHT_HIP_TARGET_ANGLE = 0     # Right hip straightens
# SWITCHED_LEFT_KNEE_TARGET_ANGLE = -40   # Left knee bends
# SWITCHED_LEFT_HIP_TARGET_ANGLE = -90    # Left hip bends

# === Lean Ramp Timing ===
LEAN_RAMP_DURATION = 1.0  # seconds for lean to complete
LEAN_RAMP_START_TIME = 0.0
LEAN_RAMP_END_TIME = LEAN_RAMP_START_TIME + LEAN_RAMP_DURATION

# === Hip/Knee Ramp Timing (start after lean completes) ===
ramp_start_time = LEAN_RAMP_END_TIME
ramp_end_time_right_hip = ramp_start_time + RIGHT_HIP_RAMP_DURATION
ramp_end_time_left_hip = ramp_start_time + LEFT_HIP_RAMP_DURATION
ramp_end_time_right_knee = ramp_start_time + RIGHT_KNEE_RAMP_DURATION
ramp_end_time_left_knee = ramp_start_time + LEFT_KNEE_RAMP_DURATION

# === Model Loading ===
# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "Baymax.xml")  # Path to the MuJoCo XML model

# Modify XML for leg geometry toggle
if CYLINDER_LEGS:
    # Parse the original XML file into a tree structure for modification
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for geom in root.iter('geom'):
        if geom.attrib.get('name') in ['thigh_right', 'thigh_left', 'shin_right', 'shin_left']:
            if geom.attrib.get('type') == 'capsule':
                geom.set('type', 'cylinder')
    # # Create a temporary XML file path for the modified model
    temp_xml_path = os.path.join(current_dir, "Baymax_temp.xml")
    # Save the modified XML tree to the temporary file
    tree.write(temp_xml_path)
    # Load the modified model from the temporary XML file
    model = mujoco.MjModel.from_xml_path(temp_xml_path)
else:
    # If CYLINDER_LEGS is False, load the original XML file unchanged
    model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Reset the simulation to the initial state
mujoco.mj_resetData(model, data)

# Get body IDs for height tracking
torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')

# Get the site ID for the overall COM marker
overall_com_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'overall_com')

# === Joint and Actuator Lookup ===
# Print all joint names for debugging (uncomment for troubleshooting)
# print("\nAvailable joint names:")
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if joint_name:
        # print(f"Joint {i}: {joint_name}")
        pass

# Find the right hip and knee joints and actuators by name
right_hip_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_right')
right_hip_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_right')
right_knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_right')
right_knee_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_right')

# Find the left hip and knee joints and actuators by name
left_hip_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_left')
left_hip_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_left')
left_knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_left')
left_knee_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_left')

# Find the ab/adduction (hip_x) actuators for left and right hips
hip_x_left_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_x_left')
hip_x_right_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_x_right')

# Find the hip_z (rotation) joints and actuators
hip_z_left_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_z_left')
hip_z_right_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_z_right')
hip_z_left_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_z_left')
hip_z_right_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_z_right')
print(f"\nZ rotation IDs:")
print(f"Left hip Z - Joint: {hip_z_left_joint_id}, Actuator: {hip_z_left_actuator_id}")
print(f"Right hip Z - Joint: {hip_z_right_joint_id}, Actuator: {hip_z_right_actuator_id}")

# Let the simulation stabilize for a few steps before taking initial height
for _ in range(100):  # Run 100 steps to let the robot settle
    # Always apply Z rotation targets
    data.ctrl[hip_z_right_actuator_id] = RIGHT_HIP_Z_TARGET
    data.ctrl[hip_z_left_actuator_id] = LEFT_HIP_Z_TARGET
    mujoco.mj_step(model, data)
initial_torso_height = data.xpos[torso_id][2]  # Store initial torso height (z-coordinate)

# Initialize movement flag to ensure joints are only moved once
moved = False
angle_ramp_started = False

# Initialize leg switching variables
# leg_switch_triggered = False
# leg_switch_start_time = None
# leg_switch_completed = False

# Store initial joint angles for later comparison
initial_angles = {
    'left_hip': data.qpos[model.jnt_qposadr[left_hip_joint_id]],
    'right_hip': data.qpos[model.jnt_qposadr[right_hip_joint_id]],
    'left_knee': data.qpos[model.jnt_qposadr[left_knee_joint_id]],
    'right_knee': data.qpos[model.jnt_qposadr[right_knee_joint_id]],
    'left_hip_z': np.degrees(data.qpos[hip_z_left_joint_id]),
    'right_hip_z': np.degrees(data.qpos[hip_z_right_joint_id])
}

# Print initial angles for debugging (uncomment for troubleshooting)
# print("\nInitial angles:")
for joint, angle in initial_angles.items():
    # print(f"{joint}: {angle:.2f}°")
    pass

# Data collection for plotting
plot_data = {
    'time': [],
    'hip_y_target_right': [],
    'hip_x_target_right': [],
    'hip_z_target_right': [],
    'knee_target_right': [],
    'hip_y_actual_right': [],
    'hip_x_actual_right': [],
    'hip_z_actual_right': [],
    'knee_actual_right': [],
    'hip_y_target_left': [],
    'hip_x_target_left': [],
    'hip_z_target_left': [],
    'knee_target_left': [],
    'hip_y_actual_left': [],
    'hip_x_actual_left': [],
    'hip_z_actual_left': [],
    'knee_actual_left': []
}

def get_joint_positions(model, data):
    """
    Get the joint positions in world coordinates using MuJoCo's joint anchor data.
    """
    # Get joint IDs
    left_hip_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_left')
    left_knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_left')
    right_hip_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_right')
    right_knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_right')
    
    # Get actual joint positions using joint anchor points (most accurate)
    left_hip_pos = data.xanchor[left_hip_joint_id]
    left_knee_pos = data.xanchor[left_knee_joint_id]
    right_hip_pos = data.xanchor[right_hip_joint_id]
    right_knee_pos = data.xanchor[right_knee_joint_id]
    
    return {
        'left_hip': left_hip_pos,
        'left_knee': left_knee_pos,
        'right_hip': right_hip_pos,
        'right_knee': right_knee_pos
    }

def get_all_joint_angles(model, data):
    """
    Get the actual joint angles from MuJoCo data for plotting for BOTH legs.
    Returns angles in degrees.
    """
    # Get RIGHT joint IDs
    hip_y_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_right')
    hip_x_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_right')
    hip_z_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_z_right')
    knee_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_right')
    
    # Get RIGHT joint angles (convert from radians to degrees)
    hip_y_angle_right = np.degrees(data.qpos[hip_y_right_id])
    hip_x_angle_right = np.degrees(data.qpos[hip_x_right_id])
    hip_z_angle_right = np.degrees(data.qpos[hip_z_right_id])
    knee_angle_right = np.degrees(data.qpos[knee_right_id])

    # Get LEFT joint IDs
    hip_y_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_left')
    hip_x_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_left')
    hip_z_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_z_left')
    knee_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_left')

    # Get LEFT joint angles (convert from radians to degrees)
    hip_y_angle_left = np.degrees(data.qpos[hip_y_left_id])
    hip_x_angle_left = np.degrees(data.qpos[hip_x_left_id])
    hip_z_angle_left = np.degrees(data.qpos[hip_z_left_id])
    knee_angle_left = np.degrees(data.qpos[knee_left_id])
    
    return (hip_y_angle_right, hip_x_angle_right, hip_z_angle_right, knee_angle_right,
            hip_y_angle_left, hip_x_angle_left, hip_z_angle_left, knee_angle_left)

def plot_joint_angles():
    """
    Plot the collected target vs actual joint angles data.
    """
    print(f"\nPlotting function called!")
    print(f"Data points collected: {len(plot_data['time'])}")
    
    if not plot_data['time']:
        print("No data collected for plotting.")
        return
    
    print(f"Time range: {min(plot_data['time']):.2f}s to {max(plot_data['time']):.2f}s")
    print(f"Sample target angles - Hip Y: {plot_data['hip_y_target_right'][-1]:.2f}°, Knee: {plot_data['knee_target_right'][-1]:.2f}°")
    print(f"Sample actual angles - Hip Y: {plot_data['hip_y_actual_right'][-1]:.2f}°, Knee: {plot_data['knee_actual_right'][-1]:.2f}°")
    
    # Convert lists to numpy arrays
    time_array = np.array(plot_data['time'])
    # Right Leg Data
    hip_y_target_right = np.array(plot_data['hip_y_target_right'])
    hip_x_target_right = np.array(plot_data['hip_x_target_right'])
    hip_z_target_right = np.array(plot_data['hip_z_target_right'])
    knee_target_right = np.array(plot_data['knee_target_right'])
    hip_y_actual_right = np.array(plot_data['hip_y_actual_right'])
    hip_x_actual_right = np.array(plot_data['hip_x_actual_right'])
    hip_z_actual_right = np.array(plot_data['hip_z_actual_right'])
    knee_actual_right = np.array(plot_data['knee_actual_right'])
    # Left Leg Data
    hip_y_target_left = np.array(plot_data['hip_y_target_left'])
    hip_x_target_left = np.array(plot_data['hip_x_target_left'])
    hip_z_target_left = np.array(plot_data['hip_z_target_left'])
    knee_target_left = np.array(plot_data['knee_target_left'])
    hip_y_actual_left = np.array(plot_data['hip_y_actual_left'])
    hip_x_actual_left = np.array(plot_data['hip_x_actual_left'])
    hip_z_actual_left = np.array(plot_data['hip_z_actual_left'])
    knee_actual_left = np.array(plot_data['knee_actual_left'])
    
    print("Creating plots...")
    
    # Create a single figure with a 4x2 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Target vs Actual Joint Angles Over Time', fontsize=16)

    # --- Right Leg Plots (Column 0) ---
    # Hip Y (Right)
    axes[0, 0].plot(time_array, hip_y_target_right, 'b-', label='Target', linewidth=2)
    axes[0, 0].plot(time_array, hip_y_actual_right, 'r--', label='Actual', linewidth=2)
    axes[0, 0].set_title('Right Hip Y (Forward/Back)')
    axes[0, 0].set_ylabel('Angle (degrees)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Hip X (Right)
    axes[1, 0].plot(time_array, hip_x_target_right, 'b-', label='Target', linewidth=2)
    axes[1, 0].plot(time_array, hip_x_actual_right, 'r--', label='Actual', linewidth=2)
    axes[1, 0].set_title('Right Hip X (Side-to-Side)')
    axes[1, 0].set_ylabel('Angle (degrees)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Hip Z (Right)
    axes[2, 0].plot(time_array, hip_z_target_right, 'b-', label='Target', linewidth=2)
    axes[2, 0].plot(time_array, hip_z_actual_right, 'r--', label='Actual', linewidth=2)
    axes[2, 0].set_title('Right Hip Z (Rotation)')
    axes[2, 0].set_ylabel('Angle (degrees)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Knee (Right)
    axes[3, 0].plot(time_array, knee_target_right, 'b-', label='Target', linewidth=2)
    axes[3, 0].plot(time_array, knee_actual_right, 'r--', label='Actual', linewidth=2)
    axes[3, 0].set_title('Right Knee (Flexion/Extension)')
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_ylabel('Angle (degrees)')
    axes[3, 0].legend()
    axes[3, 0].grid(True)

    # --- Left Leg Plots (Column 1) ---
    # Hip Y (Left)
    axes[0, 1].plot(time_array, hip_y_target_left, 'b-', label='Target', linewidth=2)
    axes[0, 1].plot(time_array, hip_y_actual_left, 'r--', label='Actual', linewidth=2)
    axes[0, 1].set_title('Left Hip Y (Forward/Back)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Hip X (Left)
    axes[1, 1].plot(time_array, hip_x_target_left, 'b-', label='Target', linewidth=2)
    axes[1, 1].plot(time_array, hip_x_actual_left, 'r--', label='Actual', linewidth=2)
    axes[1, 1].set_title('Left Hip X (Side-to-Side)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Hip Z (Left)
    axes[2, 1].plot(time_array, hip_z_target_left, 'b-', label='Target', linewidth=2)
    axes[2, 1].plot(time_array, hip_z_actual_left, 'r--', label='Actual', linewidth=2)
    axes[2, 1].set_title('Left Hip Z (Rotation)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    # Knee (Left)
    axes[3, 1].plot(time_array, knee_target_left, 'b-', label='Target', linewidth=2)
    axes[3, 1].plot(time_array, knee_actual_left, 'r--', label='Actual', linewidth=2)
    axes[3, 1].set_title('Left Knee (Flexion/Extension)')
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].legend()
    axes[3, 1].grid(True)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("Showing plots...")
    plt.show()
    print("Plots displayed!")
    
    # Print summary statistics
    print("\n--- Right Leg Summary ---")
    print(f"Hip Y - Max Error: {np.max(np.abs(hip_y_target_right - hip_y_actual_right)):.2f}°, RMS Error: {np.sqrt(np.mean((hip_y_target_right - hip_y_actual_right)**2)):.2f}°")
    print(f"Hip X - Max Error: {np.max(np.abs(hip_x_target_right - hip_x_actual_right)):.2f}°, RMS Error: {np.sqrt(np.mean((hip_x_target_right - hip_x_actual_right)**2)):.2f}°")
    print(f"Hip Z - Max Error: {np.max(np.abs(hip_z_target_right - hip_z_actual_right)):.2f}°, RMS Error: {np.sqrt(np.mean((hip_z_target_right - hip_z_actual_right)**2)):.2f}°")
    print(f"Knee  - Max Error: {np.max(np.abs(knee_target_right - knee_actual_right)):.2f}°, RMS Error: {np.sqrt(np.mean((knee_target_right - knee_actual_right)**2)):.2f}°")
    
    print("\n--- Left Leg Summary ---")
    print(f"Hip Y - Max Error: {np.max(np.abs(hip_y_target_left - hip_y_actual_left)):.2f}°, RMS Error: {np.sqrt(np.mean((hip_y_target_left - hip_y_actual_left)**2)):.2f}°")
    print(f"Hip X - Max Error: {np.max(np.abs(hip_x_target_left - hip_x_actual_left)):.2f}°, RMS Error: {np.sqrt(np.mean((hip_x_target_left - hip_x_actual_left)**2)):.2f}°")
    print(f"Hip Z - Max Error: {np.max(np.abs(hip_z_target_left - hip_z_actual_left)):.2f}°, RMS Error: {np.sqrt(np.mean((hip_z_target_left - hip_z_actual_left)**2)):.2f}°")
    print(f"Knee  - Max Error: {np.max(np.abs(knee_target_left - knee_actual_left)):.2f}°, RMS Error: {np.sqrt(np.mean((knee_target_left - knee_actual_left)**2)):.2f}°")

# === Angle Calculation Functions ===
def print_leg_angles(model, data):
    """
    Compute and print:
    - The angle between the left and right thigh vectors (hip extension/retraction)
    - The anatomical knee flexion/extension angle for each leg (thigh-to-shin)
    Uses actual joint positions for more accurate calculations.
    """
    # Helper function to compute the angle (in degrees) between two vectors
    def angle_between(v1, v2):
        # Normalize both vectors
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        # Compute dot product
        dot = np.dot(v1_u, v2_u)
        # Clip to [-1, 1] to avoid floating-point errors with arccos
        dot = np.clip(dot, -1.0, 1.0)
        # Return angle in degrees
        return np.degrees(np.arccos(dot))


    joint_positions = get_joint_positions(model, data)
    left_hip_pos = joint_positions['left_hip']
    left_knee_pos = joint_positions['left_knee']
    right_hip_pos = joint_positions['right_hip']
    right_knee_pos = joint_positions['right_knee']

    # --- Calculate thigh vectors (hip to knee) for each leg ---
    left_thigh_vec = left_knee_pos - left_hip_pos
    right_thigh_vec = right_knee_pos - right_hip_pos

    # --- Hip angle: angle between left and right thigh vectors (projected onto XZ plane) ---
    # Project both thigh vectors onto the XZ plane (ignore Y component)
    left_thigh_vec_proj = left_thigh_vec.copy()
    right_thigh_vec_proj = right_thigh_vec.copy()
    left_thigh_vec_proj[1] = 0
    right_thigh_vec_proj[1] = 0
    # Normalize and compute the angle between the projected vectors
    # HIP ANGLE: angle between left and right thigh vectors (projected onto XZ plane)
    hip_angle = angle_between(left_thigh_vec_proj, right_thigh_vec_proj)
    print(f"Hip extension/retraction angle (forward/backward): {hip_angle:.2f} degrees")

    # --- (Optional) Hip abduction/adduction using joint positions ---
    hip_x_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_left')
    hip_x_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_right')
    left_ext = data.qpos[model.jnt_qposadr[hip_x_left_id]]
    right_ext = data.qpos[model.jnt_qposadr[hip_x_right_id]]
    abd_angle_rad = left_ext - right_ext
    abd_angle_deg = np.degrees(abd_angle_rad)
    # print(f"Hip abduction/adduction angle (side-to-side): {abd_angle_deg:.2f} degrees")

    # --- Estimate ankle positions by extending the shin vectors ---
    # Use the shin's orientation and geometry to find the ankle position in world coordinates
    # The vector [0, 0, -0.35] is the local offset from knee to ankle in the shin's frame (from XML)
    left_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_left')
    right_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_right')
    
    # Get the shin's orientation and geometry to find the ankle position in world coordinates
    # The vector [0, 0, -0.35] is the local offset from knee to ankle in the shin's frame (from XML)
    left_shin_dir = data.xmat[left_shin_id].reshape(3, 3) @ np.array([0, 0, -0.35])
    left_ankle_pos = left_knee_pos + left_shin_dir
    right_shin_dir = data.xmat[right_shin_id].reshape(3, 3) @ np.array([0, 0, -0.35])
    right_ankle_pos = right_knee_pos + right_shin_dir

    # --- Calculate shin vectors (knee to ankle) for each leg ---
    left_shin_vec = left_ankle_pos - left_knee_pos
    right_shin_vec = right_ankle_pos - right_knee_pos

    # --- Knee angles: angle between thigh and shin vectors for each leg ---
    left_knee_angle = angle_between(left_thigh_vec, left_shin_vec)
    right_knee_angle = angle_between(right_thigh_vec, right_shin_vec)
    print(f"Left knee flexion/extension angle: {left_knee_angle:.2f} degrees")
    print(f"Right knee flexion/extension angle: {right_knee_angle:.2f} degrees")

def compute_overall_com(model, data):
    """
    Compute the overall center of mass of the entire model.
    Returns a 3D numpy array with the (x, y, z) coordinates of the COM in world coordinates.
    """
    total_mass = 0.0
    weighted_sum = np.zeros(3)
    for i in range(model.nbody):
        mass = model.body_mass[i]
        pos = data.xpos[i]
        total_mass += mass
        weighted_sum += mass * pos
    return weighted_sum / total_mass

# def detect_foot_ground_proximity(model, data):
#     """
#     Detect when the RIGHT foot gets close to the ground (within FOOT_GROUND_THRESHOLD).
#     Returns True if the right foot is close to the ground, False otherwise.
#     """
#     # Get body ID for the right shin (which represents the right foot)
#     right_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_right')
    
#     # Get the world position of the right shin center of mass
#     right_shin_pos = data.xpos[right_shin_id]
    
#     # Calculate the bottom position of the right foot (shin bottom)
#     # The shin extends 0.35 units down from the knee, so the foot is at the bottom
#     right_shin_dir = data.xmat[right_shin_id].reshape(3, 3) @ np.array([0, 0, -0.35])
#     right_foot_pos = right_shin_pos + right_shin_dir
    
#     # Check if the right foot is close to the ground (z-coordinate close to 0)
#     right_foot_height = right_foot_pos[2]
    
#     # Return True if the right foot is within the threshold of the ground
#     return right_foot_height <= FOOT_GROUND_THRESHOLD

def get_single_joint_angle(model, data, joint_name, convert_to_degrees=True):
    """
    Get a single joint angle directly using MuJoCo's API.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_name: Name of the joint (e.g., 'hip_y_right', 'knee_left')
        convert_to_degrees: If True, converts the angle from radians to degrees
    
    Returns:
        float: Joint angle (in degrees if convert_to_degrees=True, otherwise in radians)
    """
    # Get the joint ID
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    
    # Get the angle in radians using the joint's position address
    angle_radians = data.qpos[model.jnt_qposadr[joint_id]]
    
    # Convert to degrees if requested
    if convert_to_degrees:
        return np.degrees(angle_radians)
    return angle_radians

# === Main Simulation Loop ===
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set camera to a zoomed out position for better visibility
    viewer.cam.distance = 10.0  # Distance from camera to target
    viewer.cam.azimuth = 90     # Camera rotation around the z-axis in degrees
    viewer.cam.elevation = -20  # Camera elevation in degrees

    # Always set up a renderer for screenshots
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=viewer.cam)

    # Video recording setup (if enabled)
    if RECORD_VIDEO:
        video_filename = 'baymax_simulation.avi'
        video_path = os.path.join(current_dir, video_filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # Use PLAYBACK_SPEED to control video playback speed
        # Higher PLAYBACK_SPEED values make the video play slower
        video_fps = TARGET_FPS / PLAYBACK_SPEED
        out = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height), isColor=True)
        if not out.isOpened():
            raise RuntimeError("Failed to create video writer. Check if you have write permissions in this directory.")
        frames_written = 0
        print(f"Recording video with playback speed {PLAYBACK_SPEED}x (FPS: {video_fps:.1f})")

    start_time = time.time()
    # Simulation parameters
    fps = TARGET_FPS
    timestep = model.opt.timestep
    steps_per_frame = int(1.0 / (fps * timestep))
    SIMULATION_SPEED = 6.0  # 1.0 is real-time, higher values are slower
    MAX_SIM_TIME = 10.0     # Limit simulation to 10 seconds

    # --- Screenshot logic ---
    screenshot_times = []
    # 1 at start, 3 during ramp, 1 at end
    screenshot_times.append(0.0)  # Start
    # Use 9 points and take [3, 6, 8] for mid-action screenshots, so screenshot 2 starts even later
    ramp_all_times = np.linspace(ramp_start_time, max(ramp_end_time_right_hip, ramp_end_time_left_hip, ramp_end_time_right_knee, ramp_end_time_left_knee), 9)
    ramp_mid_times = [ramp_all_times[3], ramp_all_times[6], ramp_all_times[8]]
    screenshot_times.extend(ramp_mid_times)
    # Add a screenshot about 1 second after the last ramp screenshot, but not exceeding MAX_SIM_TIME
    post_ramp_time = min(ramp_mid_times[-1] + 1.0, MAX_SIM_TIME)
    screenshot_times.append(post_ramp_time)
    screenshot_times.append(MAX_SIM_TIME)  # End
    screenshot_taken = [False] * len(screenshot_times)
    screenshot_paths = [os.path.join(current_dir, f"baymax_shot_{i+1}.png") for i in range(len(screenshot_times))]

    # Add variable to track minimum right foot height
    min_right_foot_height = float('inf')
    frame_count = 0 # Initialize frame counter for plotting
    
    while viewer.is_running():
        current_time = time.time() - start_time
        
        # ALWAYS maintain Z rotation targets (don't zero them)
        data.ctrl[hip_z_right_actuator_id] = RIGHT_HIP_Z_TARGET
        data.ctrl[hip_z_left_actuator_id] = LEFT_HIP_Z_TARGET

        # Collect data for plotting (every 10th frame to avoid too much data)
        if frame_count % 10 == 0:
            # Get current target angles
            # Right leg
            hip_y_target_current_right = data.ctrl[right_hip_actuator_id]
            hip_x_target_current_right = data.ctrl[hip_x_right_actuator_id]
            hip_z_target_current_right = data.ctrl[hip_z_right_actuator_id]
            knee_target_current_right = data.ctrl[right_knee_actuator_id]
            # Left leg
            hip_y_target_current_left = data.ctrl[left_hip_actuator_id]
            hip_x_target_current_left = data.ctrl[hip_x_left_actuator_id]
            hip_z_target_current_left = data.ctrl[hip_z_left_actuator_id]
            knee_target_current_left = data.ctrl[left_knee_actuator_id]
            
            # Get current actual angles using direct MuJoCo readings
            hip_y_actual_current_right = get_single_joint_angle(model, data, 'hip_y_right')
            hip_x_actual_current_right = get_single_joint_angle(model, data, 'hip_x_right')
            hip_z_actual_current_right = get_single_joint_angle(model, data, 'hip_z_right')
            knee_actual_current_right = get_single_joint_angle(model, data, 'knee_right')
            
            hip_y_actual_current_left = get_single_joint_angle(model, data, 'hip_y_left')
            hip_x_actual_current_left = get_single_joint_angle(model, data, 'hip_x_left')
            hip_z_actual_current_left = get_single_joint_angle(model, data, 'hip_z_left')
            knee_actual_current_left = get_single_joint_angle(model, data, 'knee_left')
            
            # Store data
            plot_data['time'].append(current_time)
            # Right leg data
            plot_data['hip_y_target_right'].append(hip_y_target_current_right)
            plot_data['hip_x_target_right'].append(hip_x_target_current_right)
            plot_data['hip_z_target_right'].append(hip_z_target_current_right)
            plot_data['knee_target_right'].append(knee_target_current_right)
            plot_data['hip_y_actual_right'].append(hip_y_actual_current_right)
            plot_data['hip_x_actual_right'].append(hip_x_actual_current_right)
            plot_data['hip_z_actual_right'].append(hip_z_actual_current_right)
            plot_data['knee_actual_right'].append(knee_actual_current_right)
            # Left leg data
            plot_data['hip_y_target_left'].append(hip_y_target_current_left)
            plot_data['hip_x_target_left'].append(hip_x_target_current_left)
            plot_data['hip_z_target_left'].append(hip_z_target_current_left)
            plot_data['knee_target_left'].append(knee_target_current_left)
            plot_data['hip_y_actual_left'].append(hip_y_actual_current_left)
            plot_data['hip_x_actual_left'].append(hip_x_actual_current_left)
            plot_data['hip_z_actual_left'].append(hip_z_actual_current_left)
            plot_data['knee_actual_left'].append(knee_actual_current_left)
        
        # Take screenshots at specified times
        for idx, t in enumerate(screenshot_times):
            if not screenshot_taken[idx] and current_time >= t:
                renderer.update_scene(data, camera=viewer.cam)
                pixels = renderer.render()
                cv2.imwrite(screenshot_paths[idx], cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                screenshot_taken[idx] = True
        if current_time > MAX_SIM_TIME:
            final_torso_height = data.xpos[torso_id][2]  # Get final torso height
            print(f"\nHeight Analysis:")
            print(f"Torso - Initial height (center of mass): {initial_torso_height:.3f} meters")
            print(f"Torso - Final height: {final_torso_height:.3f} meters")
            print(f"Torso - Height change: {final_torso_height - initial_torso_height:.3f} meters")
            
            # Plot the collected data
            print(f"\nSimulation ended. About to call plotting function...")
            try:
                plot_joint_angles()
            except Exception as e:
                print(f"Error in plotting function: {e}")
                print("Attempting to show plots anyway...")
                plt.show()
            break  # Exit after 10 seconds
        # --- Lean and Z rotation ramp first ---
        if current_time < LEAN_RAMP_END_TIME:
            # Only ramp lean if LEAN_ANGLE is not 0
            if LEAN_ANGLE != 0:
                alpha = (current_time - LEAN_RAMP_START_TIME) / LEAN_RAMP_DURATION
                alpha = min(max(alpha, 0.0), 1.0)
                data.ctrl[hip_x_left_actuator_id] = alpha * LEAN_ANGLE
                data.ctrl[hip_x_right_actuator_id] = -alpha * LEAN_ANGLE
            else:
                # If LEAN_ANGLE is 0, hold at 0
                data.ctrl[hip_x_left_actuator_id] = 0
                data.ctrl[hip_x_right_actuator_id] = 0

            # Always hold Z rotation and other joints at their target angles (0 or otherwise)
            data.ctrl[hip_z_right_actuator_id] = RIGHT_HIP_Z_TARGET
            data.ctrl[hip_z_left_actuator_id] = LEFT_HIP_Z_TARGET
            data.ctrl[right_knee_actuator_id] = RIGHT_KNEE_TARGET_ANGLE
            data.ctrl[right_hip_actuator_id] = 0  # Hold at 0 during lean phase
            data.ctrl[left_hip_actuator_id] = 0   # Hold at 0 during lean phase
            data.ctrl[left_knee_actuator_id] = LEFT_KNEE_TARGET_ANGLE

        # --- Hip/knee ramps after lean is complete ---
        elif (current_time >= ramp_start_time and current_time < max(ramp_end_time_right_hip, ramp_end_time_left_hip, ramp_end_time_right_knee, ramp_end_time_left_knee)):
            # Right hip - only ramp if target is not 0
            if RIGHT_HIP_TARGET_ANGLE != 0:
                if current_time < ramp_end_time_right_hip:
                    alpha_right_hip = (current_time - ramp_start_time) / RIGHT_HIP_RAMP_DURATION
                    alpha_right_hip = min(max(alpha_right_hip, 0.0), 1.0)
                    data.ctrl[right_hip_actuator_id] = alpha_right_hip * RIGHT_HIP_TARGET_ANGLE
                else:
                    data.ctrl[right_hip_actuator_id] = RIGHT_HIP_TARGET_ANGLE
            else:
                data.ctrl[right_hip_actuator_id] = 0

            # Left hip - only ramp if target is not 0
            if LEFT_HIP_TARGET_ANGLE != 0:
                if current_time < ramp_end_time_left_hip:
                    alpha_left_hip = (current_time - ramp_start_time) / LEFT_HIP_RAMP_DURATION
                    alpha_left_hip = min(max(alpha_left_hip, 0.0), 1.0)
                    data.ctrl[left_hip_actuator_id] = alpha_left_hip * LEFT_HIP_TARGET_ANGLE
                else:
                    data.ctrl[left_hip_actuator_id] = LEFT_HIP_TARGET_ANGLE
            else:
                data.ctrl[left_hip_actuator_id] = 0

            # Right knee - only ramp if target is not 0
            if RIGHT_KNEE_TARGET_ANGLE != 0:
                if current_time < ramp_end_time_right_knee:
                    alpha_right_knee = (current_time - ramp_start_time) / RIGHT_KNEE_RAMP_DURATION
                    alpha_right_knee = min(max(alpha_right_knee, 0.0), 1.0)
                    data.ctrl[right_knee_actuator_id] = alpha_right_knee * RIGHT_KNEE_TARGET_ANGLE
                else:
                    data.ctrl[right_knee_actuator_id] = RIGHT_KNEE_TARGET_ANGLE
            else:
                data.ctrl[right_knee_actuator_id] = 0

            # Left knee - only ramp if target is not 0
            if LEFT_KNEE_TARGET_ANGLE != 0:
                if current_time < ramp_end_time_left_knee:
                    alpha_left_knee = (current_time - ramp_start_time) / LEFT_KNEE_RAMP_DURATION
                    alpha_left_knee = min(max(alpha_left_knee, 0.0), 1.0)
                    data.ctrl[left_knee_actuator_id] = alpha_left_knee * LEFT_KNEE_TARGET_ANGLE
                else:
                    data.ctrl[left_knee_actuator_id] = LEFT_KNEE_TARGET_ANGLE
            else:
                data.ctrl[left_knee_actuator_id] = 0

            # Always maintain lean and Z rotation at their target angles
            if LEAN_ANGLE != 0:
                data.ctrl[hip_x_left_actuator_id] = LEAN_ANGLE
                data.ctrl[hip_x_right_actuator_id] = -LEAN_ANGLE
            else:
                data.ctrl[hip_x_left_actuator_id] = 0
                data.ctrl[hip_x_right_actuator_id] = 0

            data.ctrl[hip_z_right_actuator_id] = RIGHT_HIP_Z_TARGET
            data.ctrl[hip_z_left_actuator_id] = LEFT_HIP_Z_TARGET

        # Once all ramps are complete, maintain all joints at their final target angles
        else:
            # Set each joint to its target angle (0 or non-zero)
            data.ctrl[right_knee_actuator_id] = RIGHT_KNEE_TARGET_ANGLE
            data.ctrl[right_hip_actuator_id] = RIGHT_HIP_TARGET_ANGLE
            data.ctrl[left_hip_actuator_id] = LEFT_HIP_TARGET_ANGLE
            data.ctrl[left_knee_actuator_id] = LEFT_KNEE_TARGET_ANGLE
            data.ctrl[hip_x_left_actuator_id] = LEAN_ANGLE
            data.ctrl[hip_x_right_actuator_id] = -LEAN_ANGLE
            data.ctrl[hip_z_right_actuator_id] = RIGHT_HIP_Z_TARGET
            data.ctrl[hip_z_left_actuator_id] = LEFT_HIP_Z_TARGET
        
        # Run fixed number of steps for smooth video
        for _ in range(steps_per_frame):
            # ALWAYS maintain Z rotation targets before each physics step
            data.ctrl[hip_z_right_actuator_id] = RIGHT_HIP_Z_TARGET
            data.ctrl[hip_z_left_actuator_id] = LEFT_HIP_Z_TARGET
            
            # Debug: Print Z rotation angles and control signals before step
            right_z_angle = np.degrees(data.qpos[hip_z_right_joint_id])
            right_z_ctrl = data.ctrl[hip_z_right_actuator_id]
            right_z_qvel = np.degrees(data.qvel[hip_z_right_joint_id])  # Angular velocity
            right_z_qacc = np.degrees(data.qacc[hip_z_right_joint_id])  # Angular acceleration
            print(f"\nRight Hip Z - Angle: {right_z_angle:.2f}°, Control: {right_z_ctrl:.2f}, Velocity: {right_z_qvel:.2f}°/s, Acceleration: {right_z_qacc:.2f}°/s²")
            
            mujoco.mj_step(model, data)
            
            # Debug: Print Z rotation angles after step
            right_z_angle_after = np.degrees(data.qpos[hip_z_right_joint_id])
            print(f"After step - Right Hip Z Angle: {right_z_angle_after:.2f}°, Change: {right_z_angle_after - right_z_angle:.2f}°")
        
        # Update the overall COM site position AFTER physics step
        com = compute_overall_com(model, data)
        print(f"COM position: {com}")  # Debug print
        model.site_pos[overall_com_site_id] = com # Update the overall COM site position
        mujoco.mj_forward(model, data)  # Update the scene after modifying site position
        # (Optional) Print current joint angles and changes
        current_angles = {
            'left_hip': data.qpos[model.jnt_qposadr[left_hip_joint_id]],
            'right_hip': data.qpos[model.jnt_qposadr[right_hip_joint_id]],
            'left_knee': data.qpos[model.jnt_qposadr[left_knee_joint_id]],
            'right_knee': data.qpos[model.jnt_qposadr[right_knee_joint_id]],
            'left_hip_z': np.degrees(data.qpos[hip_z_left_joint_id]),
            'right_hip_z': np.degrees(data.qpos[hip_z_right_joint_id])
        }
        angle_changes = {
            joint: current_angles[joint] - initial_angles[joint]
            for joint in current_angles
        }
        if moved:
            print(f"\nCurrent Z rotation angles:")
            print(f"Right hip Z: {current_angles['right_hip_z']:.2f}°")
            print(f"Left hip Z: {current_angles['left_hip_z']:.2f}°")
        # Update the viewer window
        viewer.sync()
        # Record video frame if enabled
        if RECORD_VIDEO:
            renderer.update_scene(data, camera=viewer.cam)
            pixels = renderer.render()
            out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            frames_written += 1
        # Print the hip and knee angles for analysis
        print_leg_angles(model, data)
        
        # Example: Get and print actual joint positions (uncomment to see)
        # joint_positions = get_joint_positions(model, data)
        # print(f"Joint positions - Left hip: {joint_positions['left_hip']}, Right hip: {joint_positions['right_hip']}")
        # print(f"Joint positions - Left knee: {joint_positions['left_knee']}, Right knee: {joint_positions['right_knee']}")
        
        # Debug: Print foot heights and ground proximity status
        if moved:
            left_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_left')
            right_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_right')
            
            left_shin_pos = data.xpos[left_shin_id]
            right_shin_pos = data.xpos[right_shin_id]
            
            left_shin_dir = data.xmat[left_shin_id].reshape(3, 3) @ np.array([0, 0, -0.35])
            left_foot_pos = left_shin_pos + left_shin_dir
            
            right_shin_dir = data.xmat[right_shin_id].reshape(3, 3) @ np.array([0, 0, -0.35])
            right_foot_pos = right_shin_pos + right_shin_dir
            
            # Track minimum right foot height
            if right_foot_pos[2] < min_right_foot_height:
                min_right_foot_height = right_foot_pos[2]
            
            print(f"Foot heights - Left: {left_foot_pos[2]:.3f}m, Right: {right_foot_pos[2]:.3f}m")
            print(f"Leg switch status - Triggered: {False}, Completed: {False}")
        
        # Control simulation rate (slower for easier observation)
        time.sleep(0.01 * SIMULATION_SPEED)
        frame_count += 1 # Increment frame counter
    # Clean up video recording if enabled
    if RECORD_VIDEO:
        out.release()
        if os.path.exists(video_path):
            pass  # Uncomment print statements for debugging
        else:
            pass  # Uncomment print statements for debugging
    # After simulation loop, print minimum right foot height
    print(f"\nMinimum right foot height during simulation: {min_right_foot_height:.4f} m") 