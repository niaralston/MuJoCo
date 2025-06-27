import mujoco
import mujoco.viewer
import time
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Clear terminal before running
os.system('cls')

# === Simulation and Recording Settings ===
RECORD_VIDEO = False
TARGET_FPS = 30
PLAYBACK_SPEED = 4.0

# === Testing Configuration ===
# Choose which parameter to test (only one can be True)
TEST_STIFFNESS = False
TEST_DAMPING = True
TEST_KP = False

# Test values for each parameter
KP_VALUES = [30.75, 30.77, 30.79]  # Values to test when TEST_KP = True
STIFFNESS_VALUES = [999, 1000, 1001, 1100]  # Values to test when TEST_STIFFNESS = True
DAMPING_VALUES = [100, 1000, 10000]  # Values to test when TEST_DAMPING = True

# Individual ramp durations for each joint (seconds)
RIGHT_HIP_RAMP_DURATION = 0.5
LEFT_HIP_RAMP_DURATION = 0.5
RIGHT_KNEE_RAMP_DURATION = 0.4
LEFT_KNEE_RAMP_DURATION = 0.4

CYLINDER_LEGS = False

# === Actuation Target Angles (degrees) ===
RIGHT_KNEE_TARGET_ANGLE = 0
RIGHT_HIP_TARGET_ANGLE = 90
LEFT_HIP_TARGET_ANGLE = 0
LEFT_KNEE_TARGET_ANGLE = 0
LEAN_ANGLE = 0

# === Lean Ramp Timing ===
LEAN_RAMP_DURATION = 1.0
LEAN_RAMP_START_TIME = 0.0
LEAN_RAMP_END_TIME = LEAN_RAMP_START_TIME + LEAN_RAMP_DURATION

# === Hip/Knee Ramp Timing (start after lean completes) ===
ramp_start_time = LEAN_RAMP_END_TIME
ramp_end_time_right_hip = ramp_start_time + RIGHT_HIP_RAMP_DURATION
ramp_end_time_left_hip = ramp_start_time + LEFT_HIP_RAMP_DURATION
ramp_end_time_right_knee = ramp_start_time + RIGHT_KNEE_RAMP_DURATION
ramp_end_time_left_knee = ramp_start_time + LEFT_KNEE_RAMP_DURATION

def get_single_joint_angle(model, data, joint_name, convert_to_degrees=True):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    angle_radians = data.qpos[model.jnt_qposadr[joint_id]]
    if convert_to_degrees:
        return np.degrees(angle_radians)
    return angle_radians

def run_simulation(test_value):
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "Baymax.xml")

    # Modify XML to set test value
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the joint settings in the joint_stiffness class
    joint_stiffness = root.find(".//default/default[@class='joint_stiffness']/joint")
    if joint_stiffness is None:
        raise ValueError("Could not find joint_stiffness class in XML file")

    if TEST_STIFFNESS:
        # Update the joint stiffness value
        joint_stiffness.set('stiffness', str(test_value))
    elif TEST_DAMPING:
        # Update the joint damping value
        joint_stiffness.set('damping', str(test_value))
    else:  # TEST_KP
        # Update the default kp value
        position_default = root.find(".//default/position")
        if position_default is not None:
            position_default.set('kp', str(test_value))
    
    # Create a temporary XML file
    temp_xml_path = os.path.join(current_dir, f"Baymax_temp_{test_value}.xml")
    tree.write(temp_xml_path)
    
    # Load the modified model
    model = mujoco.MjModel.from_xml_path(temp_xml_path)
    data = mujoco.MjData(model)

    # Initialize data storage for all joints
    time_data = []
    joint_data = {
        'hip_y_right': {'target': [], 'actual': []},
        'hip_x_right': {'target': [], 'actual': []},
        'knee_right': {'target': [], 'actual': []},
        'hip_y_left': {'target': [], 'actual': []},
        'hip_x_left': {'target': [], 'actual': []},
        'knee_left': {'target': [], 'actual': []}
    }
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Run simulation without visualization
    start_time = time.time()
    frame_count = 0
    
    while True:
        current_time = time.time() - start_time
        
        if current_time >= 6.0:  # Run for 6 seconds
            break
            
        # Store data every 10th frame
        if frame_count % 10 == 0:
            time_data.append(current_time)
            
            # Store data for all joints
            for joint_name in joint_data.keys():
                actuator_name = joint_name.replace('hip_y', 'hip_y').replace('hip_x', 'hip_x')
                joint_data[joint_name]['target'].append(
                    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)])
                joint_data[joint_name]['actual'].append(
                    get_single_joint_angle(model, data, joint_name))
        
        # Apply control based on timing
        if current_time < LEAN_RAMP_END_TIME:
            # Hold all joints at initial positions
            for actuator_name in ['hip_y_right', 'hip_y_left', 'knee_right', 'knee_left']:
                data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)] = 0
            # Apply lean
            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_x_right')] = -LEAN_ANGLE
            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_x_left')] = LEAN_ANGLE
            
        elif current_time >= ramp_start_time:
            # Right hip Y
            if current_time < ramp_end_time_right_hip:
                alpha = (current_time - ramp_start_time) / RIGHT_HIP_RAMP_DURATION
                alpha = min(max(alpha, 0.0), 1.0)
                data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_right')] = alpha * RIGHT_HIP_TARGET_ANGLE
            else:
                data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_right')] = RIGHT_HIP_TARGET_ANGLE
                
            # Left hip Y
            if current_time < ramp_end_time_left_hip:
                alpha = (current_time - ramp_start_time) / LEFT_HIP_RAMP_DURATION
                alpha = min(max(alpha, 0.0), 1.0)
                data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_left')] = alpha * LEFT_HIP_TARGET_ANGLE
            else:
                data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_left')] = LEFT_HIP_TARGET_ANGLE
                
            # Right knee
            if current_time < ramp_end_time_right_knee:
                alpha = (current_time - ramp_start_time) / RIGHT_KNEE_RAMP_DURATION
                alpha = min(max(alpha, 0.0), 1.0)
                data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_right')] = alpha * RIGHT_KNEE_TARGET_ANGLE
            else:
                data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_right')] = RIGHT_KNEE_TARGET_ANGLE
                
            # Left knee
            if current_time < ramp_end_time_left_knee:
                alpha = (current_time - ramp_start_time) / LEFT_KNEE_RAMP_DURATION
                alpha = min(max(alpha, 0.0), 1.0)
                data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_left')] = alpha * LEFT_KNEE_TARGET_ANGLE
            else:
                data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_left')] = LEFT_KNEE_TARGET_ANGLE
            
            # Maintain lean
            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_x_right')] = -LEAN_ANGLE
            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_x_left')] = LEAN_ANGLE
        
        # Step simulation
        for _ in range(int(1.0 / (TARGET_FPS * model.opt.timestep))):
            mujoco.mj_step(model, data)
        
        frame_count += 1
    
    # Clean up temporary file
    os.remove(temp_xml_path)
    
    return time_data, joint_data

def plot_combined_results(results):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    # Determine which parameter is being tested
    if TEST_STIFFNESS:
        param_name = "Stiffness"
    elif TEST_DAMPING:
        param_name = "Damping"
    else:
        param_name = "kp"
    
    fig.suptitle(f'Target vs Actual Joint Angles Over Time\nComparison of Different {param_name} Values', fontsize=16)
    
    # Joint names and their subplot positions
    joint_plots = {
        'hip_y_right': (0, 0, 'Right Hip Y (Forward/Back)'),
        'hip_y_left': (0, 1, 'Left Hip Y (Forward/Back)'),
        'hip_x_right': (1, 0, 'Right Hip X (Side-to-Side)'),
        'hip_x_left': (1, 1, 'Left Hip X (Side-to-Side)'),
        'knee_right': (2, 0, 'Right Knee'),
        'knee_left': (2, 1, 'Left Knee')
    }
    
    # Create a color map that can handle any number of test values
    num_values = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, num_values))
    
    # Plot each joint
    for joint_name, (row, col, title) in joint_plots.items():
        ax = axes[row, col]
        
        # Plot target angle once (it's the same for all runs)
        ax.plot(results[0][0], results[0][1][joint_name]['target'], 'k--', 
                label='Target', linewidth=2)
        
        # Plot actual angles for each test value
        for i, (time_data, joint_data, value) in enumerate(results):
            param_label = f"{param_name}={value}"
            ax.plot(time_data, joint_data[joint_name]['actual'], color=colors[i], 
                    label=param_label, linewidth=1.5, alpha=0.8)
        
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('kp_comparison.png' if TEST_KP else ('stiffness_comparison.png' if TEST_STIFFNESS else 'damping_comparison.png'))
    plt.show()

# Validate test configuration
if sum([TEST_STIFFNESS, TEST_DAMPING, TEST_KP]) != 1:
    raise ValueError("Exactly one test type must be True (TEST_STIFFNESS, TEST_DAMPING, or TEST_KP)")

# Run simulations with selected test values
if TEST_STIFFNESS:
    test_values = STIFFNESS_VALUES
elif TEST_DAMPING:
    test_values = DAMPING_VALUES
else:
    test_values = KP_VALUES

results = []

print(f"Running simulations testing {'stiffness' if TEST_STIFFNESS else ('damping' if TEST_DAMPING else 'kp')} values...")
for value in test_values:
    print(f"Running simulation with {'stiffness' if TEST_STIFFNESS else ('damping' if TEST_DAMPING else 'kp')}={value}")
    time_data, joint_data = run_simulation(value)
    results.append((time_data, joint_data, value))

print("Creating combined plot...")
plot_combined_results(results) 