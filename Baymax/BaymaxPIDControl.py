import mujoco
import mujoco.viewer
import time # needed for screenshots, joint timings, and ending the sim after 10s
import os # needed for file paths
import cv2  # For video recording
import numpy as np
from scipy.spatial.transform import Rotation as R # needed for rotation calculations
import xml.etree.ElementTree as ET # needed for modifying the XML file
import matplotlib.pyplot as plt  # For plotting
import sys  # Added for sys.exit()

# Clear terminal before running
os.system('cls')

# === Control Parameters ===
USE_RAMP = True  # Set to False for on/off control, True for ramped control
KP = 0.0    # Removed since P control is handled by XML
# Three different integral gains based on error magnitude
KI_LARGE = 0   # Used when error > 30 degrees (large corrections)
KI_MEDIUM = 0  # Used when error is between 10-30 degrees (medium corrections)
KI_SMALL = 0   # Used when error < 10 degrees (fine-tuning near target)
KD = 0.0    # Removed D term since we're using XML's damping

# Error thresholds for gain scheduling (in degrees)
ERROR_LARGE_THRESHOLD = 20.0
ERROR_SMALL_THRESHOLD = 5.0

# Anti-windup limits for integral errors (in degrees)
INTEGRAL_LIMIT = 5.0  # Reduced to prevent large integral accumulation

# === Simulation and Recording Settings ===
RECORD_VIDEO = False  # Set to True to record the simulation as a video
TARGET_FPS = 30       # Target frames per second for simulation and video
PLAYBACK_SPEED = 4.0  # Playback speed for video - higher values make playback slower

# Individual ramp durations for each joint (seconds)
RIGHT_HIP_RAMP_DURATION = 1.0  # Duration for hip movement
LEFT_HIP_RAMP_DURATION = 1.0   # Duration for hip movement
RIGHT_KNEE_RAMP_DURATION = 1.0  # Duration for knee movement (slightly faster than hip)
LEFT_KNEE_RAMP_DURATION = 1.0   # Duration for knee movement (slightly faster than hip)

# Knee movement delay relative to hip (seconds)
RIGHT_KNEE_DELAY = 0
LEFT_KNEE_DELAY = 0   

# === Foot Ground Detection Settings ===
# FOOT_GROUND_THRESHOLD = 0.25  # meters - distance from ground to trigger leg switch
# LEG_SWITCH_RAMP_DURATION = 0.5  # seconds for leg switch to complete

CYLINDER_LEGS = False  # Set to True for flat-bottomed (cylinder) legs, False for rounded (capsule) legs

# === Actuation Target Angles (degrees) ===
RIGHT_KNEE_TARGET_ANGLE = -40  # Target angle for right knee actuator
RIGHT_HIP_TARGET_ANGLE = -90   # Target angle for right hip actuator Y
LEFT_HIP_TARGET_ANGLE = 0   # Target angle for left hip actuator Y
LEFT_KNEE_TARGET_ANGLE = 0     # Target angle for left knee actuator
LEAN_ANGLE = 0  # degrees, adjust as needed # x hip rotation

# === Switched Leg Target Angles (degrees) ===
# SWITCHED_RIGHT_KNEE_TARGET_ANGLE = 0    # Right leg straightens
# SWITCHED_RIGHT_HIP_TARGET_ANGLE = 0     # Right hip straightens
# SWITCHED_LEFT_KNEE_TARGET_ANGLE = -40   # Left knee bends
# SWITCHED_LEFT_HIP_TARGET_ANGLE = -90    # Left hip bends

# === Lean Ramp Timing ===
LEAN_RAMP_DURATION = 0  # seconds for lean to complete
LEAN_RAMP_START_TIME = 0.0
LEAN_RAMP_END_TIME = LEAN_RAMP_START_TIME + LEAN_RAMP_DURATION

# === Hip/Knee Ramp Timing (start after lean completes) ===
ramp_start_time = LEAN_RAMP_END_TIME
ramp_end_time_right_hip = ramp_start_time + RIGHT_HIP_RAMP_DURATION
ramp_end_time_left_hip = ramp_start_time + LEFT_HIP_RAMP_DURATION
# Knee timing now includes delay
ramp_start_time_right_knee = ramp_start_time + RIGHT_KNEE_DELAY
ramp_start_time_left_knee = ramp_start_time + LEFT_KNEE_DELAY
ramp_end_time_right_knee = ramp_start_time_right_knee + RIGHT_KNEE_RAMP_DURATION
ramp_end_time_left_knee = ramp_start_time_left_knee + LEFT_KNEE_RAMP_DURATION

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

def calculate_zmp(model, data, prev_lin_mom=None, prev_ang_mom=None, control_dt=0.005):
    """
    Calculate the Zero Moment Point (ZMP) using the simpler implementation from rewards.py
    """
    Gv = 9.80665
    total_mass = mujoco.mj_getTotalmass(model)
    Mg = total_mass * Gv

    # Get robot state
    com_pos = data.subtree_com[1].copy()
    lin_mom = data.subtree_linvel[1].copy() * total_mass
    ang_mom = data.subtree_angmom[1].copy() + np.cross(com_pos, lin_mom)

    # Calculate rates of change
    if prev_lin_mom is not None and prev_ang_mom is not None:
        d_lin_mom = (lin_mom - prev_lin_mom) / control_dt
        d_ang_mom = (ang_mom - prev_ang_mom) / control_dt
    else:
        d_lin_mom = lin_mom / control_dt
        d_ang_mom = ang_mom / control_dt

    Fgz = d_lin_mom[2] + Mg

    # Check for ground contact
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    has_contact = False
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 == floor_id or contact.geom2 == floor_id:
            has_contact = True
            break

    # Calculate ZMP
    if has_contact and Fgz > 20:  # Only calculate ZMP with ground contact and significant vertical force
        zmp_x = (Mg * com_pos[0] - d_ang_mom[1]) / Fgz
        zmp_y = (Mg * com_pos[1] + d_ang_mom[0]) / Fgz
    else:
        # If no contact or small force, ZMP is same as COM projection
        zmp_x = com_pos[0]
        zmp_y = com_pos[1]

    # Debug print every 100 frames
    if frame_count % 100 == 0:
        print(f"\nZMP Debug [frame {frame_count}]:")
        print(f"COM position: {com_pos}")
        print(f"ZMP position: ({zmp_x:.3f}, {zmp_y:.3f})")
        print(f"Ground contact: {has_contact}")
        print(f"Vertical force (Fgz): {Fgz:.2f} N")

    return zmp_x, zmp_y, lin_mom, ang_mom

# Try to get ZMP visualization site ID
zmp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'zmp')

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

# Store initial joint angles for later comparison
initial_angles = {
    'left_hip': data.qpos[model.jnt_qposadr[left_hip_joint_id]],
    'right_hip': data.qpos[model.jnt_qposadr[right_hip_joint_id]],
    'left_knee': data.qpos[model.jnt_qposadr[left_knee_joint_id]],
    'right_knee': data.qpos[model.jnt_qposadr[right_knee_joint_id]]
}

# Print initial angles for debugging (uncomment for troubleshooting)
# print("\nInitial angles:")
for joint, angle in initial_angles.items():
    # print(f"{joint}: {angle:.2f}°")
    pass

# Data collection for plotting
class DataCollector:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data = {
            'time': [],
            'hip_y_target_right': [],
            'hip_x_target_right': [],
            'knee_target_right': [],
            'hip_y_actual_right': [],
            'hip_x_actual_right': [],
            'knee_actual_right': [],
            'hip_y_target_left': [],
            'hip_x_target_left': [],
            'knee_target_left': [],
            'hip_y_actual_left': [],
            'hip_x_actual_left': [],
            'knee_actual_left': [],
            # Add new fields for raw target angles
            'hip_y_target_before_controller_right': [],
            'knee_target_before_controller_right': [],
            'hip_y_target_before_controller_left': [],
            'knee_target_before_controller_left': [],
            # Force components for each foot
            'left_foot_fx': [],
            'left_foot_fy': [],
            'left_foot_fz': [],
            'right_foot_fx': [],
            'right_foot_fy': [],
            'right_foot_fz': [],
            # Add ZMP and COM data collection
            'zmp_x': [],
            'zmp_y': [],
            'com_x': [],
            'com_y': [],
            'com_z': [],
            'stability_margin': []
        }
    
    def add_datapoint(self, time, model, data, actuator_ids, raw_targets=None, zmp_pos=None, com_pos=None, stability_margin=None):
        """
        Store all data points including ZMP and COM positions
        """
        # Get contact forces with components
        contact_forces = get_foot_ground_contact_forces(model, data)
        
        # Store all data
        self.data['time'].append(float(time))
        # Right leg
        self.data['hip_y_target_right'].append(float(data.ctrl[actuator_ids['right_hip']]))
        self.data['hip_x_target_right'].append(float(data.ctrl[actuator_ids['right_hip_x']]))
        self.data['knee_target_right'].append(float(data.ctrl[actuator_ids['right_knee']]))
        self.data['hip_y_actual_right'].append(float(get_single_joint_angle(model, data, 'hip_y_right')))
        self.data['hip_x_actual_right'].append(float(get_single_joint_angle(model, data, 'hip_x_right')))
        self.data['knee_actual_right'].append(float(get_single_joint_angle(model, data, 'knee_right')))
        # Left leg
        self.data['hip_y_target_left'].append(float(data.ctrl[actuator_ids['left_hip']]))
        self.data['hip_x_target_left'].append(float(data.ctrl[actuator_ids['left_hip_x']]))
        self.data['knee_target_left'].append(float(data.ctrl[actuator_ids['left_knee']]))
        self.data['hip_y_actual_left'].append(float(get_single_joint_angle(model, data, 'hip_y_left')))
        self.data['hip_x_actual_left'].append(float(get_single_joint_angle(model, data, 'hip_x_left')))
        self.data['knee_actual_left'].append(float(get_single_joint_angle(model, data, 'knee_left')))
        # Store raw target angles
        if raw_targets:
            self.data['hip_y_target_before_controller_right'].append(float(raw_targets['right_hip']))
            self.data['knee_target_before_controller_right'].append(float(raw_targets['right_knee']))
            self.data['hip_y_target_before_controller_left'].append(float(raw_targets['left_hip']))
            self.data['knee_target_before_controller_left'].append(float(raw_targets['left_knee']))
        else:
            self.data['hip_y_target_before_controller_right'].append(0.0)
            self.data['knee_target_before_controller_right'].append(0.0)
            self.data['hip_y_target_before_controller_left'].append(0.0)
            self.data['knee_target_before_controller_left'].append(0.0)
        # Force components
        self.data['left_foot_fx'].append(float(contact_forces['left_foot'][0]))
        self.data['left_foot_fy'].append(float(contact_forces['left_foot'][1]))
        self.data['left_foot_fz'].append(float(contact_forces['left_foot'][2]))
        self.data['right_foot_fx'].append(float(contact_forces['right_foot'][0]))
        self.data['right_foot_fy'].append(float(contact_forces['right_foot'][1]))
        self.data['right_foot_fz'].append(float(contact_forces['right_foot'][2]))
        
        # Calculate and store ZMP data directly
        Gv = 9.80665
        total_mass = mujoco.mj_getTotalmass(model)
        Mg = total_mass * Gv
        
        # Get robot state
        com_pos = data.subtree_com[1].copy()
        lin_mom = data.subtree_linvel[1].copy() * total_mass
        ang_mom = data.subtree_angmom[1].copy() + np.cross(com_pos, lin_mom)
        
        # Calculate forces
        d_lin_mom = lin_mom / 0.005  # Using default control_dt
        d_ang_mom = ang_mom / 0.005
        Fgz = d_lin_mom[2] + Mg
        
        # Check ground contact
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        left_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'shin_left')
        right_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'shin_right')
        
        has_contact = False
        for i in range(data.ncon):
            contact = data.contact[i]
            if ((contact.geom1 == floor_id and (contact.geom2 == left_shin_id or contact.geom2 == right_shin_id)) or
                ((contact.geom1 == left_shin_id or contact.geom1 == right_shin_id) and contact.geom2 == floor_id)):
                has_contact = True
                break
        
        # Calculate ZMP
        if has_contact and Fgz > 20:
            zmp_x = (Mg * com_pos[0] - d_ang_mom[1]) / Fgz
            zmp_y = (Mg * com_pos[1] + d_ang_mom[0]) / Fgz
        else:
            zmp_x = com_pos[0]
            zmp_y = com_pos[1]
            
        # Store ZMP and COM data
        self.data['zmp_x'].append(float(zmp_x))
        self.data['zmp_y'].append(float(zmp_y))
        self.data['com_x'].append(float(com_pos[0]))
        self.data['com_y'].append(float(com_pos[1]))
        self.data['com_z'].append(float(com_pos[2]))
        
        if stability_margin is not None:
            self.data['stability_margin'].append(float(stability_margin))

# Create data collector
plot_data = DataCollector()

# Integral error tracking
integral_errors = {
    'hip_y_right': 0.0,
    'hip_x_right': 0.0,
    'knee_right': 0.0,
    'hip_y_left': 0.0,
    'hip_x_left': 0.0,
    'knee_left': 0.0
}

def apply_pi_control(current_angle, target_angle, integral_key, dt):
    """
    Apply I control with gain scheduling based on error magnitude.
    Returns: Control effort that combines XML's P control with our I term
    """
    # Calculate error
    error = target_angle - current_angle
    error_magnitude = abs(error)
    
    # Select KI based on error magnitude
    if error_magnitude > ERROR_LARGE_THRESHOLD:
        ki = KI_LARGE
    elif error_magnitude > ERROR_SMALL_THRESHOLD:
        ki = KI_MEDIUM
    else:
        ki = KI_SMALL
    
    # Update integral term with anti-windup
    integral_errors[integral_key] += error * dt
    integral_errors[integral_key] = np.clip(integral_errors[integral_key], -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
    
    # Return the target angle for P control, but subtract the I term
    # This way, as the error persists, the I term will push against it
    control_effort = target_angle - ki * integral_errors[integral_key]
    
    return control_effort

def reset_integral_errors():
    """Reset all integral errors to zero"""
    for key in integral_errors:
        integral_errors[key] = 0.0

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
    knee_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_right')
    
    # Get RIGHT joint angles (convert from radians to degrees)
    hip_y_angle_right = np.degrees(data.qpos[hip_y_right_id])
    hip_x_angle_right = np.degrees(data.qpos[hip_x_right_id])
    knee_angle_right = np.degrees(data.qpos[knee_right_id])

    # Get LEFT joint IDs
    hip_y_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_left')
    hip_x_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_left')
    knee_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_left')

    # Get LEFT joint angles (convert from radians to degrees)
    hip_y_angle_left = np.degrees(data.qpos[hip_y_left_id])
    hip_x_angle_left = np.degrees(data.qpos[hip_x_left_id])
    knee_angle_left = np.degrees(data.qpos[knee_left_id])
    
    return (hip_y_angle_right, hip_x_angle_right, knee_angle_right,
            hip_y_angle_left, hip_x_angle_left, knee_angle_left)

# Get XML values for plotting
def get_xml_control_params():
    """Get the control parameters from the XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get Kp from position default
    kp = root.find(".//position").get('kp')
    
    # Get damping and stiffness from joint_stiffness class
    joint_params = root.find(".//default[@class='joint_stiffness']/joint")
    damping = joint_params.get('damping')
    stiffness = joint_params.get('stiffness')
    
    return float(kp), float(damping), float(stiffness)

def plot_joint_angles():
    """
    Plot the collected target vs actual joint angles data, contact forces, and ZMP analysis.
    """
    print(f"\nPlotting function called!")
    
    # Get XML control parameters
    kp, damping, stiffness = get_xml_control_params()
    
    # Ensure we have data to plot
    if not plot_data.data['time']:
        print("No data collected for plotting.")
        return
    
    # Create multiple figures for different aspects of the analysis
    fig_angles, axes_angles = plt.subplots(3, 2, figsize=(15, 10))
    fig_forces, (ax_forces1, ax_forces2) = plt.subplots(2, 1, figsize=(12, 8))
    fig_zmp = plt.figure(figsize=(15, 10))
    
    # === First figure: Joint angles ===
    # Right leg plots (left column)
    axes_angles[0, 0].plot(plot_data.data['time'], plot_data.data['hip_y_target_right'], 'r--', label='Controller Target')
    axes_angles[0, 0].plot(plot_data.data['time'], plot_data.data['hip_y_actual_right'], 'b-', label='Actual')
    axes_angles[0, 0].set_title('Right Hip Y')
    axes_angles[0, 0].set_ylabel('Angle (deg)')
    axes_angles[0, 0].grid(True)
    axes_angles[0, 0].legend()

    axes_angles[1, 0].plot(plot_data.data['time'], plot_data.data['hip_x_target_right'], 'r--', label='Controller Target')
    axes_angles[1, 0].plot(plot_data.data['time'], plot_data.data['hip_x_actual_right'], 'b-', label='Actual')
    axes_angles[1, 0].set_title('Right Hip X')
    axes_angles[1, 0].set_ylabel('Angle (deg)')
    axes_angles[1, 0].grid(True)
    axes_angles[1, 0].legend()

    axes_angles[2, 0].plot(plot_data.data['time'], plot_data.data['knee_target_right'], 'r--', label='Controller Target')
    axes_angles[2, 0].plot(plot_data.data['time'], plot_data.data['knee_actual_right'], 'b-', label='Actual')
    axes_angles[2, 0].set_title('Right Knee')
    axes_angles[2, 0].set_xlabel('Time (s)')
    axes_angles[2, 0].set_ylabel('Angle (deg)')
    axes_angles[2, 0].grid(True)
    axes_angles[2, 0].legend()

    # Left leg plots (right column)
    axes_angles[0, 1].plot(plot_data.data['time'], plot_data.data['hip_y_target_left'], 'r--', label='Controller Target')
    axes_angles[0, 1].plot(plot_data.data['time'], plot_data.data['hip_y_actual_left'], 'b-', label='Actual')
    axes_angles[0, 1].set_title('Left Hip Y')
    axes_angles[0, 1].grid(True)
    axes_angles[0, 1].legend()

    axes_angles[1, 1].plot(plot_data.data['time'], plot_data.data['hip_x_target_left'], 'r--', label='Controller Target')
    axes_angles[1, 1].plot(plot_data.data['time'], plot_data.data['hip_x_actual_left'], 'b-', label='Actual')
    axes_angles[1, 1].set_title('Left Hip X')
    axes_angles[1, 1].grid(True)
    axes_angles[1, 1].legend()

    axes_angles[2, 1].plot(plot_data.data['time'], plot_data.data['knee_target_left'], 'r--', label='Controller Target')
    axes_angles[2, 1].plot(plot_data.data['time'], plot_data.data['knee_actual_left'], 'b-', label='Actual')
    axes_angles[2, 1].set_title('Left Knee')
    axes_angles[2, 1].set_xlabel('Time (s)')
    axes_angles[2, 1].grid(True)
    axes_angles[2, 1].legend()

    # Add main title and subtitle with control parameters
    fig_angles.suptitle('Joint Angles vs Time\n' + 
                       f'Control Parameters: Hip Kp=35, Knee Kp=25, Ki=[{KI_SMALL:.1f}, {KI_MEDIUM:.1f}, {KI_LARGE:.1f}], Damping={damping}, Stiffness={stiffness}', 
                       fontsize=9, y=0.95)
    
    # === Second figure: Ground reaction forces ===
    # Left foot forces
    ax_forces1.plot(plot_data.data['time'], plot_data.data['left_foot_fx'], 'r-', label='Fx')
    ax_forces1.plot(plot_data.data['time'], plot_data.data['left_foot_fy'], 'g-', label='Fy')
    ax_forces1.plot(plot_data.data['time'], plot_data.data['left_foot_fz'], 'b-', label='Fz')
    ax_forces1.set_title('Left Foot Ground Reaction Forces')
    ax_forces1.set_ylabel('Force (N)')
    ax_forces1.grid(True)
    ax_forces1.legend()

    # Right foot forces
    ax_forces2.plot(plot_data.data['time'], plot_data.data['right_foot_fx'], 'r-', label='Fx')
    ax_forces2.plot(plot_data.data['time'], plot_data.data['right_foot_fy'], 'g-', label='Fy')
    ax_forces2.plot(plot_data.data['time'], plot_data.data['right_foot_fz'], 'b-', label='Fz')
    ax_forces2.set_title('Right Foot Ground Reaction Forces')
    ax_forces2.set_xlabel('Time (s)')
    ax_forces2.set_ylabel('Force (N)')
    ax_forces2.grid(True)
    ax_forces2.legend()

    fig_forces.suptitle('Ground Reaction Forces vs Time', fontsize=16)
    
    # === Third figure: ZMP Analysis ===
    gs = fig_zmp.add_gridspec(2, 2)
    ax_zmp_xy = fig_zmp.add_subplot(gs[0, 0])  # Top-left: XY plot
    ax_zmp_time = fig_zmp.add_subplot(gs[0, 1])  # Top-right: Time series
    ax_margin = fig_zmp.add_subplot(gs[1, :])    # Bottom: Stability margin
    
    # Plot ZMP and COM trajectories in XY plane
    ax_zmp_xy.plot(plot_data.data['zmp_x'], plot_data.data['zmp_y'], 'g-', label='ZMP', linewidth=2)
    ax_zmp_xy.plot(plot_data.data['com_x'], plot_data.data['com_y'], 'r--', label='COM', linewidth=2)
    
    # Set reasonable axis limits based on robot dimensions
    max_range = 2.0  # Maximum expected distance in meters
    ax_zmp_xy.set_xlim(-max_range, max_range)
    ax_zmp_xy.set_ylim(-max_range, max_range)
    
    # Get current foot positions for support polygon
    if data.ncon > 0:  # Only show support circles if we have contacts
        foot_positions = get_support_polygon(model, data)
        
        # Draw circles for each foot
        foot_radius = 0.1  # Radius of the foot contact area
        
        # Draw left foot circle
        left_circle = plt.Circle(
            (foot_positions['left'][0], foot_positions['left'][1]),
            foot_radius,
            fill=True,
            alpha=0.2,
            color='blue',
            label='Support Area'
        )
        ax_zmp_xy.add_patch(left_circle)
        
        # Draw right foot circle
        right_circle = plt.Circle(
            (foot_positions['right'][0], foot_positions['right'][1]),
            foot_radius,
            fill=True,
            alpha=0.2,
            color='blue'
        )
        ax_zmp_xy.add_patch(right_circle)
    
    ax_zmp_xy.set_title('ZMP and COM Trajectories (Top View)')
    ax_zmp_xy.set_xlabel('X Position (m)')
    ax_zmp_xy.set_ylabel('Y Position (m)')
    ax_zmp_xy.grid(True)
    ax_zmp_xy.legend()
    ax_zmp_xy.axis('equal')  # Equal aspect ratio
    
    # Plot ZMP and COM positions over time
    ax_zmp_time.plot(plot_data.data['time'], plot_data.data['zmp_x'], 'g-', label='ZMP X', linewidth=2)
    ax_zmp_time.plot(plot_data.data['time'], plot_data.data['zmp_y'], 'g--', label='ZMP Y', linewidth=2)
    ax_zmp_time.plot(plot_data.data['time'], plot_data.data['com_x'], 'r-', label='COM X', linewidth=2)
    ax_zmp_time.plot(plot_data.data['time'], plot_data.data['com_y'], 'r--', label='COM Y', linewidth=2)
    ax_zmp_time.set_title('ZMP and COM Positions vs Time')
    ax_zmp_time.set_xlabel('Time (s)')
    ax_zmp_time.set_ylabel('Position (m)')
    ax_zmp_time.grid(True)
    ax_zmp_time.legend()
    
    # Plot stability margin over time
    if plot_data.data['stability_margin']:
        ax_margin.plot(plot_data.data['time'], plot_data.data['stability_margin'], 'b-', linewidth=2)
        ax_margin.axhline(y=0, color='r', linestyle='--', label='Stability Threshold')
        ax_margin.set_title('Stability Margin vs Time')
        ax_margin.set_xlabel('Time (s)')
        ax_margin.set_ylabel('Margin (m)')
        ax_margin.grid(True)
    
    # Adjust layout and display
    fig_zmp.suptitle('ZMP Analysis', fontsize=16)
    
    plt.tight_layout()
    plt.show()

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
    # Calculate hip angle between left and right thigh vectors (projected onto XZ plane)
    hip_angle = angle_between(left_thigh_vec_proj, right_thigh_vec_proj)

    # Calculate hip abduction/adduction
    hip_x_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_left')
    hip_x_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_right')
    left_ext = data.qpos[model.jnt_qposadr[hip_x_left_id]]
    right_ext = data.qpos[model.jnt_qposadr[hip_x_right_id]]
    abd_angle_rad = left_ext - right_ext
    abd_angle_deg = np.degrees(abd_angle_rad)

    # Estimate ankle positions
    left_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_left')
    right_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_right')
    
    # Calculate ankle positions using shin orientation
    left_shin_dir = data.xmat[left_shin_id].reshape(3, 3) @ np.array([0, 0, -0.35])
    left_ankle_pos = left_knee_pos + left_shin_dir
    right_shin_dir = data.xmat[right_shin_id].reshape(3, 3) @ np.array([0, 0, -0.35])
    right_ankle_pos = right_knee_pos + right_shin_dir

    # Calculate shin vectors (knee to ankle)
    left_shin_vec = left_ankle_pos - left_knee_pos
    right_shin_vec = right_ankle_pos - right_knee_pos

    # Calculate knee angles between thigh and shin vectors
    left_knee_angle = angle_between(left_thigh_vec, left_shin_vec)
    right_knee_angle = angle_between(right_thigh_vec, right_shin_vec)

def compute_overall_com(model, data):
    """
    Compute the overall center of mass of the entire model and its velocity.
    Uses MuJoCo's built-in subtree computations for better efficiency.
    
    Returns:
        tuple containing:
        - 3D numpy array with COM position
        - 3D numpy array with COM velocity
    """
    # Get COM position directly from MuJoCo's subtree computation
    com_pos = data.subtree_com[1].copy()
    
    # Get COM velocity from subtree linear velocity
    com_vel = data.subtree_linvel[1].copy()
    
    return com_pos, com_vel

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

def get_foot_ground_contact_forces(model, data):
    """
    Calculate the ground contact forces for both feet (bottom of shins).
    Returns a dictionary with force components (fx, fy, fz) for each foot.
    """
    forces = {
        'left_foot': np.zeros(3),
        'right_foot': np.zeros(3)
    }
    
    # Get the IDs of the shin geoms and floor
    left_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'shin_left')
    right_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'shin_right')
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    
    # Debug print every 100 frames
    if frame_count % 100 == 0:
        print(f"\nContact Debug [frame {frame_count}]:")
        print(f"Number of contacts: {data.ncon}")
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            print(f"\nContact {i}:")
            print(f"Between: {geom1_name} and {geom2_name}")
            print(f"Position: {contact.pos}")
            
            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, contact_force)
            print(f"Raw force: {contact_force[0:3]}")
    
    # Iterate through all contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Calculate contact force for this contact
        contact_force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, contact_force)
        
        # Get the force components (fx, fy, fz)
        # Note: MuJoCo returns forces in the world frame
        force_components = contact_force[0:3]
        
        # Check which geoms are involved in the contact
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        # Add forces to the appropriate foot if it's in contact with the floor
        # Note: The force direction is from geom1 to geom2, so we need to flip it
        # if the floor is geom2
        if (geom1_id == floor_id and geom2_id == left_shin_id):
            forces['left_foot'] -= force_components  # Flip direction
        elif (geom1_id == left_shin_id and geom2_id == floor_id):
            forces['left_foot'] += force_components
        elif (geom1_id == floor_id and geom2_id == right_shin_id):
            forces['right_foot'] -= force_components  # Flip direction
        elif (geom1_id == right_shin_id and geom2_id == floor_id):
            forces['right_foot'] += force_components
    
    return forces

def calculate_stability_margin(zmp_pos, foot_positions):
    """
    Calculate the minimum distance from ZMP to the nearest foot circle center.
    For hemisphere feet, the stability margin is positive if the ZMP is within
    the foot radius of either foot center, negative otherwise.
    """
    foot_radius = 0.1  # Should match the radius used in plotting
    
    # Calculate distances from ZMP to each foot center
    left_dist = np.linalg.norm(zmp_pos[:2] - foot_positions['left'])
    right_dist = np.linalg.norm(zmp_pos[:2] - foot_positions['right'])
    
    # Get the minimum distance to either foot
    min_dist = min(left_dist, right_dist)
    
    # Return positive margin if ZMP is within foot radius, negative otherwise
    return foot_radius - min_dist

def get_support_polygon(model, data):
    """
    Calculate the support polygon based on foot positions.
    For hemisphere feet, we'll represent each foot as a circle in the XY plane.
    """
    # Get foot positions
    left_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_left')
    right_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_right')
    
    # Calculate foot positions using shin orientation
    left_shin_pos = data.xpos[left_shin_id]
    right_shin_pos = data.xpos[right_shin_id]
    
    left_shin_dir = data.xmat[left_shin_id].reshape(3, 3) @ np.array([0, 0, -0.35])
    right_shin_dir = data.xmat[right_shin_id].reshape(3, 3) @ np.array([0, 0, -0.35])
    
    left_foot_pos = left_shin_pos + left_shin_dir
    right_foot_pos = right_shin_pos + right_shin_dir
    
    # Only return the x,y coordinates
    return {'left': left_foot_pos[:2], 'right': right_foot_pos[:2]}

# === Main Simulation Loop ===
# Define simulation parameters before the viewer launch
fps = TARGET_FPS
timestep = model.opt.timestep
steps_per_frame = int(1.0 / (fps * timestep))
SIMULATION_SPEED = 6.0  # 1.0 is real-time, higher values are slower
MAX_SIM_TIME = 10.0     # Limit simulation to 6 seconds

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set camera to a zoomed out position for better visibility
    viewer.cam.distance = 10.0  # Distance from camera to target
    viewer.cam.azimuth = 90     # Camera rotation around the z-axis in degrees
    viewer.cam.elevation = -20  # Camera elevation in degrees

    # For on/off control, set initial target angles
    if not USE_RAMP:
        # Set initial target angles for right leg
        data.ctrl[right_hip_actuator_id] = apply_pi_control(get_single_joint_angle(model, data, 'hip_y_right'), RIGHT_HIP_TARGET_ANGLE, 'hip_y_right', timestep)
        data.ctrl[right_knee_actuator_id] = apply_pi_control(get_single_joint_angle(model, data, 'knee_right'), RIGHT_KNEE_TARGET_ANGLE, 'knee_right', timestep)
        
        # Set initial target angles for left leg (all 0)
        data.ctrl[left_hip_actuator_id] = apply_pi_control(get_single_joint_angle(model, data, 'hip_y_left'), 0, 'hip_y_left', timestep)
        data.ctrl[left_knee_actuator_id] = apply_pi_control(get_single_joint_angle(model, data, 'knee_left'), 0, 'knee_left', timestep)
        
        # Set lean angles (both 0)
        data.ctrl[hip_x_left_actuator_id] = apply_pi_control(get_single_joint_angle(model, data, 'hip_x_left'), 0, 'hip_x_left', timestep)
        data.ctrl[hip_x_right_actuator_id] = apply_pi_control(get_single_joint_angle(model, data, 'hip_x_right'), 0, 'hip_x_right', timestep)

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

    # --- Screenshot logic ---
    screenshot_times = []
    # Evenly distribute 10 screenshots across the simulation duration
    for i in range(10):
        screenshot_times.append(i * (MAX_SIM_TIME / 9))  # Divide by 9 to get 10 points (0 to MAX_SIM_TIME)
    screenshot_taken = [False] * len(screenshot_times)
    screenshot_paths = [os.path.join(current_dir, f"baymax_shot_{i+1}.png") for i in range(len(screenshot_times))]

    # Add variable to track minimum right foot height
    min_right_foot_height = float('inf')
    frame_count = 0 # Initialize frame counter for plotting
    
    # Reset data collector at start of simulation
    plot_data.reset()
    start_time = time.time()
    
    # Let the simulation stabilize for a few steps before taking initial height
    stabilization_steps = 100
    for i in range(stabilization_steps):  # Run 100 steps to let the robot settle
        mujoco.mj_step(model, data)
    initial_torso_height = data.xpos[torso_id][2]  # Store initial torso height (z-coordinate)

    # Initialize movement flag to ensure joints are only moved once
    moved = False
    angle_ramp_started = False

    # Initialize leg switching variables
    # leg_switch_triggered = False
    # leg_switch_start_time = None
    # leg_switch_completed = False
    
    # Previous momenta for ZMP calculation
    prev_lin_mom = None
    prev_ang_mom = None
    
    # Initialize simulation time
    sim_time = 0.0
    
    while viewer.is_running():
        current_time = sim_time  # Use simulation time instead of wall clock time
        
        # Set hip X (side-to-side) rotations based on LEAN_ANGLE
        if LEAN_ANGLE != 0:
            data.ctrl[hip_x_left_actuator_id] = LEAN_ANGLE
            data.ctrl[hip_x_right_actuator_id] = -LEAN_ANGLE
        else:
            data.ctrl[hip_x_left_actuator_id] = 0
            data.ctrl[hip_x_right_actuator_id] = 0

        # Calculate ZMP and stability data
        com_pos, com_vel = compute_overall_com(model, data)
        x_zmp, y_zmp, lin_mom, ang_mom = calculate_zmp(model, data, prev_lin_mom, prev_ang_mom)
        zmp_pos = np.array([x_zmp, y_zmp, 0.001])
        
        # Update momenta for next iteration
        prev_lin_mom = lin_mom
        prev_ang_mom = ang_mom

        # Collect all data for plotting (every frame for more detailed data)
        actuator_ids = {
            'right_hip': right_hip_actuator_id,
            'right_hip_x': hip_x_right_actuator_id,
            'right_knee': right_knee_actuator_id,
            'left_hip': left_hip_actuator_id,
            'left_hip_x': hip_x_left_actuator_id,
            'left_knee': left_knee_actuator_id
        }
        
        # Calculate raw target angles based on current time
        raw_targets = {'right_hip': 0.0, 'right_knee': 0.0, 'left_hip': 0.0, 'left_knee': 0.0}
        
        # Calculate target angles based on current phase of motion
        if current_time < LEAN_RAMP_END_TIME:
            # During lean phase, all joints target 0
            pass
        elif current_time >= ramp_start_time and current_time < max(ramp_end_time_right_hip, ramp_end_time_left_hip, ramp_end_time_right_knee, ramp_end_time_left_knee):
            # During ramp phase
            # Right hip
            if current_time < ramp_end_time_right_hip:
                alpha_right_hip = (current_time - ramp_start_time) / RIGHT_HIP_RAMP_DURATION
                alpha_right_hip = min(max(alpha_right_hip, 0.0), 1.0)
                raw_targets['right_hip'] = alpha_right_hip * RIGHT_HIP_TARGET_ANGLE
            else:
                raw_targets['right_hip'] = RIGHT_HIP_TARGET_ANGLE
            
            # Left hip
            if current_time < ramp_end_time_left_hip:
                alpha_left_hip = (current_time - ramp_start_time) / LEFT_HIP_RAMP_DURATION
                alpha_left_hip = min(max(alpha_left_hip, 0.0), 1.0)
                raw_targets['left_hip'] = alpha_left_hip * LEFT_HIP_TARGET_ANGLE
            else:
                raw_targets['left_hip'] = LEFT_HIP_TARGET_ANGLE
            
            # Right knee (after delay)
            if current_time >= (ramp_start_time + RIGHT_KNEE_DELAY):
                if current_time < ramp_end_time_right_knee:
                    alpha_right_knee = (current_time - (ramp_start_time + RIGHT_KNEE_DELAY)) / RIGHT_KNEE_RAMP_DURATION
                    alpha_right_knee = min(max(alpha_right_knee, 0.0), 1.0)
                    raw_targets['right_knee'] = alpha_right_knee * RIGHT_KNEE_TARGET_ANGLE
                else:
                    raw_targets['right_knee'] = RIGHT_KNEE_TARGET_ANGLE
            
            # Left knee (after delay)
            if current_time >= (ramp_start_time + LEFT_KNEE_DELAY):
                if current_time < ramp_end_time_left_knee:
                    alpha_left_knee = (current_time - (ramp_start_time + LEFT_KNEE_DELAY)) / LEFT_KNEE_RAMP_DURATION
                    alpha_left_knee = min(max(alpha_left_knee, 0.0), 1.0)
                    raw_targets['left_knee'] = alpha_left_knee * LEFT_KNEE_TARGET_ANGLE
                else:
                    raw_targets['left_knee'] = LEFT_KNEE_TARGET_ANGLE
        else:
            # After all ramps complete
            raw_targets['right_hip'] = RIGHT_HIP_TARGET_ANGLE
            raw_targets['right_knee'] = RIGHT_KNEE_TARGET_ANGLE
            raw_targets['left_hip'] = LEFT_HIP_TARGET_ANGLE
            raw_targets['left_knee'] = LEFT_KNEE_TARGET_ANGLE
        
        # Get support polygon and calculate stability margin
        support_polygon = get_support_polygon(model, data)
        stability_margin = calculate_stability_margin(zmp_pos, support_polygon)
        
        # Add all data points
        plot_data.add_datapoint(
            current_time, 
            model, 
            data, 
            actuator_ids,
            raw_targets=raw_targets,
            zmp_pos=zmp_pos,
            com_pos=com_pos,
            stability_margin=stability_margin
        )
        
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
            
            # Calculate and print final steady-state errors for all joints
            final_errors = {
                'R_Hip_Y': plot_data.data['hip_y_target_right'][-1] - plot_data.data['hip_y_actual_right'][-1],
                'R_Hip_X': plot_data.data['hip_x_target_right'][-1] - plot_data.data['hip_x_actual_right'][-1],
                'R_Knee': plot_data.data['knee_target_right'][-1] - plot_data.data['knee_actual_right'][-1],
                'L_Hip_Y': plot_data.data['hip_y_target_left'][-1] - plot_data.data['hip_y_actual_left'][-1],
                'L_Hip_X': plot_data.data['hip_x_target_left'][-1] - plot_data.data['hip_x_actual_left'][-1],
                'L_Knee': plot_data.data['knee_target_left'][-1] - plot_data.data['knee_actual_left'][-1]
            }
            print(f"\nMinimum right foot height during simulation: {min_right_foot_height:.4f} m")
            print(f"Final steady-state errors (deg) - R_Hip_Y: {final_errors['R_Hip_Y']:.2f}, R_Hip_X: {final_errors['R_Hip_X']:.2f}, R_Knee: {final_errors['R_Knee']:.2f}, L_Hip_Y: {final_errors['L_Hip_Y']:.2f}, L_Hip_X: {final_errors['L_Hip_X']:.2f}, L_Knee: {final_errors['L_Knee']:.2f}")
            
            break  # Exit after MAX_SIM_TIME seconds
        # Get current angles for all joints
        current_hip_y_right = get_single_joint_angle(model, data, 'hip_y_right')
        current_hip_y_left = get_single_joint_angle(model, data, 'hip_y_left')
        current_knee_right = get_single_joint_angle(model, data, 'knee_right')
        current_knee_left = get_single_joint_angle(model, data, 'knee_left')
        current_hip_x_right = get_single_joint_angle(model, data, 'hip_x_right')
        current_hip_x_left = get_single_joint_angle(model, data, 'hip_x_left')

        # Data collection is now handled by the DataCollector class in add_datapoint()
        
        if USE_RAMP:
            # --- Lean and Z rotation ramp first ---
            if current_time < LEAN_RAMP_END_TIME:
                # Only ramp lean if LEAN_ANGLE is not 0
                if LEAN_ANGLE != 0:
                    alpha = (current_time - LEAN_RAMP_START_TIME) / LEAN_RAMP_DURATION
                    alpha = min(max(alpha, 0.0), 1.0)
                    target_lean_left = alpha * LEAN_ANGLE
                    target_lean_right = -alpha * LEAN_ANGLE
                    
                    # Apply PI control for lean
                    data.ctrl[hip_x_left_actuator_id] = apply_pi_control(current_hip_x_left, target_lean_left, 'hip_x_left', timestep)
                    data.ctrl[hip_x_right_actuator_id] = apply_pi_control(current_hip_x_right, target_lean_right, 'hip_x_right', timestep)
                else:
                    # If LEAN_ANGLE is 0, actively hold at 0
                    data.ctrl[hip_x_left_actuator_id] = apply_pi_control(current_hip_x_left, 0, 'hip_x_left', timestep)
                    data.ctrl[hip_x_right_actuator_id] = apply_pi_control(current_hip_x_right, 0, 'hip_x_right', timestep)

                # Always actively control all joints to their initial positions
                data.ctrl[right_hip_actuator_id] = apply_pi_control(current_hip_y_right, 0, 'hip_y_right', timestep)
                data.ctrl[left_hip_actuator_id] = apply_pi_control(current_hip_y_left, 0, 'hip_y_left', timestep)
                data.ctrl[right_knee_actuator_id] = apply_pi_control(current_knee_right, 0, 'knee_right', timestep)
                data.ctrl[left_knee_actuator_id] = apply_pi_control(current_knee_left, 0, 'knee_left', timestep)

            # --- Hip/knee ramps after lean is complete ---
            elif (current_time >= ramp_start_time and current_time < max(ramp_end_time_right_hip, ramp_end_time_left_hip, ramp_end_time_right_knee, ramp_end_time_left_knee)):
                
                # Don't start knee movement until after delay
                knee_movement_started_right = current_time >= (ramp_start_time + RIGHT_KNEE_DELAY)
                knee_movement_started_left = current_time >= (ramp_start_time + LEFT_KNEE_DELAY)

                # Right hip - only ramp if target is not 0
                if RIGHT_HIP_TARGET_ANGLE != 0:
                    if current_time < ramp_end_time_right_hip:
                        alpha_right_hip = (current_time - ramp_start_time) / RIGHT_HIP_RAMP_DURATION
                        alpha_right_hip = min(max(alpha_right_hip, 0.0), 1.0)
                        target_angle = alpha_right_hip * RIGHT_HIP_TARGET_ANGLE
                    else:
                        target_angle = RIGHT_HIP_TARGET_ANGLE
                    data.ctrl[right_hip_actuator_id] = apply_pi_control(current_hip_y_right, target_angle, 'hip_y_right', timestep)
                else:
                    data.ctrl[right_hip_actuator_id] = apply_pi_control(current_hip_y_right, 0, 'hip_y_right', timestep)

                # Left hip - only ramp if target is not 0
                if LEFT_HIP_TARGET_ANGLE != 0:
                    if current_time < ramp_end_time_left_hip:
                        alpha_left_hip = (current_time - ramp_start_time) / LEFT_HIP_RAMP_DURATION
                        alpha_left_hip = min(max(alpha_left_hip, 0.0), 1.0)
                        target_angle = alpha_left_hip * LEFT_HIP_TARGET_ANGLE
                    else:
                        target_angle = LEFT_HIP_TARGET_ANGLE
                    data.ctrl[left_hip_actuator_id] = apply_pi_control(current_hip_y_left, target_angle, 'hip_y_left', timestep)
                else:
                    data.ctrl[left_hip_actuator_id] = apply_pi_control(current_hip_y_left, 0, 'hip_y_left', timestep)

                # Right knee - actively control to 0 during delay, then ramp to target
                if knee_movement_started_right:
                    if current_time < ramp_end_time_right_knee:
                        # Calculate alpha based on time since knee movement started
                        alpha_right_knee = (current_time - (ramp_start_time + RIGHT_KNEE_DELAY)) / RIGHT_KNEE_RAMP_DURATION
                        alpha_right_knee = min(max(alpha_right_knee, 0.0), 1.0)
                        target_angle = alpha_right_knee * RIGHT_KNEE_TARGET_ANGLE
                    else:
                        target_angle = RIGHT_KNEE_TARGET_ANGLE
                else:
                    # Actively hold at 0 during delay period
                    target_angle = 0
                data.ctrl[right_knee_actuator_id] = apply_pi_control(current_knee_right, target_angle, 'knee_right', timestep)

                # Left knee - only ramp if target is not 0 and after delay
                if LEFT_KNEE_TARGET_ANGLE != 0 and knee_movement_started_left:
                    if current_time < ramp_end_time_left_knee:
                        # Calculate alpha based on time since knee movement started
                        alpha_left_knee = (current_time - (ramp_start_time + LEFT_KNEE_DELAY)) / LEFT_KNEE_RAMP_DURATION
                        alpha_left_knee = min(max(alpha_left_knee, 0.0), 1.0)
                        target_angle = alpha_left_knee * LEFT_KNEE_TARGET_ANGLE
                    else:
                        target_angle = LEFT_KNEE_TARGET_ANGLE
                    data.ctrl[left_knee_actuator_id] = apply_pi_control(current_knee_left, target_angle, 'knee_left', timestep)
                else:
                    data.ctrl[left_knee_actuator_id] = apply_pi_control(current_knee_left, 0, 'knee_left', timestep)

                # Always maintain lean at target angles
                if LEAN_ANGLE != 0:
                    data.ctrl[hip_x_left_actuator_id] = apply_pi_control(current_hip_x_left, LEAN_ANGLE, 'hip_x_left', timestep)
                    data.ctrl[hip_x_right_actuator_id] = apply_pi_control(current_hip_x_right, -LEAN_ANGLE, 'hip_x_right', timestep)
                else:
                    data.ctrl[hip_x_left_actuator_id] = apply_pi_control(current_hip_x_left, 0, 'hip_x_left', timestep)
                    data.ctrl[hip_x_right_actuator_id] = apply_pi_control(current_hip_x_right, 0, 'hip_x_right', timestep)

            # Once all ramps are complete, maintain all joints at their final target angles
            else:
                # Get current angles
                current_hip_y_right = get_single_joint_angle(model, data, 'hip_y_right')
                current_hip_y_left = get_single_joint_angle(model, data, 'hip_y_left')
                current_knee_right = get_single_joint_angle(model, data, 'knee_right')
                current_knee_left = get_single_joint_angle(model, data, 'knee_left')
                current_hip_x_right = get_single_joint_angle(model, data, 'hip_x_right')
                current_hip_x_left = get_single_joint_angle(model, data, 'hip_x_left')
                
                # Apply PI control to all joints
                data.ctrl[right_knee_actuator_id] = apply_pi_control(current_knee_right, RIGHT_KNEE_TARGET_ANGLE, 'knee_right', timestep)
                data.ctrl[right_hip_actuator_id] = apply_pi_control(current_hip_y_right, RIGHT_HIP_TARGET_ANGLE, 'hip_y_right', timestep)
                data.ctrl[left_hip_actuator_id] = apply_pi_control(current_hip_y_left, LEFT_HIP_TARGET_ANGLE, 'hip_y_left', timestep)
                data.ctrl[left_knee_actuator_id] = apply_pi_control(current_knee_left, LEFT_KNEE_TARGET_ANGLE, 'knee_left', timestep)
                
                # Set lean angles
                if LEAN_ANGLE != 0:
                    data.ctrl[hip_x_left_actuator_id] = apply_pi_control(current_hip_x_left, LEAN_ANGLE, 'hip_x_left', timestep)
                    data.ctrl[hip_x_right_actuator_id] = apply_pi_control(current_hip_x_right, -LEAN_ANGLE, 'hip_x_right', timestep)
                else:
                    data.ctrl[hip_x_left_actuator_id] = apply_pi_control(current_hip_x_left, 0, 'hip_x_left', timestep)
                    data.ctrl[hip_x_right_actuator_id] = apply_pi_control(current_hip_x_right, 0, 'hip_x_right', timestep)
        else:
            # On/off control - maintain the initial target angles throughout the simulation
            data.ctrl[right_hip_actuator_id] = apply_pi_control(current_hip_y_right, RIGHT_HIP_TARGET_ANGLE, 'hip_y_right', timestep)
            data.ctrl[right_knee_actuator_id] = apply_pi_control(current_knee_right, RIGHT_KNEE_TARGET_ANGLE, 'knee_right', timestep)
            data.ctrl[left_hip_actuator_id] = apply_pi_control(current_hip_y_left, 0, 'hip_y_left', timestep)
            data.ctrl[left_knee_actuator_id] = apply_pi_control(current_knee_left, 0, 'knee_left', timestep)
            data.ctrl[hip_x_left_actuator_id] = apply_pi_control(current_hip_x_left, 0, 'hip_x_left', timestep)
            data.ctrl[hip_x_right_actuator_id] = apply_pi_control(current_hip_x_right, 0, 'hip_x_right', timestep)
        
        # Run fixed number of steps for smooth video
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
            sim_time += timestep  # Increment simulation time by the physics timestep
            
        # Update visualization
        if viewer is not None and viewer.is_running():
            # Calculate overall COM position and velocity
            com_pos, com_vel = compute_overall_com(model, data)
            
            # Update COM visualization site
            if overall_com_site_id != -1:
                model.site_pos[overall_com_site_id] = com_pos
            
            # Calculate and visualize ZMP if the site exists
            if zmp_site_id != -1:
                x_zmp, y_zmp, lin_mom, ang_mom = calculate_zmp(model, data, prev_lin_mom, prev_ang_mom)
                model.site_pos[zmp_site_id] = np.array([x_zmp, y_zmp, 0.001])  # Slightly above ground
                
                # Store momenta for next iteration
                prev_lin_mom = lin_mom
                prev_ang_mom = ang_mom
            
            # Update the viewer
            viewer.sync()
            
            # Record video frame if enabled
            if RECORD_VIDEO:
                renderer.update_scene(data, camera=viewer.cam)
                pixels = renderer.render()
                out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                frames_written += 1
        
        # Track foot heights
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
            
            # print(f"Foot heights - Left: {left_foot_pos[2]:.3f}m, Right: {right_foot_pos[2]:.3f}m")
            # print(f"Leg switch status - Triggered: {False}, Completed: {False}")
        
        # Control simulation rate (slower for easier observation)
        time.sleep(0.01 * SIMULATION_SPEED)
        frame_count += 1 # Increment frame counter

    # Clean up video recording if enabled
    if RECORD_VIDEO:
        print("\nFinalizing video recording...")
        out.release()
        # Wait a moment to ensure file is written
        time.sleep(1)
        
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            if file_size > 0:
                print(f"Video saved successfully to {video_path}")
                print(f"Video file size: {file_size / (1024*1024):.1f} MB")
                print(f"Frames written: {frames_written}")
            else:
                print(f"Error: Video file {video_path} was created but is empty")
                sys.exit(1)
        else:
            print(f"Error: Video file {video_path} was not created")
            sys.exit(1)
            
    # Plot the collected data after video is saved
    print(f"\nVideo saved successfully. Now generating plots...")
    try:
        plot_joint_angles()
    except Exception as e:
        print(f"Error in plotting function: {e}")
        print("Attempting to show plots anyway...")
        plt.show()