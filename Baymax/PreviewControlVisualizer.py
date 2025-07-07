import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.linalg import solve_discrete_are
from numpy.linalg import inv
import cv2

# Import preview control functions
from PreviewControl import (
    create_system,
    create_controller,
    create_step_pattern,
    calculate_controller,
    calculate_state
)

# === Simulation Parameters ===
dt = 5.e-3  # Time step (must match the one in XML)
step_duration = 0.5  # Duration of each step
num_steps = 5  # Number of steps to take
preview_window = int(1.6/dt)  # Preview window for the controller (matches create_controller)
time_end = 4.0  # Total simulation time in seconds
TARGET_FPS = 60  # For smooth visualization
SIMULATION_SPEED = 1.0  # 1.0 is real-time, higher values are slower

# === Controller Parameters ===
g = 9.81  # Gravity
# Set desired constant CoM height for the walking controller
Zc = 0.8  # Fixed CoM height for walking

# === Step Pattern Parameters ===
STEP_LENGTH = -0.2  # Step length in X direction (increased from -0.3 to -0.4)
STEP_WIDTH = 0.08  # Step width in Y direction


def leg_inverse_kinematics(x, z, debug=True):
    """
    Calculate leg joint angles from relative x and z positions.
    x: forward/backward distance from hip to target
    z: vertical distance from hip to target (negative goes down)
    Returns hip pitch and knee angles in degrees
    """
    # Robot dimensions from Baymax.xml
    l1 = 0.403  # Thigh length (hip to knee distance)
    l2 = 0.6   # Shin length (knee to ankle distance)
    
    # First calculate theta2 (knee angle)
    r_squared = x*x + z*z
    cos_theta2 = (r_squared - l1*l1 - l2*l2) / (2 * l1 * l2)
    
    # Check if the point is reachable
    if cos_theta2 > 1:
        if debug:
            print(f"Warning: Point ({x}, {z}) is too far to reach!")
        cos_theta2 = 1
    elif cos_theta2 < -1:
        if debug:
            print(f"Warning: Point ({x}, {z}) requires too much knee bend!")
        cos_theta2 = -1
    
    # Calculate knee angle (theta2)
    sin_theta2 = -np.sqrt(1 - cos_theta2*cos_theta2)  # Negative for knee bending backwards
    theta2 = np.arctan2(sin_theta2, cos_theta2)
    
    # Calculate hip angle (theta1)
    k1 = l1 + l2 * cos_theta2
    k2 = l2 * sin_theta2
    
    # Use atan2 to get the right quadrant
    gamma = np.arctan2(k2, k1)
    theta1 = np.arctan2(x, -z) - gamma
    
    if debug:
        print(f"Target (x,z): ({x:.3f}, {z:.3f}) -> Hip: {np.degrees(theta1):.1f}°, Knee: {np.degrees(theta2):.1f}°")
    
    return 0.0, np.degrees(theta1), np.degrees(theta2)  # Return 0 for yaw since we're ignoring it

class DataCollector:
    def __init__(self, Zc):
        self.Zc = Zc
        self.reset()

    def reset(self):
        """Initialize/reset data storage"""
        self.data = {
            'time': [], 
            'com_x': [], 'com_y': [], 'com_z': [],
            'zmp_x': [], 'zmp_y': [],
            'preview_com_x': [], 'preview_com_y': [],
            'preview_zmp_x': [], 'preview_zmp_y': [],
            'ref_zmp_x': [], 'ref_zmp_y': [],
            'rel_pos_x_left': [], 'rel_pos_x_right': [],
            'rel_pos_y_left': [], 'rel_pos_y_right': [],
            # Target angles
            'target_left_hip_yaw': [], 'target_left_hip_pitch': [],
            'target_right_hip_yaw': [], 'target_right_hip_pitch': [],
            'target_left_knee': [], 'target_right_knee': [],
            # Actual angles
            'actual_left_hip_yaw': [], 'actual_left_hip_pitch': [],
            'actual_right_hip_yaw': [], 'actual_right_hip_pitch': [],
            'actual_left_knee': [], 'actual_right_knee': [],
            # Controller error integrator
            'error_integrator_x': [], 'error_integrator_y': []  # Error integrator from preview controller
        }

class StepTracker:
    def __init__(self):
        self.right_leg_stepping = True  # Start with right leg stepping
        self.step_start_pos = {'left': 0.0, 'right': 0.0}  # Starting position for each leg's step
        self.cumulative_distance = {'left': 0.0, 'right': 0.0}  # Track progress of each leg

def calculate_relative_hip_positions_single(current_hip_x_left, current_hip_x_right, current_ref_zmp_x, current_ref_zmp_y, step_tracker):
    """
    Calculate the relative positions of the hips with respect to the reference ZMP,
    tracking cumulative step progress for each leg during walking.
    """
    # Update step progress for the currently stepping leg
    rel_pos_x_left = 0.0
    rel_pos_x_right = 0.0
    
    if step_tracker.right_leg_stepping:
        # Track right leg progress
        if step_tracker.step_start_pos['right'] == 0.0:
            step_tracker.step_start_pos['right'] = current_hip_x_right
        
        # Calculate cumulative distance for right leg (keep negative values)
        step_tracker.cumulative_distance['right'] = current_hip_x_right - step_tracker.step_start_pos['right']
        
        # Show progress for right leg
        rel_pos_x_right = step_tracker.cumulative_distance['right']
        rel_pos_x_left = 0.0
        
        # Check if right leg step is complete (compare with negative STEP_LENGTH)
        if step_tracker.cumulative_distance['right'] <= STEP_LENGTH:
            step_tracker.right_leg_stepping = False
            step_tracker.step_start_pos['left'] = current_hip_x_left
            step_tracker.cumulative_distance['right'] = 0.0
            
    else:
        # Track left leg progress
        if step_tracker.step_start_pos['left'] == 0.0:
            step_tracker.step_start_pos['left'] = current_hip_x_left
        
        # Calculate cumulative distance for left leg (keep negative values)
        step_tracker.cumulative_distance['left'] = current_hip_x_left - step_tracker.step_start_pos['left']
        
        # Show progress for left leg
        rel_pos_x_left = step_tracker.cumulative_distance['left']
        rel_pos_x_right = 0.0
        
        # Check if left leg step is complete (compare with negative STEP_LENGTH)
        if step_tracker.cumulative_distance['left'] <= STEP_LENGTH:
            step_tracker.right_leg_stepping = True
            step_tracker.step_start_pos['right'] = current_hip_x_right
            step_tracker.cumulative_distance['left'] = 0.0
    
    return rel_pos_x_left, rel_pos_x_right

def calculate_lateral_distances_single(current_zmp_y, current_com_y):
    """
    Calculate the lateral distances between hip positions and ZMP.
    
    Args:
        current_zmp_y: Current Y position of the Zero Moment Point
        current_com_y: Current Y position of the Center of Mass
    
    Returns:
        tuple (rel_pos_y_left, rel_pos_y_right): Lateral distances for left and right hips
    """
    # Calculate hip positions relative to COM
    # Hips are offset from COM by STEP_WIDTH/2
    left_hip_y = current_com_y + STEP_WIDTH/2
    right_hip_y = current_com_y - STEP_WIDTH/2
    
    # Calculate raw distances from ZMP to each hip
    left_distance = left_hip_y - current_zmp_y
    right_distance = right_hip_y - current_zmp_y
    
    # Use ZMP Y position to determine swing phase
    # When ZMP Y is negative or zero, right leg is swinging (left is support)
    # When ZMP Y is positive, left leg is swinging (right is support)
    if current_zmp_y <= 0:
        # Right leg is in swing
        rel_pos_y_right = right_distance
        rel_pos_y_left = 0.0
    else:
        # Left leg is in swing
        rel_pos_y_left = left_distance
        rel_pos_y_right = 0.0
    
    return rel_pos_y_left, rel_pos_y_right

def plot_results(plot_data, Zc):
    """Generate plots of the simulation results"""
    # Create figure with 2x3 subplots
    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Walking Control Variables', fontsize=16)

    # Plot ZMP and CoM trajectories
    ax1.plot(plot_data['time'], plot_data['zmp_x'], 'r-', label='Actual ZMP X', linewidth=2)
    ax1.plot(plot_data['time'], plot_data['com_x'], 'b-', label='Actual CoM X', linewidth=2)
    ax1.plot(plot_data['time'], plot_data['preview_zmp_x'], 'r--', label='Preview ZMP X', linewidth=1)
    ax1.plot(plot_data['time'], plot_data['preview_com_x'], 'b--', label='Preview CoM X', linewidth=1)
    ax1.plot(plot_data['time'], plot_data['ref_zmp_x'], 'k:', label='Ref ZMP X', linewidth=2)
    ax1.plot(plot_data['time'], plot_data['rel_pos_x_left'], 'm-', label='Left Hip Step Progress', linewidth=1)
    ax1.plot(plot_data['time'], plot_data['rel_pos_x_right'], 'c-', label='Right Hip Step Progress', linewidth=1)
    ax1.set_title('X-axis Motion')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.grid(True)
    ax1.legend()

    # Plot Y-axis motion
    ax2.plot(plot_data['time'], plot_data['zmp_y'], 'r-', label='Actual ZMP Y', linewidth=2)
    ax2.plot(plot_data['time'], plot_data['com_y'], 'b-', label='Actual CoM Y', linewidth=2)
    ax2.plot(plot_data['time'], plot_data['preview_zmp_y'], 'r--', label='Preview ZMP Y', linewidth=1)
    ax2.plot(plot_data['time'], plot_data['preview_com_y'], 'b--', label='Preview CoM Y', linewidth=1)
    ax2.plot(plot_data['time'], plot_data['ref_zmp_y'], 'k:', label='Ref ZMP Y', linewidth=2)
    ax2.plot(plot_data['time'], plot_data['rel_pos_y_left'], 'm-', label='Left Hip-ZMP Lateral Distance', linewidth=1)
    ax2.plot(plot_data['time'], plot_data['rel_pos_y_right'], 'c-', label='Right Hip-ZMP Lateral Distance', linewidth=1)
    ax2.set_title('Y-axis Motion')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.grid(True)
    ax2.legend()

    # Plot hip angles - both target and actual
    ax3.plot(plot_data['time'], plot_data['target_left_hip_yaw'], 'r-', label='Target Left Hip X', linewidth=2)
    ax3.plot(plot_data['time'], plot_data['actual_left_hip_yaw'], 'r--', label='Actual Left Hip X', linewidth=1)
    ax3.plot(plot_data['time'], plot_data['target_right_hip_yaw'], 'b-', label='Target Right Hip X', linewidth=2)
    ax3.plot(plot_data['time'], plot_data['actual_right_hip_yaw'], 'b--', label='Actual Right Hip X', linewidth=1)
    ax3.plot(plot_data['time'], plot_data['target_left_hip_pitch'], 'g-', label='Target Left Hip Y', linewidth=2)
    ax3.plot(plot_data['time'], plot_data['actual_left_hip_pitch'], 'g--', label='Actual Left Hip Y', linewidth=1)
    ax3.plot(plot_data['time'], plot_data['target_right_hip_pitch'], 'm-', label='Target Right Hip Y', linewidth=2)
    ax3.plot(plot_data['time'], plot_data['actual_right_hip_pitch'], 'm--', label='Actual Right Hip Y', linewidth=1)
    ax3.set_title('Hip Angles')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (degrees)')
    ax3.grid(True)
    ax3.legend()
    
    # Plot knee angles - both target and actual
    ax4.plot(plot_data['time'], plot_data['target_left_knee'], 'r-', label='Target Left Knee', linewidth=2)
    ax4.plot(plot_data['time'], plot_data['actual_left_knee'], 'r--', label='Actual Left Knee', linewidth=1)
    ax4.plot(plot_data['time'], plot_data['target_right_knee'], 'b-', label='Target Right Knee', linewidth=2)
    ax4.plot(plot_data['time'], plot_data['actual_right_knee'], 'b--', label='Actual Right Knee', linewidth=1)
    ax4.set_title('Knee Angles')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle (degrees)')
    ax4.grid(True)
    ax4.legend()

    # Plot error integrator
    ax5.plot(plot_data['time'], plot_data['error_integrator_x'], 'r-', label='X Error Integrator', linewidth=2)
    ax5.plot(plot_data['time'], plot_data['error_integrator_y'], 'b-', label='Y Error Integrator', linewidth=2)
    ax5.set_title('Preview Controller Error Integrator')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Accumulated Error')
    ax5.grid(True)
    ax5.legend()

    # Plot error integrator phase plot
    ax6.plot(plot_data['error_integrator_x'], plot_data['error_integrator_y'], 'k-', linewidth=1)
    ax6.plot(plot_data['error_integrator_x'][0], plot_data['error_integrator_y'][0], 'go', label='Start')
    ax6.plot(plot_data['error_integrator_x'][-1], plot_data['error_integrator_y'][-1], 'ro', label='End')
    ax6.set_title('Error Integrator Phase Plot')
    ax6.set_xlabel('X Error')
    ax6.set_ylabel('Y Error')
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()
    plt.show()

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

    return zmp_x, zmp_y, lin_mom, ang_mom

def main():
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "Baymax.xml")

    # Load the model and create data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Reset the simulation and print initial state
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)  # Update all derived quantities
    
    
    # Print debug info about CoM
    com_pos = data.subtree_com[1].copy()  # Index 1 is the root body's subtree (entire robot)

    # Calculate total mass (needed for multiple calculations)
    total_mass = mujoco.mj_getTotalmass(model)
    Gv = 9.80665  # Gravitational constant
    Mg = total_mass * Gv

    # Debug printing for CoM calculation
    print("\nCenter of Mass (CoM) Information:")
    print(f"Number of bodies: {model.nbody}")
    print("\nFull subtree_com array:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"Body {i} ({body_name}):")
        print(f"  CoM position: x={data.subtree_com[i][0]:.3f}, y={data.subtree_com[i][1]:.3f}, z={data.subtree_com[i][2]:.3f}")
    
    print(f"\nUsing root body CoM height: {Zc:.3f}m")
    
    print("\nInitial robot state:")
    # Print robot state information
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    print("\nRobot State:")
    print(f"Torso position (x, y, z): {data.xpos[torso_id]}")
    print(f"Torso orientation (quaternion): {data.xquat[torso_id]}")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_angle = data.qpos[model.jnt_qposadr[i]]  # Use jnt_qposadr for correct indexing
            print(f"Joint {name}: {np.degrees(joint_angle):.2f} degrees")

    # Initialize preview controller with actual CoM height
    (A, B, C, X, U, P) = create_system(dt, Zc, g, total_mass)
    (Gi, Gx, G) = create_controller(A, B, C, 1.0, 1.e-6, 10, preview_window)
    
    # Calculate total simulation steps
    steps = int(time_end / dt)
    
    # Create reference ZMP trajectory
    (pxref, pyref) = create_step_pattern(STEP_LENGTH, STEP_WIDTH, num_steps, 
                                       step_duration, dt, time_end)
    
    # Ensure we have enough reference points for the entire simulation plus preview window
    needed_length = steps + preview_window
    while len(pxref) < needed_length:
        pxref.append(pxref[-1])
        pyref.append(pyref[-1])
    
    # Convert to numpy array
    pref = np.array([pxref, pyref])

    # Initialize data storage dictionary
    plot_data = {
        'time': [], 'com_x': [], 'com_y': [], 'com_z': [],
        'zmp_x': [], 'zmp_y': [],
        'preview_com_x': [], 'preview_com_y': [],
        'preview_zmp_x': [], 'preview_zmp_y': [],
        'ref_zmp_x': [], 'ref_zmp_y': [],
        'hip_angles_left': [], 'hip_angles_right': [],
        'knee_angles_left': [], 'knee_angles_right': [],
        # Add hip position tracking
        'left_hip_x': [], 'left_hip_y': [], 'left_hip_z': [],
        'right_hip_x': [], 'right_hip_y': [], 'right_hip_z': [],
        # Add target angle tracking
        'target_left_hip_yaw': [], 'target_left_hip_pitch': [],
        'target_right_hip_yaw': [], 'target_right_hip_pitch': [],
        'target_left_knee': [], 'target_right_knee': [],
        'rel_pos_x_left': [], 'rel_pos_x_right': [],
        'rel_pos_y_left': [], 'rel_pos_y_right': [],
        # Add actual angle tracking
        'actual_left_hip_yaw': [], 'actual_left_hip_pitch': [],
        'actual_right_hip_yaw': [], 'actual_right_hip_pitch': [],
        'actual_left_knee': [], 'actual_right_knee': [],
        # Add error integrator
        'error_integrator_x': [], 'error_integrator_y': []
    }

    # Get site IDs for visualization markers
    zmp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'zmp')
    com_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'overall_com')

    # Setup visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera position for better visibility
        viewer.cam.distance = 15.0  # Increased distance
        viewer.cam.azimuth = 90   # Side view
        viewer.cam.elevation = -20  # Looking slightly up
        viewer.cam.lookat[0] = 0.4  # Look at point x (ahead of robot)
        viewer.cam.lookat[1] = 0  # Look at point y
        viewer.cam.lookat[2] = 1  # Look at point z (1 meter up)

        print("\nCamera settings:")
        print(f"Distance: {viewer.cam.distance}")
        print(f"Azimuth: {viewer.cam.azimuth}")
        print(f"Elevation: {viewer.cam.elevation}")
        print(f"Look at point: {viewer.cam.lookat}")

        # Setup renderer for video
        width, height = 1280, 720
        renderer = mujoco.Renderer(model, height=height, width=width)

        # Initialize step tracker for single-timestep calculations
        step_tracker = StepTracker()

        # Simulation loop
        step = 0
        sim_time = 0
        esum = np.zeros(shape=(2,1))  # Error sum for controller

        # Previous momenta for ZMP calculation
        prev_lin_mom = None
        prev_ang_mom = None

        # Get joint and actuator IDs
        joint_ids = {
            'hip_x_left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_left'),
            'hip_y_left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_left'),
            'knee_left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_left'),
            'hip_x_right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_right'),
            'hip_y_right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_right'),
            'knee_right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_right')
        }
        
        actuator_ids = {
            'hip_x_left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_x_left'),
            'hip_y_left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_left'),
            'knee_left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_left'),
            'hip_x_right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_x_right'),
            'hip_y_right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_right'),
            'knee_right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_right')
        }

        while viewer.is_running() and sim_time < time_end:
            current_time = sim_time  # Use simulation time instead of wall clock time
            
            # Calculate ZMP and stability data
            com_pos = data.subtree_com[1].copy()  # Get current COM position
            
            # Get robot state for ZMP calculation
            lin_mom = data.subtree_linvel[1].copy() * total_mass
            ang_mom = data.subtree_angmom[1].copy() + np.cross(com_pos, lin_mom)
            
            # Calculate rates of change
            if prev_lin_mom is not None and prev_ang_mom is not None:
                d_lin_mom = (lin_mom - prev_lin_mom) / dt
                d_ang_mom = (ang_mom - prev_ang_mom) / dt
            else:
                d_lin_mom = lin_mom / dt
                d_ang_mom = ang_mom / dt
            
            Fgz = d_lin_mom[2] + Mg
            
            # Check for ground contact
            floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
            has_contact = False
            for i in range(data.ncon):
                contact = data.contact[i]
                if contact.geom1 == floor_id or contact.geom2 == floor_id:
                    has_contact = True
                    break
            
            # Calculate actual ZMP
            if has_contact and Fgz > 20:
                zmp_x = (Mg * com_pos[0] - d_ang_mom[1]) / Fgz
                zmp_y = (Mg * com_pos[1] + d_ang_mom[0]) / Fgz
            else:
                zmp_x = com_pos[0]
                zmp_y = com_pos[1]

            # Calculate control input
            (U, esum) = calculate_controller(Gi, Gx, G, X, P, esum, pref, step)
            
            # Debug print error integrator shape and values (only for first few steps)
            if step < 5:  # Only print first 5 steps to avoid spam
                print(f"\nStep {step} Error Integrator:")
                print(f"Shape: {esum.shape}")
                print(f"Values: X={float(esum[0]):.6f}, Y={float(esum[1]):.6f}")
            
            # Calculate next state
            (X, P) = calculate_state(A, B, C, X, U)

            # Calculate relative positions and angles for this timestep
            rel_pos_x_left, rel_pos_x_right = calculate_relative_hip_positions_single(
                X[0, 0],  # Current planned COM X position
                X[0, 0],  # Current planned COM X position
                P[0, 0],  # Current reference ZMP X
                P[1, 0],  # Current reference ZMP Y
                step_tracker
            )

            # Calculate lateral distances for this timestep using reference ZMP Y
            rel_pos_y_left, rel_pos_y_right = calculate_lateral_distances_single(
                pref[1, step],  # Current reference ZMP Y (from reference trajectory)
                X[1, 0]        # Current planned COM Y
            )

            # Calculate angles directly using leg_inverse_kinematics
            _, left_hip_pitch, left_knee = leg_inverse_kinematics(rel_pos_x_left, -Zc)
            _, right_hip_pitch, right_knee = leg_inverse_kinematics(rel_pos_x_right, -Zc)

            # Set actuator commands (keep angles in degrees since XML uses degrees)
            data.ctrl[actuator_ids['hip_x_left']] = 0.0  # Keep hip X at 0 degrees
            data.ctrl[actuator_ids['hip_y_left']] = left_hip_pitch
            data.ctrl[actuator_ids['knee_left']] = left_knee
            data.ctrl[actuator_ids['hip_x_right']] = 0.0  # Keep hip X at 0 degrees
            data.ctrl[actuator_ids['hip_y_right']] = right_hip_pitch
            data.ctrl[actuator_ids['knee_right']] = right_knee

            # Get torso position for marker placement
            torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
            torso_pos = data.xpos[torso_id]

            # Update visualization markers relative to torso position
            if zmp_site_id >= 0:
                data.site_xpos[zmp_site_id] = [torso_pos[0] + P[0, 0], torso_pos[1] + P[1, 0], 0.01]
            if com_site_id >= 0:
                data.site_xpos[com_site_id] = [torso_pos[0] + com_pos[0], torso_pos[1] + com_pos[1], com_pos[2]]

            # Store data
            plot_data['time'].append(sim_time)
            plot_data['com_x'].append(com_pos[0])
            plot_data['com_y'].append(com_pos[1])
            plot_data['com_z'].append(com_pos[2])
            plot_data['zmp_x'].append(zmp_x)
            plot_data['zmp_y'].append(zmp_y)
            plot_data['preview_com_x'].append(X[0, 0])
            plot_data['preview_com_y'].append(X[1, 0])
            plot_data['preview_zmp_x'].append(P[0, 0])
            plot_data['preview_zmp_y'].append(P[1, 0])
            plot_data['ref_zmp_x'].append(pref[0, step])
            plot_data['ref_zmp_y'].append(pref[1, step])
            
            # Store error integrator values
            plot_data['error_integrator_x'].append(float(esum[0]))  # Convert from numpy array to float
            plot_data['error_integrator_y'].append(float(esum[1]))
            
            # Store relative positions
            plot_data['rel_pos_x_left'].append(rel_pos_x_left)
            plot_data['rel_pos_x_right'].append(rel_pos_x_right)
            plot_data['rel_pos_y_left'].append(rel_pos_y_left)
            plot_data['rel_pos_y_right'].append(rel_pos_y_right)
            
            # Store calculated target angles
            plot_data['target_left_hip_yaw'].append(0.0)  # We're commanding 0 for hip X
            plot_data['target_left_hip_pitch'].append(left_hip_pitch)
            plot_data['target_right_hip_yaw'].append(0.0)  # We're commanding 0 for hip X
            plot_data['target_right_hip_pitch'].append(right_hip_pitch)
            plot_data['target_left_knee'].append(left_knee)
            plot_data['target_right_knee'].append(right_knee)

            # Measure and store actual angles (convert from radians to degrees)
            plot_data['actual_left_hip_yaw'].append(np.degrees(data.qpos[model.jnt_qposadr[joint_ids['hip_x_left']]]))
            plot_data['actual_left_hip_pitch'].append(np.degrees(data.qpos[model.jnt_qposadr[joint_ids['hip_y_left']]]))
            plot_data['actual_right_hip_yaw'].append(np.degrees(data.qpos[model.jnt_qposadr[joint_ids['hip_x_right']]]))
            plot_data['actual_right_hip_pitch'].append(np.degrees(data.qpos[model.jnt_qposadr[joint_ids['hip_y_right']]]))
            plot_data['actual_left_knee'].append(np.degrees(data.qpos[model.jnt_qposadr[joint_ids['knee_left']]]))
            plot_data['actual_right_knee'].append(np.degrees(data.qpos[model.jnt_qposadr[joint_ids['knee_right']]]))

            # Update momenta for next iteration
            prev_lin_mom = lin_mom
            prev_ang_mom = ang_mom

            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update visualization
            viewer.sync()
            
            # Update time and step
            sim_time += dt
            step += 1

            # Control simulation speed
            time.sleep(dt * SIMULATION_SPEED)

        # Generate plots
        plot_results(plot_data, Zc)

if __name__ == "__main__":
    main()