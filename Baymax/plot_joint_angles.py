import mujoco
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def print_leg_angles(model, data):
    """
    Compute and print:
    - The angle between the left and right thigh vectors (hip extension/retraction)
    - The anatomical knee flexion/extension angle for each leg (thigh-to-shin)
    Uses the center of mass of the thigh and shin bodies to approximate joint positions.
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

    # --- Get body IDs for thighs and shins ---
    left_thigh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thigh_left')
    left_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_left')
    right_thigh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thigh_right')
    right_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shin_right')

    # --- Get world positions of thigh and shin centers of mass ---
    left_hip_pos = data.xpos[left_thigh_id]
    left_knee_pos = data.xpos[left_shin_id]
    right_hip_pos = data.xpos[right_thigh_id]
    right_knee_pos = data.xpos[right_shin_id]

    # --- Calculate thigh vectors (hip to knee) for each leg ---
    left_thigh_vec = left_knee_pos - left_hip_pos
    right_thigh_vec = right_knee_pos - right_hip_pos

    # --- Hip angle: angle between left and right thigh vectors (projected onto XZ plane) ---
    left_thigh_vec_proj = left_thigh_vec.copy()
    right_thigh_vec_proj = right_thigh_vec.copy()
    left_thigh_vec_proj[1] = 0
    right_thigh_vec_proj[1] = 0
    hip_angle = angle_between(left_thigh_vec_proj, right_thigh_vec_proj)

    # --- Estimate ankle positions by extending the shin vectors ---
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
    
    return hip_angle, left_knee_angle, right_knee_angle

def get_joint_angles(model, data):
    """
    Get the actual joint angles from MuJoCo data for the right leg joints.
    Returns angles in degrees.
    """
    # Get joint IDs
    hip_y_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_y_right')
    hip_x_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_x_right')
    hip_z_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip_z_right')
    knee_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_right')
    
    # Get joint angles (convert from radians to degrees)
    hip_y_angle = np.degrees(data.qpos[hip_y_right_id])
    hip_x_angle = np.degrees(data.qpos[hip_x_right_id])
    hip_z_angle = np.degrees(data.qpos[hip_z_right_id])
    knee_angle = np.degrees(data.qpos[knee_right_id])
    
    return hip_y_angle, hip_x_angle, hip_z_angle, knee_angle

def plot_joint_angles_vs_time():
    """
    Function to plot target vs actual joint angles over time using the target angles from Baymax.py
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "Baymax.xml")

    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Find the actuators
    hip_y_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_right')
    hip_x_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_x_right')
    hip_z_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_z_right')
    knee_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_right')

    # Target angles from Baymax.py
    RIGHT_KNEE_TARGET_ANGLE = -40
    RIGHT_HIP_TARGET_ANGLE = -90
    LEFT_HIP_TARGET_ANGLE = 0
    LEFT_KNEE_TARGET_ANGLE = 0
    LEAN_ANGLE = 12

    # Simulation parameters
    simulation_time = 5.0  # Time to run simulation
    timestep = model.opt.timestep
    steps_per_sim = int(simulation_time / timestep)
    
    # Create time array
    time_array = np.linspace(0, simulation_time, steps_per_sim)
    
    # Arrays to store target and actual angles
    hip_y_target = np.full(steps_per_sim, RIGHT_HIP_TARGET_ANGLE)
    hip_x_target = np.full(steps_per_sim, LEAN_ANGLE)  # Using lean angle for hip_x
    hip_z_target = np.zeros(steps_per_sim)  # No rotation target
    knee_target = np.full(steps_per_sim, RIGHT_KNEE_TARGET_ANGLE)
    
    # Arrays to store actual angles
    hip_y_actual = np.zeros(steps_per_sim)
    hip_x_actual = np.zeros(steps_per_sim)
    hip_z_actual = np.zeros(steps_per_sim)
    knee_actual = np.zeros(steps_per_sim)
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    print("Running simulation and collecting data...")
    
    # Run simulation
    for i in range(steps_per_sim):
        # Set control signals
        data.ctrl[hip_y_actuator_id] = hip_y_target[i]
        data.ctrl[hip_x_actuator_id] = hip_x_target[i]
        data.ctrl[hip_z_actuator_id] = hip_z_target[i]
        data.ctrl[knee_actuator_id] = knee_target[i]
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Get actual joint angles
        hip_y_actual[i], hip_x_actual[i], hip_z_actual[i], knee_actual[i] = get_joint_angles(model, data)
    
    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Target vs Actual Joint Angles Over Time', fontsize=16)
    
    # Plot 1: Hip Y (forward/back)
    axes[0, 0].plot(time_array, hip_y_target, 'b-', label='Target', linewidth=2)
    axes[0, 0].plot(time_array, hip_y_actual, 'r--', label='Actual', linewidth=2)
    axes[0, 0].set_title('Hip Y (Forward/Back)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (degrees)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Hip X (side-to-side)
    axes[0, 1].plot(time_array, hip_x_target, 'b-', label='Target', linewidth=2)
    axes[0, 1].plot(time_array, hip_x_actual, 'r--', label='Actual', linewidth=2)
    axes[0, 1].set_title('Hip X (Side-to-Side)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angle (degrees)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Hip Z (rotation)
    axes[1, 0].plot(time_array, hip_z_target, 'b-', label='Target', linewidth=2)
    axes[1, 0].plot(time_array, hip_z_actual, 'r--', label='Actual', linewidth=2)
    axes[1, 0].set_title('Hip Z (Rotation)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Angle (degrees)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Knee
    axes[1, 1].plot(time_array, knee_target, 'b-', label='Target', linewidth=2)
    axes[1, 1].plot(time_array, knee_actual, 'r--', label='Actual', linewidth=2)
    axes[1, 1].set_title('Knee (Flexion/Extension)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angle (degrees)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Hip Y - Max Error: {np.max(np.abs(hip_y_target - hip_y_actual)):.2f}°, RMS Error: {np.sqrt(np.mean((hip_y_target - hip_y_actual)**2)):.2f}°")
    print(f"Hip X - Max Error: {np.max(np.abs(hip_x_target - hip_x_actual)):.2f}°, RMS Error: {np.sqrt(np.mean((hip_x_target - hip_x_actual)**2)):.2f}°")
    print(f"Hip Z - Max Error: {np.max(np.abs(hip_z_target - hip_z_actual)):.2f}°, RMS Error: {np.sqrt(np.mean((hip_z_target - hip_z_actual)**2)):.2f}°")
    print(f"Knee  - Max Error: {np.max(np.abs(knee_target - knee_actual)):.2f}°, RMS Error: {np.sqrt(np.mean((knee_target - knee_actual)**2)):.2f}°")

if __name__ == "__main__":
    plot_joint_angles_vs_time() 