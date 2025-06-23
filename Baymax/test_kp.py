import mujoco
import time
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

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

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "Baymax.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Find the actuators
hip_y_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_right')
knee_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_right')

# Target angles and simulation parameters
hip_target_angles = [10.0, 20.0, 30.0, 40.0]  # Different hip target angles
knee_target_angles = [5.0, 10.0, 15.0, 20.0]  # Different knee target angles
simulation_time = 2.0  # Time to run each simulation
timestep = model.opt.timestep
steps_per_sim = int(simulation_time / timestep)

# Kp values to test
kp_values = [19.0, 19.2, 19.4, 19.6, 19.8, 20.0, 20.2, 20.4, 20.6, 20.8, 21.0]

# Store results for each kp
kp_results = {}

print("\nTesting different kp values across angle ranges")
print(f"Simulation time per test: {simulation_time}s")

for kp in kp_values:
    print(f"\nTesting kp = {kp}")
    
    # Set kp for all actuators
    model.actuator_gainprm[:, 0] = kp
    
    # Initialize error tracking
    total_hip_error = 0
    total_knee_error = 0
    num_tests = 0
    max_hip_error = 0
    max_knee_error = 0
    
    # Test all combinations of hip and knee angles
    for hip_target in hip_target_angles:
        for knee_target in knee_target_angles:
            print(f"Testing Hip: {hip_target}°, Knee: {knee_target}°")
            
            # Reset simulation
            mujoco.mj_resetData(model, data)
            
            # Set control signals
            data.ctrl[hip_y_actuator_id] = hip_target
            data.ctrl[knee_actuator_id] = knee_target
            
            # Run simulation
            for _ in range(steps_per_sim):
                mujoco.mj_step(model, data)
            
            # Get final angles
            final_hip_angle, final_left_knee, final_right_knee = print_leg_angles(model, data)
            hip_error = abs(final_hip_angle - hip_target)
            knee_error = abs(final_right_knee - knee_target)
            
            # Update error tracking
            total_hip_error += hip_error
            total_knee_error += knee_error
            num_tests += 1
            max_hip_error = max(max_hip_error, hip_error)
            max_knee_error = max(max_knee_error, knee_error)
            
            print(f"Results:")
            print(f"Final hip angle: {final_hip_angle:.1f}° (Error: {hip_error:.1f}°)")
            print(f"Final knee angle: {final_right_knee:.1f}° (Error: {knee_error:.1f}°)")
    
    # Calculate average errors
    avg_hip_error = total_hip_error / num_tests
    avg_knee_error = total_knee_error / num_tests
    
    # Store results
    kp_results[kp] = {
        'avg_hip_error': avg_hip_error,
        'avg_knee_error': avg_knee_error,
        'max_hip_error': max_hip_error,
        'max_knee_error': max_knee_error
    }
    
    print(f"\nSummary for kp = {kp}:")
    print(f"Average hip error: {avg_hip_error:.2f}°")
    print(f"Average knee error: {avg_knee_error:.2f}°")
    print(f"Maximum hip error: {max_hip_error:.2f}°")
    print(f"Maximum knee error: {max_knee_error:.2f}°")

# Find best kp values
best_hip_kp = min(kp_results.items(), key=lambda x: x[1]['avg_hip_error'])
best_knee_kp = min(kp_results.items(), key=lambda x: x[1]['avg_knee_error'])

print("\nOverall Results:")
print(f"Best kp for hip: {best_hip_kp[0]} (avg error: {best_hip_kp[1]['avg_hip_error']:.2f}°)")
print(f"Best kp for knee: {best_knee_kp[0]} (avg error: {best_knee_kp[1]['avg_knee_error']:.2f}°)")

# Print all results in a table format
print("\nDetailed Results Table:")
print("kp\tAvg Hip Error\tAvg Knee Error\tMax Hip Error\tMax Knee Error")
print("-" * 70)
for kp, results in sorted(kp_results.items()):
    print(f"{kp}\t{results['avg_hip_error']:.2f}\t\t{results['avg_knee_error']:.2f}\t\t{results['max_hip_error']:.2f}\t\t{results['max_knee_error']:.2f}") 