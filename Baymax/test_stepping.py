import mujoco
import mujoco.viewer
import os
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Clear terminal before running
os.system('cls')

# === Simulation and Recording Settings ===
RECORD_VIDEO = False  # Set to True to record the simulation as a video
TARGET_FPS = 30       # Target frames per second for simulation and video
VISUALIZE = False  # Toggle this to False to disable the visualizer
SIMULATION_SPEED = 3.0 if VISUALIZE else 0  # Slow for watching, fast for headless
SIMULATION_TIME = 10.0  # Simulated seconds to run for each configuration
ANGLE_RAMP_DURATION_HIP = 2.0  # Seconds over which to ramp up the hip (thigh) joint angles
ANGLE_RAMP_DURATION_KNEE = 1.0  # Seconds over which to ramp up the knee joint angles

# === Model Loading ===
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "Baymax.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# === Joint and Actuator Lookup ===
right_hip_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_right')
left_hip_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_y_left')
left_knee_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_left')

# === Angle configurations to test ===
test_configurations = [
    (-90, 30, -50),
    (-120, 45, -90),
    (-60, 30, -30),
    (-100, 20, -40),
    (-80, 40, -60),
    (-70, 25, -35),
    (-110, 35, -70),
    (-85, 15, -45)
]

# === Main Testing Loop ===
def run_test_for_configuration(left_hip_angle, right_hip_angle, knee_angle):
    mujoco.mj_resetData(model, data)
    # Set initial positions
    data.qpos[0:3] = [0, 0, 1.0]
    data.qpos[3:7] = [1, 0, 0, 0]
    # Stabilize
    for _ in range(100):
        mujoco.mj_step(model, data)
    # Get body IDs for height tracking
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    initial_torso_height = data.xpos[torso_id][2]
    initial_pelvis_height = data.xpos[pelvis_id][2]
    moved = False
    ramp_start_time = 5.0
    ramp_end_time_hip = ramp_start_time + ANGLE_RAMP_DURATION_HIP
    ramp_end_time_knee = ramp_start_time + ANGLE_RAMP_DURATION_KNEE
    steps_per_frame = int(1.0 / (TARGET_FPS * model.opt.timestep))
    last_print_time = -1
    if VISUALIZE:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 10.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            while viewer.is_running() and data.time < SIMULATION_TIME:
                # Gradual ramp logic
                if ramp_start_time <= data.time < ramp_end_time_hip or ramp_start_time <= data.time < ramp_end_time_knee:
                    # Hip ramp
                    if ramp_start_time <= data.time < ramp_end_time_hip:
                        alpha_hip = (data.time - ramp_start_time) / ANGLE_RAMP_DURATION_HIP
                        alpha_hip = min(max(alpha_hip, 0.0), 1.0)
                    else:
                        alpha_hip = 1.0
                    # Knee ramp
                    if ramp_start_time <= data.time < ramp_end_time_knee:
                        alpha_knee = (data.time - ramp_start_time) / ANGLE_RAMP_DURATION_KNEE
                        alpha_knee = min(max(alpha_knee, 0.0), 1.0)
                    else:
                        alpha_knee = 1.0
                    data.ctrl[left_knee_actuator_id] = alpha_knee * knee_angle
                    data.ctrl[left_hip_actuator_id] = alpha_hip * left_hip_angle
                    data.ctrl[right_hip_actuator_id] = alpha_hip * right_hip_angle
                elif data.time >= max(ramp_end_time_hip, ramp_end_time_knee) and not moved:
                    data.ctrl[left_knee_actuator_id] = knee_angle
                    data.ctrl[left_hip_actuator_id] = left_hip_angle
                    data.ctrl[right_hip_actuator_id] = right_hip_angle
                    moved = True
                    print(f"Applied joint angles at {max(ramp_end_time_hip, ramp_end_time_knee):.1f}s: left_hip={left_hip_angle}, right_hip={right_hip_angle}, knee={knee_angle}")
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                if int(data.time) != last_print_time:
                    print(f"Sim time: {data.time:.1f}s - Torso height: {data.xpos[torso_id][2]:.3f}m")
                    last_print_time = int(data.time)
                viewer.sync()
                time.sleep(0.01 * SIMULATION_SPEED)
    else:
        # Run faster without visualizer
        while data.time < SIMULATION_TIME:
            # Gradual ramp logic
            if ramp_start_time <= data.time < ramp_end_time_hip or ramp_start_time <= data.time < ramp_end_time_knee:
                # Hip ramp
                if ramp_start_time <= data.time < ramp_end_time_hip:
                    alpha_hip = (data.time - ramp_start_time) / ANGLE_RAMP_DURATION_HIP
                    alpha_hip = min(max(alpha_hip, 0.0), 1.0)
                else:
                    alpha_hip = 1.0
                # Knee ramp
                if ramp_start_time <= data.time < ramp_end_time_knee:
                    alpha_knee = (data.time - ramp_start_time) / ANGLE_RAMP_DURATION_KNEE
                    alpha_knee = min(max(alpha_knee, 0.0), 1.0)
                else:
                    alpha_knee = 1.0
                data.ctrl[left_knee_actuator_id] = alpha_knee * knee_angle
                data.ctrl[left_hip_actuator_id] = alpha_hip * left_hip_angle
                data.ctrl[right_hip_actuator_id] = alpha_hip * right_hip_angle
            elif data.time >= max(ramp_end_time_hip, ramp_end_time_knee) and not moved:
                data.ctrl[left_knee_actuator_id] = knee_angle
                data.ctrl[left_hip_actuator_id] = left_hip_angle
                data.ctrl[right_hip_actuator_id] = right_hip_angle
                moved = True
                print(f"Applied joint angles at {max(ramp_end_time_hip, ramp_end_time_knee):.1f}s: left_hip={left_hip_angle}, right_hip={right_hip_angle}, knee={knee_angle}")
            # Run more steps per iteration when not visualizing
            for _ in range(steps_per_frame * 10):  # 10x more steps per iteration
                mujoco.mj_step(model, data)
            if int(data.time) != last_print_time:
                print(f"Sim time: {data.time:.1f}s - Torso height: {data.xpos[torso_id][2]:.3f}m")
                last_print_time = int(data.time)
            # No sleep needed when not visualizing
    final_torso_height = data.xpos[torso_id][2]
    final_pelvis_height = data.xpos[pelvis_id][2]
    print(f"Initial torso height: {initial_torso_height:.3f}m, Final torso height: {final_torso_height:.3f}m")
    print(f"Initial pelvis height: {initial_pelvis_height:.3f}m, Final pelvis height: {final_pelvis_height:.3f}m")
    print(f"Height change: {final_torso_height - initial_torso_height:.3f}m")
    print("="*80)
    return initial_torso_height, final_torso_height, initial_pelvis_height, final_pelvis_height

def main():
    print("Testing multiple configurations exactly as in Baymax.py...")
    print("="*80)
    results = []
    for left_hip, right_hip, knee in test_configurations:
        print(f"\nTesting configuration: left_hip={left_hip}, right_hip={right_hip}, knee={knee}")
        res = run_test_for_configuration(left_hip, right_hip, knee)
        results.append((left_hip, right_hip, knee) + res)
        if VISUALIZE:
            print("Waiting 2 seconds before next configuration...")
            time.sleep(2)
    print("\nSummary of Results:")
    print("="*80)
    for (left_hip, right_hip, knee, init_torso, final_torso, init_pelvis, final_pelvis) in results:
        print(f"Angles: ({left_hip:4d}, {right_hip:4d}, {knee:4d}) | Torso Height: {init_torso:.3f} → {final_torso:.3f} | Pelvis Height: {init_pelvis:.3f} → {final_pelvis:.3f}")

if __name__ == "__main__":
    main() 