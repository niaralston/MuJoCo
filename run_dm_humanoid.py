from dm_control import suite
from dm_control import viewer
import numpy as np

# Load the humanoid task - we'll use the "stand" task which is simpler to start with
env = suite.load(domain_name="humanoid", task_name="stand")

# Define a simple policy that tries to maintain a stable standing pose
def standing_policy(time_step):
    # Get the physics state
    if not time_step.first():
        # Extract relevant state information
        joint_angles = time_step.observation['joint_angles']
        velocity = time_step.observation['velocity']
        
        # Simple PD controller for each joint
        kp = 100.0  # Proportional gain
        kd = 10.0   # Derivative gain
        
        # Target joint angles for standing (approximately upright pose)
        target_angles = np.zeros_like(joint_angles)
        
        # Compute control signal
        position_error = target_angles - joint_angles
        velocity_error = -velocity[6:]  # Skip root velocity
        
        # Combine position and velocity feedback
        action = kp * position_error + kd * velocity_error
        
        # Clip actions to valid range [-1, 1]
        action = np.clip(action, -1, 1)
    else:
        # On first step, return neutral action
        action = np.zeros(env.action_spec().shape)
    
    return action

print("Starting humanoid simulation...")
print("Controls: Left mouse button - rotate view")
print("         Right mouse button - zoom")
print("         Middle mouse button - pan")
print("         ESC - quit")

# Launch the viewer with our environment and policy
viewer.launch(env, policy=standing_policy) 