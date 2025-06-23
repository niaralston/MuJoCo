"""
Run the Humanoid Standing Task with a basic stabilizing policy.
"""

from dm_control import suite, viewer
import numpy as np

# Create the environment
environment = suite.load(domain_name="humanoid", task_name="stand")

# Define a simple policy that tries to maintain a standing posture
def standing_policy(time_step):
    """Returns actions that attempt to maintain a standing posture."""
    if not time_step.observation:
        return np.zeros(environment.action_spec().shape)
    
    # Get the action specification to know valid ranges
    spec = environment.action_spec()
    
    # Simple policy: Try to keep joints near zero position with small random noise
    # This won't make it stand perfectly but shows how the simulation works
    action = np.zeros(spec.shape)
    # Add small random noise to create some movement
    action += np.random.uniform(-0.1, 0.1, spec.shape)
    # Clip actions to valid range
    action = np.clip(action, spec.minimum, spec.maximum)
    
    return action

# Launch the viewer with our policy
viewer.launch(environment, policy=standing_policy)