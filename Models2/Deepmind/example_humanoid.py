"""Example of loading and visualizing the Humanoid environment.

This example shows how to:
1. Load the humanoid environment
2. Apply random actions and visualize the results
3. Render the environment
"""

from dm_control import suite
from dm_control import viewer
import numpy as np

# Load the humanoid task
env = suite.load(domain_name="humanoid", task_name="walk")

# Define a simple policy that generates random actions
def random_policy(time_step):
    """Outputs random actions for the given timestep."""
    del time_step  # Unused
    # Get the action spec from the environment
    action_spec = env.action_spec()
    # Generate random actions within the valid range
    random_action = np.random.uniform(
        action_spec.minimum,
        action_spec.maximum,
        size=action_spec.shape)
    return random_action

# Launch the viewer
viewer.launch(env, policy=random_policy) 