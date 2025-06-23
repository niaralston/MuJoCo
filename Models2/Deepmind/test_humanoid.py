from dm_control import suite
from dm_control import viewer
import numpy as np

# Load the humanoid walk environment
env = suite.load('humanoid', 'walk')

# Define a random policy that outputs small random actions
def random_policy(time_step):
    # Get the action specification from the environment
    action_spec = env.action_spec()
    # Generate random actions but scale them down to be smaller
    random_actions = np.random.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
        size=action_spec.shape
    )
    # Scale down the actions to make movements less extreme
    scaling_factor = 0.1
    return random_actions * scaling_factor

print("Starting humanoid simulation...")
print("Controls:")
print("  Spacebar - pause/unpause")
print("  Left mouse button - rotate view")
print("  Right mouse button - zoom")
print("  Middle mouse button - pan")
print("  ESC - quit")

# Launch the viewer
viewer.launch(env, policy=random_policy) 