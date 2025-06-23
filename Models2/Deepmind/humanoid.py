# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Humanoid Domain - A MuJoCo-based simulation environment for a humanoid robot.

This module implements a humanoid robot simulation with different tasks:
- Standing still
- Walking at a specific speed
- Running at a specific speed

The simulation uses the MuJoCo physics engine through DeepMind's dm_control library.
"""

import collections

from dm_control import mujoco  # Main MuJoCo physics engine interface
from dm_control.rl import control  # Reinforcement learning environment tools
from dm_control.suite import base  # Base classes for tasks
from dm_control.suite import common  # Common utilities
from dm_control.suite.utils import randomizers  # For random state initialization
from dm_control.utils import containers  # Container data structures
from dm_control.utils import rewards  # Reward computation utilities
import numpy as np

# Global constants for the simulation
_DEFAULT_TIME_LIMIT = 25  # Maximum duration of an episode in seconds
_CONTROL_TIMESTEP = .025  # Time between each control action (40Hz)

# Performance thresholds
_STAND_HEIGHT = 1.4  # Minimum height (meters) for maximum standing reward
_WALK_SPEED = 1    # Target speed (m/s) for walking task
_RUN_SPEED = 10   # Target speed (m/s) for running task

# Container for all available tasks in this domain
SUITE = containers.TaggedTasks()

def get_model_and_assets():
    """Loads the humanoid model XML and associated assets (meshes, textures, etc.)."""
    return common.read_model('humanoid.xml'), common.ASSETS

@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Creates a task where the humanoid must maintain an upright standing posture.
    
    This function sets up a virtual environment where a robot tries to learn to stand.
    Think of it like creating a virtual gym where a robot practices standing up.
    
    Args:
        time_limit: How many seconds each attempt should last (default is 25 seconds)
        random: A seed number to make random actions reproducible (like rolling the same dice)
        environment_kwargs: Extra settings for the environment (optional)
    
    Returns:
        A complete simulation environment ready for the robot to start practicing
    """
    # Step 1: Create the robot's physical body
    # - Loads the 3D model of the robot from an XML file
    # - This includes all body parts (legs, arms, torso) and their properties
    # - Like loading a character model in a video game
    physics = Physics.from_xml_string(*get_model_and_assets())

    # Step 2: Create the standing task
    # move_speed=0: The robot should try to stand still (not walk or run)
    # pure_state=False: Give us processed, easy-to-understand data about the robot's state
    # random: Use the provided random seed (for reproducible results)
    task = Humanoid(move_speed=0, pure_state=False, random=random)

    # Step 3: Handle optional extra settings
    # If no extra settings were provided, use an empty dictionary
    # This prevents errors when no extra settings are given
    environment_kwargs = environment_kwargs or {}

    # Step 4: Create and return the complete simulation environment
    # This combines everything needed for the robot to start learning:
    #  - physics: The robot's physical body and how it moves
    #  - task: What the robot is trying to learn (standing)
    #  - time_limit: How long each practice attempt lasts
    #  - control_timestep: How often the robot can make decisions (40 times per second)
    #  - **environment_kwargs: Any extra settings provided
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Creates a task where the humanoid must walk at a specific speed.
    
    Similar to stand(), but with a target walking speed of _WALK_SPEED m/s.
    """
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Humanoid(move_speed=_WALK_SPEED, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Creates a task where the humanoid must run at a specific speed.
    
    Similar to walk(), but with a faster target speed of _RUN_SPEED m/s.
    """
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Humanoid(move_speed=_RUN_SPEED, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

@SUITE.add()
def run_pure_state(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
    """Creates a running task that returns raw state observations.
    
    Unlike the regular run() task, this returns the pure MuJoCo state rather
    than processed features, which can be useful for some learning approaches.
    """
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Humanoid(move_speed=_RUN_SPEED, pure_state=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

class Physics(mujoco.Physics):
    """Extended physics class with humanoid-specific helper methods.
    
    This class adds convenience methods for accessing relevant physical quantities
    of the humanoid, such as joint angles, body positions, and velocities.
    """

    def torso_upright(self):
        """Returns how upright the torso is (1 = perfectly upright, 0 = horizontal)."""
        return self.named.data.xmat['torso', 'zz']

    def head_height(self):
        """Returns the height of the humanoid's head above the ground in meters."""
        return self.named.data.xpos['head', 'z']

    def center_of_mass_position(self):
        """Returns the 3D position of the humanoid's center of mass."""
        return self.named.data.subtree_com['torso'].copy()

    def center_of_mass_velocity(self):
        """Returns the 3D velocity of the humanoid's center of mass."""
        return self.named.data.sensordata['torso_subtreelinvel'].copy()

    def torso_vertical_orientation(self):
        """Returns the torso's orientation relative to the vertical (z) axis."""
        return self.named.data.xmat['torso', ['zx', 'zy', 'zz']]

    def joint_angles(self):
        """Returns all joint angles excluding the root (global) position/orientation."""
        return self.data.qpos[7:].copy()  # Skip the 7 DoFs of the free root joint

    def extremities(self):
        """Returns positions of hands and feet relative to the torso.
        
        This transforms the global positions of the extremities into the torso's
        reference frame, which makes them invariant to the humanoid's global position
        and orientation.
        """
        torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
        torso_pos = self.named.data.xpos['torso']
        positions = []
        for side in ('left_', 'right_'):
            for limb in ('hand', 'foot'):
                torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
                positions.append(torso_to_limb.dot(torso_frame))
        return np.hstack(positions)

class Humanoid(base.Task):
    """The main task class defining the humanoid's objectives and rewards.
    
    This class implements the core logic for the humanoid tasks, including:
    - Initialization of episodes
    - Generation of observations
    - Computation of rewards
    """

    def __init__(self, move_speed, pure_state, random=None):
        """Initializes a humanoid task.
        
        Args:
            move_speed: Target horizontal speed in m/s (0 for standing task)
            pure_state: If True, return raw MuJoCo state; if False, return processed features
            random: Random number generator or seed for reproducibility
        """
        self._move_speed = move_speed
        self._pure_state = pure_state
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets up the humanoid state at the start of each episode.
        
        This method repeatedly randomizes the humanoid's configuration until
        finding one without any body part collisions.
        
        Args:
            physics: The physics simulation instance
        """
        # Find a collision-free random initial configuration
        penetrating = True
        while penetrating:
            # Randomize joint angles within their limits
            randomizers.randomize_limited_and_rotational_joints(physics, self.random)
            # Update physics and check for collisions
            physics.after_reset()
            penetrating = physics.data.ncon > 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns the current observation of the humanoid's state.
        
        Args:
            physics: The physics simulation instance
        
        Returns:
            Either raw state or processed features depending on self._pure_state
        """
        obs = collections.OrderedDict()
        if self._pure_state:
            # Return raw MuJoCo state
            obs['position'] = physics.position()
            obs['velocity'] = physics.velocity()
        else:
            # Return processed features
            obs['joint_angles'] = physics.joint_angles()
            obs['head_height'] = physics.head_height()
            obs['extremities'] = physics.extremities()
            obs['torso_vertical'] = physics.torso_vertical_orientation()
            obs['com_velocity'] = physics.center_of_mass_velocity()
            obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Computes the reward for the current state.
        
        The reward combines multiple components:
        1. Standing reward: based on head height and torso orientation
        2. Control reward: penalizes large control inputs
        3. Movement reward: based on horizontal velocity (if move_speed > 0)
        
        Args:
            physics: The physics simulation instance
        
        Returns:
            A scalar reward value
        """
        # Reward for standing upright
        standing = rewards.tolerance(physics.head_height(),
                                   bounds=(_STAND_HEIGHT, float('inf')),
                                   margin=_STAND_HEIGHT/4)
        upright = rewards.tolerance(physics.torso_upright(),
                                  bounds=(0.9, float('inf')), sigmoid='linear',
                                  margin=1.9, value_at_margin=0)
        stand_reward = standing * upright
        
        # Reward for minimal control input (energy efficiency)
        small_control = rewards.tolerance(physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5
        
        if self._move_speed == 0:
            # For standing task: reward staying still
            horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move
        else:
            # For walking/running tasks: reward moving at target speed
            com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
            move = rewards.tolerance(com_velocity,
                                   bounds=(self._move_speed, float('inf')),
                                   margin=self._move_speed, value_at_margin=0,
                                   sigmoid='linear')
            move = (5*move + 1) / 6
            return small_control * stand_reward * move
