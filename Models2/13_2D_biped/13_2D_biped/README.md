| Variable | Description | Dimensions | Example Usage |
|----------|-------------|------------|---------------|
| data.time | Current simulation time in seconds | Scalar | Used to track simulation progress |
| data.qpos | Joint positions (generalized coordinates) | [nq] | data.qpos[4] = 0.5 sets initial joint position |
| data.qvel | Joint velocities | [nv] | Used to get joint velocities |
| data.ctrl | Control signals for actuators | [nu] | data.ctrl[0] = -0.5 applies torque to swing leg |
| data.xpos | Cartesian positions of all bodies | [nbody, 3] | pos_foot1 = data.xpos[2, :] gets foot position |
| data.xquat | Quaternion orientations of all bodies | [nbody, 4] | quat_leg1 = data.xquat[1, :] gets leg orientation |
| data.sensordata | Sensor readings | [nsensor] | Used for force sensors, gyros, etc. |
| data.cfrc_ext | External forces/torques on bodies | [nbody, 6] | Used for contact forces |
| data.actuator | Actuator states | [nu] | Contains actuator lengths and velocities |

Special indices for our biped:

Body indices:
- 1: First leg
- 2: First foot  
- 3: Second leg
- 4: Second foot

Control indices:
- 0: Hip joint torque
- 2: Leg 1 extension
- 4: Leg 2 extension

Important derived values:

- abs_leg1 = -euler_leg1[1]: Absolute angle of leg 1 (negative pitch)
  - Positive: leg leans backward (ready to push)
  - Negative: leg leans forward (ready to catch)