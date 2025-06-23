# --- Lean and Z rotation ramp first ---
if current_time < LEAN_RAMP_END_TIME:
    # Ramp up the lean and Z rotation together
    alpha = (current_time - LEAN_RAMP_START_TIME) / LEAN_RAMP_DURATION
    alpha = min(max(alpha, 0.0), 1.0)
    # Apply lean
    data.ctrl[hip_x_left_actuator_id] = alpha * LEAN_ANGLE
    data.ctrl[hip_x_right_actuator_id] = -alpha * LEAN_ANGLE
    # Apply Z rotation
    data.ctrl[hip_z_right_actuator_id] = alpha * RIGHT_HIP_Z_TARGET
    data.ctrl[hip_z_left_actuator_id] = alpha * LEFT_HIP_Z_TARGET
    # Keep other joints at zero
    data.ctrl[right_knee_actuator_id] = 0
    data.ctrl[right_hip_actuator_id] = 0
    data.ctrl[left_hip_actuator_id] = 0
    data.ctrl[left_knee_actuator_id] = 0
# --- Hip/knee ramps after lean is complete ---
elif (ramp_start_time <= current_time < ramp_end_time_right_hip or
      ramp_start_time <= current_time < ramp_end_time_left_hip or
      ramp_start_time <= current_time < ramp_end_time_right_knee or
      ramp_start_time <= current_time < ramp_end_time_left_knee):
    # --- Hip ramp logic ---
    # Right hip
    if ramp_start_time <= current_time < ramp_end_time_right_hip:
        alpha_right_hip = (current_time - ramp_start_time) / RIGHT_HIP_RAMP_DURATION
        alpha_right_hip = min(max(alpha_right_hip, 0.0), 1.0)
    else:
        alpha_right_hip = 1.0
    # Left hip
    if ramp_start_time <= current_time < ramp_end_time_left_hip:
        alpha_left_hip = (current_time - ramp_start_time) / LEFT_HIP_RAMP_DURATION
        alpha_left_hip = min(max(alpha_left_hip, 0.0), 1.0)
    else:
        alpha_left_hip = 1.0
    # --- Knee ramp logic ---
    # Right knee
    if ramp_start_time <= current_time < ramp_end_time_right_knee:
        alpha_right_knee = (current_time - ramp_start_time) / RIGHT_KNEE_RAMP_DURATION
        alpha_right_knee = min(max(alpha_right_knee, 0.0), 1.0)
    else:
        alpha_right_knee = 1.0
    # Left knee
    if ramp_start_time <= current_time < ramp_end_time_left_knee:
        alpha_left_knee = (current_time - ramp_start_time) / LEFT_KNEE_RAMP_DURATION
        alpha_left_knee = min(max(alpha_left_knee, 0.0), 1.0)
    else:
        alpha_left_knee = 1.0
    # Apply the interpolated control values to the actuators
    data.ctrl[right_knee_actuator_id] = alpha_right_knee * RIGHT_KNEE_TARGET_ANGLE
    data.ctrl[right_hip_actuator_id] = alpha_right_hip * RIGHT_HIP_TARGET_ANGLE
    data.ctrl[left_hip_actuator_id] = alpha_left_hip * LEFT_HIP_TARGET_ANGLE
    data.ctrl[left_knee_actuator_id] = alpha_left_knee * LEFT_KNEE_TARGET_ANGLE
    # Apply Z rotation with ramp
    alpha_z = (current_time - ramp_start_time) / RIGHT_HIP_RAMP_DURATION
    alpha_z = min(max(alpha_z, 0.0), 1.0)
    data.ctrl[hip_z_right_actuator_id] = alpha_z * RIGHT_HIP_Z_TARGET
    data.ctrl[hip_z_left_actuator_id] = alpha_z * LEFT_HIP_Z_TARGET
    
    # Lean stays at full value after ramp
    data.ctrl[hip_x_left_actuator_id] = LEAN_ANGLE
    data.ctrl[hip_x_right_actuator_id] = -LEAN_ANGLE
    angle_ramp_started = True
# Once all ramps are complete, set the joints to their final target angles (only once)
elif (current_time >= max(ramp_end_time_right_hip, ramp_end_time_left_hip, 
                         ramp_end_time_right_knee, ramp_end_time_left_knee) and not moved):
    data.ctrl[right_knee_actuator_id] = RIGHT_KNEE_TARGET_ANGLE
    data.ctrl[right_hip_actuator_id] = RIGHT_HIP_TARGET_ANGLE
    data.ctrl[left_hip_actuator_id] = LEFT_HIP_TARGET_ANGLE
    data.ctrl[left_knee_actuator_id] = LEFT_KNEE_TARGET_ANGLE
    # Set full lean after ramp
    data.ctrl[hip_x_left_actuator_id] = LEAN_ANGLE
    data.ctrl[hip_x_right_actuator_id] = -LEAN_ANGLE
    
    # Set final Z rotation
    data.ctrl[hip_z_right_actuator_id] = RIGHT_HIP_Z_TARGET
    data.ctrl[hip_z_left_actuator_id] = LEFT_HIP_Z_TARGET
    
    moved = True
    print(f"Applied joint angles at {max(ramp_end_time_right_hip, ramp_end_time_left_hip, ramp_end_time_right_knee, ramp_end_time_left_knee):.1f}s: "
          f"right_hip_z={RIGHT_HIP_Z_TARGET}, left_hip_z={LEFT_HIP_Z_TARGET}, "
          f"right_knee={RIGHT_KNEE_TARGET_ANGLE}, left_knee={LEFT_KNEE_TARGET_ANGLE}") 