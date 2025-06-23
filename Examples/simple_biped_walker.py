import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2  # For video recording
import os   # For file path operations
import platform    # For OS detection
import subprocess # For running system commands

# Clear terminal
os.system('cls')

# --- User Settings ---
RECORD_VIDEO = True # Set to True to record the simulation, False to just view it
simend = 120  # simulation duration in seconds

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path("Examples/simple_biped.xml")
    data = mujoco.MjData(model)

    # Initialize the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset the simulation
        mujoco.mj_resetData(model, data)
        
        # Set up camera for better view
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 90  # Looking from the side, positive X is to the right
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0, 0.0, 0.1]  # Look at robot's center

        # Video recording setup
        if RECORD_VIDEO:
            fps = 60  # Frame rate for recording
            # Use a smaller resolution that works with default framebuffer
            width, height = 640, 480
            
            # Set up video file path in the current directory
            video_path = 'my_biped_walker_model1.avi'
            print(f"\nVideo will be saved to: {os.path.abspath(video_path)}")
            
            # Create renderer for video
            renderer = mujoco.Renderer(model, height=height, width=width)
            
            # Set initial camera parameters for renderer
            renderer.update_scene(data, camera=viewer.cam)
            
            # Use XVID codec which is more widely supported
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)
            
            if not out.isOpened():
                raise RuntimeError("Failed to create video writer. Check if you have write permissions in this directory.")
            
            print("Recording simulation... (Move camera with mouse to adjust view)")
            frames_written = 0  # Counter to track number of frames written
        
        # Variables for timing
        sim_start = time.time()
        last_update = sim_start
        
        print("\nWalking Direction Guide:")
        print("- Robot walks toward positive X (to the right)")
        print("- Hip angles: positive = forward swing, negative = backward swing")
        print("- Knee angles: positive = bend forward (toward walking direction)")
        
        # Set initial position
        data.qpos[:2] = [0, 0]
        
        print("Starting initialization sequence...")
        # Initialize in a stable standing position with weight on both feet
        step = 0
        phase_duration = 2000  # Faster steps for better momentum
        
        # Walking poses with proper push-off mechanics
        poses = {
            # Phase 0: Double support, weight shift to right leg
            0: {
                'left_hip': -5,      # Slight back lean
                'left_knee': 15,     # Slightly bent
                'right_hip': -10,    # Support leg back
                'right_knee': 25     # Support leg bent
            },
            # Phase 1: Right support, left leg lift and forward swing
            1: {
                'left_hip': 20,      # Forward swing
                'left_knee': 45,     # Significant bend for clearance
                'right_hip': -15,    # Support leg back
                'right_knee': 20     # Support leg slightly bent
            },
            # Phase 2: Right push-off, left leg forward reach
            2: {
                'left_hip': 25,      # Forward reach
                'left_knee': 10,     # Preparing for landing
                'right_hip': -5,     # Push-off
                'right_knee': 5      # Almost straight for push
            },
            # Phase 3: Double support, weight shift to left leg
            3: {
                'left_hip': -10,     # Support leg back
                'left_knee': 25,     # Support leg bent
                'right_hip': -5,     # Slight back lean
                'right_knee': 15     # Slightly bent
            },
            # Phase 4: Left support, right leg lift and forward swing
            4: {
                'left_hip': -15,     # Support leg back
                'left_knee': 20,     # Support leg slightly bent
                'right_hip': 20,     # Forward swing
                'right_knee': 45     # Significant bend for clearance
            },
            # Phase 5: Left push-off, right leg forward reach
            5: {
                'left_hip': -5,      # Push-off
                'left_knee': 5,      # Almost straight for push
                'right_hip': 25,     # Forward reach
                'right_knee': 10     # Preparing for landing
            }
        }
        
        # Run simulation until end time or window closes
        while viewer.is_running() and data.time < simend:
            # Maintain consistent simulation speed
            now = time.time()
            sim_time = now - sim_start
            
            # Run physics steps to catch up to wall clock
            while data.time < sim_time and data.time < simend:
                if step < 2000:  # Initialization phase
                    data.ctrl[0] = np.deg2rad(0)    # Left hip neutral
                    data.ctrl[1] = np.deg2rad(25)   # Left knee bent forward
                    data.ctrl[2] = np.deg2rad(0)    # Right hip neutral
                    data.ctrl[3] = np.deg2rad(25)   # Right knee bent forward
                    
                    if step % 100 == 0:
                        print(f"Forward position (X): {data.qpos[0]:.2f}m")
                else:  # Walking phase
                    if step == 2000:
                        print("\nStarting walking sequence...")
                    
                    phase = (step // (phase_duration // 6)) % 6
                    phase_progress = (step % (phase_duration // 6)) / (phase_duration // 6)
                    
                    # Get current and next pose
                    current_pose = poses[phase]
                    next_pose = poses[(phase + 1) % 6]
                    
                    # Smooth transition between poses
                    t = (1 - np.cos(phase_progress * np.pi)) / 2
                    
                    # Apply controls with emphasis on push-off and forward progression
                    for i, joint in enumerate(['left_hip', 'left_knee', 'right_hip', 'right_knee']):
                        current_angle = current_pose[joint]
                        next_angle = next_pose[joint]
                        
                        # Base interpolation
                        angle = current_angle * (1 - t) + next_angle * t
                        
                        # Add push-off during stance-to-swing transition
                        if ((phase in [1, 2] and i >= 2) or (phase in [4, 5] and i < 2)):  # During push-off
                            if 0.5 < phase_progress < 0.9:  # Later in stance phase
                                push_factor = np.sin((phase_progress - 0.5) * 2.5 * np.pi)
                                if i % 2 == 0:  # Hip joints
                                    angle += 15 * push_factor  # Extra hip extension
                                else:  # Knee joints
                                    angle -= 20 * push_factor  # Strong push with knee straightening
                        
                        # Add extra lift during early swing phase
                        if ((phase in [1] and i < 2) or (phase in [4] and i >= 2)):  # During swing
                            if 0.1 < phase_progress < 0.4:  # Early in swing
                                lift_factor = np.sin((phase_progress - 0.1) * 3.3 * np.pi)
                                if i % 2 == 1:  # Knee joints
                                    angle += 20 * lift_factor  # Extra knee bend for clearance
                        
                        data.ctrl[i] = np.deg2rad(angle)
                    
                    if step % 500 == 0:
                        print(f"\nPhase: {phase}")
                        print(f"Time: {data.time:.2f}s")
                        print(f"Forward position (X): {data.qpos[0]:.2f}m")
                        print(f"Current joint angles (deg):")
                        print(f"  Left hip: {np.rad2deg(data.ctrl[0]):.1f}°")
                        print(f"  Left knee: {np.rad2deg(data.ctrl[1]):.1f}°")
                        print(f"  Right hip: {np.rad2deg(data.ctrl[2]):.1f}°")
                        print(f"  Right knee: {np.rad2deg(data.ctrl[3]):.1f}°")
                        print("---")
                
                mujoco.mj_step(model, data)
                step += 1
            
            # Update visualization
            if now - last_update >= 1.0/fps:
                viewer.sync()
                if RECORD_VIDEO:
                    # Update renderer with current camera view
                    renderer.update_scene(data, camera=viewer.cam)
                    # Render and save frame
                    pixels = renderer.render()
                    out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                    frames_written += 1
                last_update = now

        # Clean up video recording
        if RECORD_VIDEO:
            print(f"\nSimulation finished. Total time: {time.time() - sim_start:.2f} seconds")
            print(f"Frames written: {frames_written}")
            print("Saving video...")
            out.release()
            
            # Verify the video file was created
            if os.path.exists(video_path):
                print(f"Video saved successfully to: {os.path.abspath(video_path)}")
                print(f"File size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
            else:
                print("Warning: Video file was not created successfully!")

if __name__ == "__main__":
    main() 