import mujoco
import mujoco.viewer
import time
import os

# Path to the balloon example
balloon_xml_path = os.path.join(os.path.dirname(__file__), "balloons.xml")  # Use absolute path based on script location

print(f"MuJoCo version: {mujoco.__version__}")
print(f"Attempting to load balloon example from: {balloon_xml_path}")

try:
    # Check if the file exists
    if not os.path.exists(balloon_xml_path):
        print(f"❌ Error: File '{balloon_xml_path}' not found!")
        exit(1)
        
    # Load the model
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path(balloon_xml_path)
    data = mujoco.MjData(model)
    print("✅ Successfully loaded balloon example!")
    
    # Print some model information that is available in version 3.3.2
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of tendons: {model.ntendon}")
    print(f"Timestep: {model.opt.timestep}")
    
    # Run the simulation
    print("\nStarting simulation...")
    print("This will open a viewer window. Close the window to end the simulation.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running() and (time.time() - start_time) < 120:  # Run for up to 2 minutes
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Update the viewer
            viewer.sync()
            
            # Print step info occasionally
            step_count = int(data.time / model.opt.timestep)
            if step_count % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Simulation time: {data.time:.2f}s, Real time: {elapsed:.2f}s")
    
    print("\n✅ Simulation completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
    # Provide more detailed error information
    import traceback
    traceback.print_exc()