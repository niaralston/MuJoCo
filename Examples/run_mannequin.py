import mujoco
import mujoco.viewer
import time
import os

# Path to the mannequin example
mannequin_xml_path = "mannequin.xml"  # Make sure this file is in the same directory

print(f"MuJoCo version: {mujoco.__version__}")
print(f"Attempting to load mannequin model from: {mannequin_xml_path}")

try:
    # Check if the file exists
    if not os.path.exists(mannequin_xml_path):
        print(f"❌ Error: File '{mannequin_xml_path}' not found!")
        exit(1)
        
    # Load the model
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path(mannequin_xml_path)
    data = mujoco.MjData(model)
    print("✅ Successfully loaded mannequin model!")
    
    # Print some model information
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of geoms: {model.ngeom}")
    print(f"Timestep: {model.opt.timestep}")
    
    # Run the simulation
    print("\nStarting simulation...")
    print("This will open a viewer window. Close the window to end the simulation.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set the camera to a good viewing angle
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 120.0
        viewer.cam.elevation = -20.0
        
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