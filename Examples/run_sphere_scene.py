
import mujoco
import mujoco.viewer
import time
import os
import sys

print(f"MuJoCo version: {mujoco.__version__}")
print(f"Python version: {sys.version}")

# Path to the flex sphere example
flex_xml_path = "flex_sphere.xml"
scene_xml_path = "scene.xml"

# Check if files exist
for file_path in [flex_xml_path, scene_xml_path]:
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found!")
        exit(1)

try:
    # Try to load the plugin directory environment variable
    plugin_dir = os.path.join(os.path.dirname(mujoco.__file__), "plugin")
    if os.path.exists(plugin_dir):
        os.environ["MUJOCO_PLUGIN_DIR"] = plugin_dir
        print(f"✅ Set MUJOCO_PLUGIN_DIR to: {plugin_dir}")
    else:
        print(f"⚠️ Plugin directory not found at {plugin_dir}")
        
        # Try with the binary distribution
        binary_plugin_dir = r"C:\mujoco-3.3.2-windows-x86_64\bin\mujoco_plugin"
        if os.path.exists(binary_plugin_dir):
            os.environ["MUJOCO_PLUGIN_DIR"] = binary_plugin_dir
            print(f"✅ Set MUJOCO_PLUGIN_DIR to: {binary_plugin_dir}")
    
    # Load the model
    print("\nLoading flex sphere model...")
    model = mujoco.MjModel.from_xml_path(flex_xml_path)
    data = mujoco.MjData(model)
    print("✅ Successfully loaded the model!")
    
    # Check for flex components
    try:
        has_flex = hasattr(model, 'nflex') and model.nflex > 0
        if has_flex:
            print(f"✅ Model has {model.nflex} flex components!")
        else:
            print("⚠️ No flex components detected in the model structure.")
            print("This might mean flex isn't supported or wasn't properly loaded.")
    except:
        print("⚠️ Could not check for flex components in model structure.")
    
    # Try to find flex-related attributes
    flex_attrs = [attr for attr in dir(model) if 'flex' in attr.lower()]
    if flex_attrs:
        print(f"Found flex-related attributes: {flex_attrs}")
    else:
        print("No flex-related attributes found in model.")
    
    # Print some model information
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of geoms: {model.ngeom}")
    print(f"Timestep: {model.opt.timestep}")
    
    # Run the simulation
    print("\nStarting simulation...")
    print("This will open a viewer window. Close the window to end the simulation.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera view
        viewer.cam.distance = 5.0
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
    print(f"\n❌ Error: {e}")
    
    # Provide more detailed error information
    import traceback
    traceback.print_exc()
    
    print("\nPossible issues:")
    print("1. Flex element might not be supported in your MuJoCo version")
    print("2. Plugin loading issue - check if elasticity plugin is available")
    print("3. XML syntax error or compatibility issue")
    
    # Check specifically for schema violation
    if "Schema violation" in str(e) and "flexcomp" in str(e):
        print("\nThis error confirms that the 'flexcomp' element is not recognized by your MuJoCo version.")
        print("Options:")
        print("1. Use a different approach (ball-joint or buoyancy) for your inflatable foot")
        print("2. Try a different MuJoCo version (e.g., build from source)")
        print("3. Use the C API directly for more control over plugin loading")