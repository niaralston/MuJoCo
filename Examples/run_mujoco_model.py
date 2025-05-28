import mujoco
import mujoco.viewer
import time
import os
import sys

def main():
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Python version: {sys.version}")
    
    # Look for XML files in both current directory and Examples subdirectory
    xml_files = []
    
    # Check current directory
    current_dir_files = [f for f in os.listdir('.') if f.endswith('.xml')]
    for file in current_dir_files:
        xml_files.append(file)
    
    # Check Examples subdirectory if it exists
    examples_dir = os.path.join('.', 'Examples')
    if os.path.exists(examples_dir) and os.path.isdir(examples_dir):
        examples_files = [os.path.join('Examples', f) for f in os.listdir(examples_dir) if f.endswith('.xml')]
        for file in examples_files:
            xml_files.append(file)
    
    if not xml_files:
        print("No XML files found in the current directory or Examples subdirectory!")
        return
    
    # Display available XML files with numbers
    print("\nAvailable MuJoCo XML files:")
    for i, file in enumerate(xml_files, 1):
        print(f"{i}. {file}")
    
    # Get user selection
    while True:
        try:
            choice = input("\nEnter the number or name of the file you want to run: ")
            
            # Check if input is a number
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(xml_files):
                    model_path = xml_files[index]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(xml_files)}")
            # Check if input is a filename
            elif choice in xml_files:
                model_path = choice
                break
            else:
                # Check if user entered filename without path
                matching_files = [f for f in xml_files if f.endswith('/' + choice) or f == choice]
                if matching_files:
                    model_path = matching_files[0]
                    break
                
                # Check if user entered filename without .xml extension
                if choice + ".xml" in xml_files:
                    model_path = choice + ".xml"
                    break
                
                # Check if user entered filename without Examples/ prefix
                examples_match = os.path.join('Examples', choice)
                if examples_match in xml_files:
                    model_path = examples_match
                    break
                
                # Check if user entered filename without Examples/ prefix and .xml extension
                examples_match_ext = os.path.join('Examples', choice + ".xml")
                if examples_match_ext in xml_files:
                    model_path = examples_match_ext
                    break
                
                print("Invalid selection. Please enter a valid number or filename.")
        except ValueError:
            print("Please enter a valid number or filename.")
    
    print(f"Loading model from: {model_path}")
    
    # Check if file exists (should always be true at this point)
    if not os.path.exists(model_path):
        print(f"❌ Error: File '{model_path}' not found!")
        return
    
    try:
        # Try to set plugin directory
        plugin_dir = os.path.join(os.path.dirname(mujoco.__file__), "plugin")
        if os.path.exists(plugin_dir):
            os.environ["MUJOCO_PLUGIN_DIR"] = plugin_dir
            print(f"✅ Set MUJOCO_PLUGIN_DIR to: {plugin_dir}")
        else:
            binary_plugin_dir = r"C:\mujoco-3.3.2-windows-x86_64\bin\mujoco_plugin"
            if os.path.exists(binary_plugin_dir):
                os.environ["MUJOCO_PLUGIN_DIR"] = binary_plugin_dir
                print(f"✅ Set MUJOCO_PLUGIN_DIR to: {binary_plugin_dir}")
        
        # Load the model
        print(f"\nLoading model from {model_path}...")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print("✅ Successfully loaded the model!")
        
        # Print model information
        print(f"Number of bodies: {model.nbody}")
        print(f"Number of geoms: {model.ngeom}")
        print(f"Timestep: {model.opt.timestep}")
        
        # Check for flex components
        try:
            flex_elements = hasattr(model, 'nflex') and model.nflex > 0
            if flex_elements:
                print(f"✅ Model has {model.nflex} flex elements!")
        except:
            pass
        
        # Ask for simulation time
        try:
            sim_time = int(input("\nEnter simulation time in seconds (default: 60): ") or "60")
        except ValueError:
            sim_time = 60
            print("Invalid input. Using default simulation time of 60 seconds.")
        
        # Run the simulation
        print("\nStarting simulation...")
        print("This will open a viewer window. Close the window to end the simulation.")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera to a reasonable distance
            viewer.cam.distance = 5.0
            
            start_time = time.time()
            
            while viewer.is_running() and (time.time() - start_time) < sim_time:
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()