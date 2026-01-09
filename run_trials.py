import subprocess
import time

models = ['nafnet', 'drunet', 'scunet', 'unet']
# models = ['nafnet'] # Test one first if needed

for model in models:
    print(f"==================================================")
    print(f"Starting Trial Run for Model: {model}")
    print(f"==================================================")
    
    cmd = [
        "python", "FMImaging_MRI_Denoise/src/train.py",
        "--config", "FMImaging_MRI_Denoise/configs/config_debug.yaml", # Use debug config but override epochs
        "--model", model,
        "--limit", "1000"
    ]
    
    # We need to ensure config_debug.yaml has 5 epochs. 
    # Or we can rely on whatever is in there. 
    # Currently config_debug.yaml has 1 epoch (from previous step). 
    # Attempting to pass overrides via CLI not supported for yaml fields easily unless I implemented it.
    # But I implemented config loading override.
    # To run 5 epochs, I should probably update config_debug.yaml to 5 epochs first.
    
    subprocess.run(cmd)

print("All trials completed.")
