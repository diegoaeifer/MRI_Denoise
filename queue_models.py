import time
import subprocess
import sys
import os

def run_model(model_name):
    print(f"Starting training for {model_name}...")
    cmd = [
        sys.executable, "src/train.py",
        "--config", "configs/config_custom.yaml",
        "--train_data_dir", r"C:\projetos\IXI\Test",
        "--val_data_dir", r"C:\projetos\IXI\Validation",
        "--model", model_name
    ]
    try:
        # We start the training synchronously inside this loop so the queue waits for completion.
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error training {model_name}: {e}. Skipping to next.")

if __name__ == "__main__":
    # Restart the training queue from 'drunet' forward as requested.
    models_to_train = [
        "drunet", 
        "unet", 
        "restormer", 
        "gsdrunet", 
        "swinir"
    ]
    
    print(f"Restarting training queue for: {', '.join(models_to_train)}")
    for model in models_to_train:
        run_model(model)
        
    print("Sequential execution queue complete.")
