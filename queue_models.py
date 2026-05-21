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
    import os
    os.makedirs("experiments/logs", exist_ok=True)
    log_file = f"experiments/logs/{model_name}_training_queue.log"
    print(f"Training log for {model_name} saving to {log_file}")
    
    with open(log_file, "w") as f:
        try:
            # We start the training synchronously inside this loop so the queue waits for completion.
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error training {model_name}: {e}. Skipping to next.")

if __name__ == "__main__":
    # Restart the training queue from 'drunet' forward as requested.
    models_to_train = [
        "drunet",
        "unet",
        "restormer",
        "gsdrunet"
    ]
    
    print(f"Restarting training queue for: {', '.join(models_to_train)}")
    for model in models_to_train:
        run_model(model)
        
    print("Sequential execution queue complete.")
