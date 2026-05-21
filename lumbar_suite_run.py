import subprocess
import sys
import os
import time
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data Path
DATA_DIR = r"C:\projetos\Lumbar_spine\rsna-2024-lumbar-spine-degenerative-classification\train_images"

# Models to run
experiments = [
    {
        "model": "ram_pretrained",
        "epochs": 25,
        "batch_size": 4,
        "lr": 5e-5,
        "note": "DeepInv RAM foundation model"
    },
    {
        "model": "nafnet_small",
        "epochs": 30,
        "batch_size": 16,
        "lr": 2e-4,
        "note": "NAFNet-small trained from scratch"
    }
]

def run_experiment(exp):
    model_name = exp['model']
    logger.info(f"=== Starting Experiment: {model_name} ===")
    logger.info(f"Config: Epochs={exp['epochs']}, Batch={exp['batch_size']}, LR={exp['lr']}")
    logger.info(f"Note: {exp['note']}")
    
    # Create temp config for overrides
    temp_conf = f"configs/temp_{model_name}.yaml"
    with open(temp_conf, "w") as f:
        f.write(f"training:\n")
        f.write(f"  epochs: {exp['epochs']}\n")
        f.write(f"  batch_size: {exp['batch_size']}\n")
        f.write(f"  learning_rate: {exp['lr']}\n")

    cmd = [
        sys.executable, "src/train.py",
        "--config", temp_conf,
        "--model", model_name,
        "--data_dir", DATA_DIR,
        "--limit", "1000",
        "--output_dir", "experiments/lumbar_suite_retries"
    ]
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = (time.time() - start_time) / 60
        logger.info(f"-> SUCCESS: {model_name} in {elapsed:.1f} min")
    except subprocess.CalledProcessError as e:
        logger.error(f"-> FAILED: {model_name} with exit code {e.returncode}")
        # Continue to next experiment? Yes, to try others.
    
if __name__ == "__main__":
    logger.info("Starting Lumbar Suite Retries...")
    os.makedirs("experiments/lumbar_suite_retries", exist_ok=True)
    
    for exp in experiments:
        run_experiment(exp)
        
    logger.info("All experiments in suite completed.")
