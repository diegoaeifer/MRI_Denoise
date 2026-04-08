import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import sys
from tqdm import tqdm
import logging

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    class SummaryWriter:
        def __init__(self, log_dir=None): pass
        def add_scalar(self, tag, scalar_value, global_step=None): pass

import piq
import pyiqa

from data.loader import DICOMLoader
from data.dataset import MRI_DICOM_Dataset, collate_fn
from models.factory import get_model
from losses.composite import CompositeLoss
from utils.metrics import calculate_roi_snr

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(config_path, args=None):
    # Load Configs
    root_conf = "configs"
    # Prioritize checking provided config_path
    with open(os.path.join(root_conf, "config_train.yaml")) as f: c_train = yaml.safe_load(f)
    with open(os.path.join(root_conf, "config_data.yaml")) as f: c_data = yaml.safe_load(f)
    with open(os.path.join(root_conf, "config_model.yaml")) as f: c_model = yaml.safe_load(f)
    
    # Merge defaults first
    config = {**c_train, **c_data, **c_model}

    # Now override with CLI provided config
    if config_path and os.path.exists(config_path):
         logger.info(f"Overriding defaults with config: {config_path}")
         with open(config_path) as f:
             custom_conf = yaml.safe_load(f)
         for key, value in custom_conf.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    # 0. Test Mode Overrides
    if args and getattr(args, 'test', False):
        logger.info("TEST MODE ACTIVE")
        config['data']['raw_path'] = r"D:\Diego trabalho\Trainer MRI\FMImaging_MRI_Denoise\data\test"
        config['training']['epochs'] = 10
        args.limit = 1000
        logger.info(f"Test overrides: Data path={config['data']['raw_path']}, Epochs={config['training']['epochs']}, Limit={args.limit}")
    
    device = torch.device(f"cuda:{config['training']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    
    # 0.5. Setup Logging Directory & FileHandler EARLY
    import datetime
    model_name = args.model if args and hasattr(args, 'model') else 'drunet'
    run_id = f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_out = "experiments"
    if args and hasattr(args, 'output_dir') and args.output_dir:
         base_out = args.output_dir
         
    log_dir = os.path.join(base_out, "logs", f"run_{run_id}")
    save_dir = os.path.join(base_out, "checkpoints", f"run_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Add FileHandler EARLY so all subsequent logs go to the file
    file_handler = logging.FileHandler(os.path.join(log_dir, "train_log.txt"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Using device: {device}")
    
    # Log FULL Configuration
    logger.info("="*40)
    logger.info("FULL CONFIGURATION")
    logger.info("="*40)
    logger.info(yaml.dump(config, default_flow_style=False))
    logger.info("="*40)
    
    logger.info("----------------------------------------")
    logger.info("Active Augmentations:")
    aug_params = config['data'].get('augmentation', {})
    logger.info(f"  noise_type: {aug_params.get('noise_type', 'gaussian')}")
    logger.info(f"  sigma_min: {aug_params.get('sigma_min')}")
    logger.info(f"  sigma_max: {aug_params.get('sigma_max')}")
    logger.info(f"  noise_grid_size: {aug_params.get('noise_grid_size')}")
    logger.info(f"  flip_prob: {aug_params.get('flip_prob')}")
    logger.info(f"  rotate_prob: {aug_params.get('rotate_prob')}")
    logger.info(f"  affine_prob: {aug_params.get('affine_prob')}")
    logger.info(f"  gamma_prob: {aug_params.get('gamma_prob')}")
    logger.info(f"  gamma_range: {aug_params.get('gamma_range')}")
    logger.info(f"  ghosting_prob: {aug_params.get('ghosting_prob')}")
    logger.info(f"  spike_prob: {aug_params.get('spike_prob')}")
    logger.info(f"  blur_prob: {aug_params.get('blur_prob')}")
    logger.info(f"  motion_prob: {aug_params.get('motion_prob')}")
    logger.info(f"  bias_field_prob: {aug_params.get('bias_field_prob')}")
    logger.info(f"  bias_field_coeffs: {aug_params.get('bias_field_coeffs')}")
    logger.info(f"  anisotropy_prob: {aug_params.get('anisotropy_prob')}")
    logger.info(f"  anisotropy_downsampling: {aug_params.get('anisotropy_downsampling')}")
    logger.info("----------------------------------------")
    
    # PATH OVERRIDES
    if args:
        if hasattr(args, 'data_dir') and args.data_dir:
             logger.info(f"Overriding data path with: {args.data_dir}")
             config['data']['raw_path'] = args.data_dir
             
    # Log Augmentations
    aug_c = config['data']['augmentation']
    logger.info("-" * 40)
    logger.info("Active Augmentations:")
    for k, v in aug_c.items():
        logger.info(f"  {k}: {v}")
    logger.info("-" * 40)
    
    # 1. Data Setup
    limit = args.limit if args and hasattr(args, 'limit') else None
    
    if args and getattr(args, 'train_data_dir', None) and getattr(args, 'val_data_dir', None):
        logger.info(f"Using explicit data directories. Train: {args.train_data_dir}, Val: {args.val_data_dir}")
        from data.nifti_loader import NiftiLoader
        
        train_loader_obj = NiftiLoader(data_path=args.train_data_dir, seed=config['data']['seed'], limit=limit)
        val_loader_obj = NiftiLoader(data_path=args.val_data_dir, seed=config['data']['seed'], limit=limit)
        
        train_files = train_loader_obj.scan_directory()
        val_files = val_loader_obj.scan_directory()
        
    else:
        loader = DICOMLoader(
            data_path=config['data']['raw_path'],
            seed=config['data']['seed'],
            split_ratios=config['data']['split_ratios'],
            limit=limit
        )
        splits = loader.create_splits(output_dir=config['data']['splits_path'])
        train_files = splits['train']
        val_files = splits['val']
    
    train_ds = MRI_DICOM_Dataset(train_files, mode='train', config=config['data'])
    val_ds = MRI_DICOM_Dataset(val_files, mode='val', config=config['data'])
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config['training']['num_workers'], collate_fn=collate_fn)
    
    # 2. Model Setup
    # 2. Model Setup
    model_name = args.model if args and hasattr(args, 'model') else 'drunet'
    logger.info(f"Initializing model: {model_name}")
    model_config = config['models'] # Extract model config for clarity
    model = get_model(model_name, model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Summary: {total_params:,} total parameters, {trainable_params:,} trainable parameters")
    
    # Log Model Architecture
    logger.info("="*40)
    logger.info("MODEL ARCHITECTURE")
    logger.info("="*40)
    logger.info(str(model))
    logger.info("="*40)
    
    # 3. Loss & Opt
    criterion = CompositeLoss(config['losses']).to(device)
    
    lr = float(config['training']['learning_rate'])
    if config['training']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    # Scheduler Setup
    scheduler_name = config['training'].get('scheduler', 'CosineAnnealing')
    if scheduler_name == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    elif scheduler_name == 'CosineAnnealing2':
        # T_max=50, eta_min=1e-6 (User Request)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    elif scheduler_name == 'ReduceLROnPlateau':
        # patience=3, factor=0.5 (User Request)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        

        
    # 4. Loop
    # run_id is now created earlier for logging setup
    
    # Update logs and checkpoint paths
    # base_out, log_dir, save_dir are now created earlier for logging setup
    
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        logger.warning("TensorBoard not available. Logging to CLI only.")
        writer = SummaryWriter() # Uses dummy class

    logger.info(f"Experiment initialized. Checkpoints: {save_dir}, Logs: {log_dir}")
    
    best_loss = float('inf')
    negative_psnr_counter = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch in loop:
            if batch is None: continue
            
            inputs = batch['input'].to(device) # (B, 2, H, W)
            targets = batch['target'].to(device) # (B, 1, H, W)
            
            optimizer.zero_grad()
            
            preds = model(inputs)
            
            # Pass model & input for SURE
            loss, loss_dict = criterion(preds, targets, model=model, input_tensor=inputs)
            
            loss.backward()
            
            # --- STABILITY FIX: Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        if len(train_loader) > 0:
            avg_train_loss = train_loss / len(train_loader)
        else:
            avg_train_loss = 0
            
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        
        model.eval()
        val_loss = 0
        # Initialize validation metrics tracking
        val_metrics = {'ms_ssim': 0.0, 'haarpsi': 0.0, 'psnr': 0.0}
        
        # Validate metrics tracking
        val_metrics = {'ms_ssim': 0.0, 'haarpsi': 0.0, 'psnr': 0.0}
        
        # We use 'piq' for high-quality MRI metrics that support 128x128 patches
        # 4 scales are used for MS-SSIM to avoid kernel size crashes (128 / 2^4 = 8 < 11)
        ms_ssim_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363]).to(device)
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                preds = model(inputs)
                loss, loss_dict = criterion(preds, targets, model=model, input_tensor=inputs)
                val_loss += loss.item()
                
                # Use piq for validation metrics
                preds_clamped = torch.clamp(preds, 0, 1)
                
                # --- STABILITY FIX: NaN Robustness ---
                if torch.isnan(preds_clamped).any() or torch.isnan(targets).any():
                    logger.warning(f"NaN detected in predictions or targets during Epoch {epoch+1} validation. Skipping metrics for this batch.")
                else:
                    try:
                        val_metrics['ms_ssim'] += piq.multi_scale_ssim(preds_clamped, targets, data_range=1.0, scale_weights=ms_ssim_weights).item()
                        val_metrics['haarpsi'] += piq.haarpsi(preds_clamped, targets, data_range=1.0).item()
                    except (AssertionError, RuntimeError) as e:
                        logger.error(f"Metric calculation failed: {e}. Values clamped range: {preds_clamped.min().item():.4f} - {preds_clamped.max().item():.4f}")
                
                if 'psnr' in loss_dict:
                    val_metrics['psnr'] += loss_dict['psnr'].item()
                
        if len(val_loader) > 0:
            avg_val_loss = val_loss / len(val_loader)
            avg_ms_ssim = val_metrics['ms_ssim'] / len(val_loader)
            avg_haarpsi = val_metrics['haarpsi'] / len(val_loader)
            avg_psnr = val_metrics['psnr'] / len(val_loader)
        else:
            avg_val_loss = 0
            avg_ms_ssim = 0
            avg_haarpsi = 0
            avg_psnr = 0
            
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metrics/Val_PSNR', avg_psnr, epoch)
        writer.add_scalar('Metrics/Val_MS_SSIM', avg_ms_ssim, epoch)
        writer.add_scalar('Metrics/Val_HAARPSI', avg_haarpsi, epoch)
        
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} [PSNR: {avg_psnr:.2f}, MS-SSIM: {avg_ms_ssim:.4f}, HAARpsi: {avg_haarpsi:.4f}]")

        
        # Early Stopping for Negative PSNR
        if avg_psnr < 0:
            negative_psnr_counter += 1
            if negative_psnr_counter >= 3:
                logger.error(f"Training ABORTED: PSNR remained negative for {negative_psnr_counter} consecutive epochs. Likely divergence.")
                sys.exit(1)
        else:
            negative_psnr_counter = 0
        
        # Stepping Scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        # Log images
        log_freq = 1 if (args and getattr(args, 'test', False)) else 5
        if (epoch + 1) % log_freq == 0:
             log_sample_images(model, val_loader, device, epoch, save_dir, writer, num_samples=10)

        # Checkpointing
        if avg_val_loss < best_loss: # Still saving based on Loss (which is Charbonnier now)
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(save_dir, f"{model_name}_best_model.pth"))
            
def log_sample_images(model, loader, device, epoch, save_dir, writer, num_samples=10):
    model.eval()
    
    # Collect samples randomly
    dataset = loader.dataset
    total_len = len(dataset)
    
    # Random indices
    import random
    indices = random.sample(range(total_len), min(num_samples, total_len))
    
    inputs_list = [] # Noisy
    targets_list = [] # Original
    sigmas_list = [] # Noise Map
    preds_list = [] # Estimated
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx] # Returns dict
            
            inp = sample['input'].unsqueeze(0).to(device) # (1, 2, H, W)
            target = sample['target'].unsqueeze(0).to(device) # (1, 1, H, W)

            pred = model(inp)
            
            # Extract components
            noisy = inp[:, 0:1, :, :]
            sigma_map = inp[:, 1:2, :, :]
            
            # Append (cpu)
            inputs_list.append(noisy.cpu()) # Noisy
            targets_list.append(target.cpu()) # Original
            sigmas_list.append(sigma_map.cpu()) # Sigma Map
            preds_list.append(pred.cpu())

    if not inputs_list:
        return

    # Concatenate
    noisy_imgs = torch.cat(inputs_list, dim=0)
    target_imgs = torch.cat(targets_list, dim=0)
    sigma_imgs = torch.cat(sigmas_list, dim=0)
    pred_imgs = torch.cat(preds_list, dim=0)
    
    # Create grid: Rows = Samples, Cols = [Noisy, Original, Sigma, Pred]
    combined = []
    for i in range(len(noisy_imgs)):
        combined.append(noisy_imgs[i])   # 1. Noisy
        combined.append(target_imgs[i])  # 2. Original
        combined.append(sigma_imgs[i])   # 3. Noise Map
        combined.append(pred_imgs[i])    # 4. Estimated
        
    combined_tensor = torch.stack(combined)
    
    import torchvision
    grid = torchvision.utils.make_grid(combined_tensor, nrow=4, normalize=True, scale_each=True)
    
    # Save to disk
    save_path = os.path.join(save_dir, "samples")
    os.makedirs(save_path, exist_ok=True)
    torchvision.utils.save_image(grid, os.path.join(save_path, f"epoch_{epoch+1}_samples.png"))
    
    # Log to TensorBoard
    writer.add_image('Visuals/Samples', grid, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_train.yaml')
    parser.add_argument('--model', type=str, default='drunet', help='Model architecture to train (drunet, nafnet, scunet, unet)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images for debugging')
    parser.add_argument('--test', action='store_true', help='Run in test mode with specific data and overrides')
    parser.add_argument('--data_dir', type=str, default=None, help='Override base data directory')
    parser.add_argument('--train_data_dir', type=str, default=None, help='Specific training data directory for NIFTI workflow')
    parser.add_argument('--val_data_dir', type=str, default=None, help='Specific validation data directory for NIFTI workflow')
    parser.add_argument('--output_dir', type=str, default=None, help='Override base output directory for logs/checkpoints')
    args = parser.parse_args()
    
    train(args.config, args)
