import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import sys
import logging
import datetime

from data.loader import DICOMLoader
from data.dataset import MRI_DICOM_Dataset, collate_fn
from models.factory import get_model
from losses.composite import CompositeLoss
from trainer import Trainer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_config(config, override):
    """
    Recursively update a dictionary.
    """
    for k, v in override.items():
        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
            update_config(config[k], v)
        else:
            config[k] = v

def main(args):
    # Load Configs
    root_conf = "configs"
    # Load defaults
    config = {}
    for cfg_name in ["config_train.yaml", "config_data.yaml", "config_model.yaml"]:
        path = os.path.join(root_conf, cfg_name)
        if os.path.exists(path):
            with open(path) as f:
                config.update(yaml.safe_load(f))

    # Override with CLI provided config
    if args.config and os.path.exists(args.config):
         logger.info(f"Overriding defaults with config: {args.config}")
         with open(args.config) as f:
             custom_conf = yaml.safe_load(f)
             update_config(config, custom_conf)

    # 0. Test Mode Overrides
    if args.test:
        logger.info("TEST MODE ACTIVE")
        config['data']['raw_path'] = os.path.join("data", "test")
        config['training']['epochs'] = 5
        args.limit = 100
    
    # Path Overrides from CLI
    if args.data_dir: config['data']['raw_path'] = args.data_dir
    if args.output_dir: config['training']['output_dir'] = args.output_dir
    
    device = torch.device(f"cuda:{config['training']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    
    # ⚡ Bolt Optimization: Enable cudnn benchmark for faster convolutions with fixed input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


    # Create Run ID
    run_id = f"{args.model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 1. Data Setup
    if args.train_data_dir and args.val_data_dir:
        logger.info("Using explicit NIFTI data directories.")
        from data.nifti_loader import NiftiLoader
        train_loader_obj = NiftiLoader(data_path=args.train_data_dir, seed=config['data']['seed'], limit=args.limit)
        val_loader_obj = NiftiLoader(data_path=args.val_data_dir, seed=config['data']['seed'], limit=args.limit)
        train_files = train_loader_obj.scan_directory()
        val_files = val_loader_obj.scan_directory()
    else:
        loader = DICOMLoader(data_path=config['data']['raw_path'], seed=config['data']['seed'], 
                            split_ratios=config['data']['split_ratios'], limit=args.limit)
        splits = loader.create_splits(output_dir=config['data']['splits_path'])
        train_files = splits['train']
        val_files = splits['val']
    
    train_ds = MRI_DICOM_Dataset(train_files, mode='train', config=config['data'])
    val_ds = MRI_DICOM_Dataset(val_files, mode='val', config=config['data'])
    
    # ⚡ Bolt Optimization: Enable pin_memory for faster host-to-device transfers
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, 
                             num_workers=config['training']['num_workers'], collate_fn=collate_fn,
                             pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                           num_workers=config['training']['num_workers'], collate_fn=collate_fn,
                           pin_memory=torch.cuda.is_available())
    
    # 2. Model Setup
    model = get_model(args.model, config['models']).to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    
    # 3. Loss, Opt, Scheduler
    criterion = CompositeLoss(config['losses']).to(device)
    
    lr = float(config['training']['learning_rate'])
    optimizer = optim.AdamW(model.parameters(), lr=lr) if config['training']['optimizer'] == 'AdamW' else optim.Adam(model.parameters(), lr=lr)
    
    T_max = config['training'].get('epochs', 100)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    
    # 4. Trainer Initialization
    trainer = Trainer(model, config, device, run_id=run_id)
    trainer.prepare(criterion, optimizer, scheduler)
    
    # Add File Log
    file_handler = logging.FileHandler(os.path.join(trainer.log_dir, "train.log"))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting Training: {run_id}")
    for epoch in range(config['training']['epochs']):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss, val_metrics = trainer.validate(val_loader, epoch)
        
        # Schedulers
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
            
        # Logging & Checkpointing
        trainer.writer.add_scalar('Loss/Train', train_loss, epoch)
        trainer.writer.add_scalar('Loss/Val', val_loss, epoch)
        for k, v in val_metrics.items():
            trainer.writer.add_scalar(f'Metrics/Val_{k.upper()}', v, epoch)
            
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PSNR: {val_metrics['psnr']:.2f}")
        
        # Save checkpoint
        is_best = val_loss < trainer.best_loss
        if is_best: trainer.best_loss = val_loss
        trainer.save_checkpoint(epoch, val_loss, is_best=is_best)
        
        # Early Stopping: abortar si PSNR es negativo por varias epocas consecutivas
        if trainer.check_divergence(val_metrics['psnr']):
            logger.error("Exiting due to training divergence.")
            sys.exit(1)
        
        # Visuals
        if (epoch + 1) % config['training'].get('log_visuals_freq', 5) == 0 or args.test:
            trainer.log_visuals(val_loader, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_train.yaml', help='Path to training config YAML')
    parser.add_argument('--model', type=str, default='drunet', help='Model architecture to train (drunet, nafnet, scunet, unet, drunet_pretrained, etc.)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images for debugging')
    parser.add_argument('--test', action='store_true', help='Run in test mode with specific data and overrides')
    parser.add_argument('--data_dir', type=str, default=None, help='Override base data directory')
    parser.add_argument('--train_data_dir', type=str, default=None, help='Specific training data directory for NIFTI workflow')
    parser.add_argument('--val_data_dir', type=str, default=None, help='Specific validation data directory for NIFTI workflow')
    parser.add_argument('--output_dir', type=str, default=None, help='Override base output directory for logs/checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load before training')
    args = parser.parse_args()
    main(args)
