import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
import datetime
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
import yaml
import torchvision
import piq

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    class SummaryWriter:
        def __init__(self, log_dir=None): pass
        def add_scalar(self, tag, scalar_value, global_step=None): pass
        def add_image(self, tag, img_tensor, global_step=None): pass

logger = logging.getLogger(__name__)

class Trainer:
    """
    Decoupled Trainer class for MRI Denoising.
    Handles training loops, validation, checkpointing, and logging.
    """
    def __init__(self, model, config, device, run_id=None):
        self.model = model
        self.config = config

        # Determine model size and log information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Model Type: {model.__class__.__name__}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")

        if 'losses' in config:
            logger.info("Active Losses:")
            for k, v in config['losses']['weights'].items():
                if v > 0:
                    logger.info(f"  {k}: {v}")
        self.device = device
        self.run_id = run_id or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.base_out = config['training'].get('output_dir', 'experiments')
        self.log_dir = os.path.join(self.base_out, "logs", self.run_id)
        self.save_dir = os.path.join(self.base_out, "checkpoints", self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = SummaryWriter()

        self.best_loss = float('inf')
        self.start_epoch = 0
        self._neg_psnr_count = 0

        # Metric calculators
        self.dists_calc = piq.DISTS().to(self.device)

    def prepare(self, criterion, optimizer, scheduler=None):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def check_divergence(self, avg_psnr, threshold=3):
        """
        Detecta divergencia monitoreando PSNR negativo consecutivo.
        Retorna True si el entrenamiento debe abortarse.
        """
        if avg_psnr < 0:
            self._neg_psnr_count += 1
            if self._neg_psnr_count >= threshold:
                logger.error(
                    f"Training ABORTED: PSNR remained negative for "
                    f"{self._neg_psnr_count} consecutive epochs. Likely divergence."
                )
                return True
        else:
            self._neg_psnr_count = 0
        return False

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        
        for batch in loop:
            if batch is None: continue
            
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(inputs)
            
            loss, loss_dict = self.criterion(preds, targets, model=self.model, input_tensor=inputs)
            loss.backward()
            
            # Log Gradient Norm for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.writer.add_scalar('Stability/GradNorm', grad_norm, epoch * len(train_loader) + loop.n)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), gn=f"{grad_norm:.2f}")
            
        return total_loss / len(train_loader) if len(train_loader) > 0 else 0

    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        val_metrics = {'ms_ssim': 0.0, 'psnr': 0.0, 'haarpsi': 0.0, 'dists': 0.0}
        
        # MRI-specific MS-SSIM weights for 128x128
        ms_ssim_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363]).to(self.device)
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                preds = self.model(inputs)
                loss, loss_dict = self.criterion(preds, targets, model=self.model, input_tensor=inputs)
                total_loss += loss.item()
                
                preds_clamped = torch.clamp(preds, 0, 1)
                
                # Calculate metrics independently of losses chosen
                if not (torch.isnan(preds_clamped).any() or torch.isnan(targets).any()):
                    try:
                        val_metrics['ms_ssim'] += piq.multi_scale_ssim(preds_clamped, targets, data_range=1.0, scale_weights=ms_ssim_weights).item()
                        val_metrics['haarpsi'] += piq.haarpsi(preds_clamped, targets, data_range=1.0).item()
                        val_metrics['psnr'] += piq.psnr(preds_clamped, targets, data_range=1.0).item()

                        # DISTS expects 3-channel input
                        preds_3c = preds_clamped.repeat(1, 3, 1, 1)
                        targets_3c = targets.repeat(1, 3, 1, 1)
                        val_metrics['dists'] += self.dists_calc(preds_3c, targets_3c).item()
                    except (AssertionError, RuntimeError) as e:
                        logger.warning(f"Error calculating validation metric: {e}")
                        pass

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        for k in val_metrics:
            val_metrics[k] /= len(val_loader) if len(val_loader) > 0 else 1
            
        return avg_loss, val_metrics

    def save_checkpoint(self, epoch, current_loss, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': current_loss,
            'config': self.config
        }
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch+1}.pth"
        path = os.path.join(self.save_dir, filename)
        torch.save(state, path)
        if is_best:
            logger.info(f"New best model saved to {path} (Loss: {current_loss:.6f})")

    def log_visuals(self, loader, epoch, num_samples=8):
        self.model.eval()
        dataset = loader.dataset
        import random
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        combined = []
        with torch.no_grad():
            for idx in indices:
                sample = dataset[idx]
                inp = sample['input'].unsqueeze(0).to(self.device)
                target = sample['target'].unsqueeze(0).to(self.device)
                pred = self.model(inp)
                
                # Cols = [Noisy, Target, Pred, Sigma, Diff]
                noisy = inp[:, 0:1, :, :].cpu()
                sigma = inp[:, 1:2, :, :].cpu()
                target = target.cpu()
                pred = pred.cpu()
                diff = torch.abs(target - pred) * 5.0  # Amplificar diff para visibilidad
                
                combined.extend([noisy[0], target[0], pred[0], sigma[0], diff[0]])
        
        grid = torchvision.utils.make_grid(torch.stack(combined), nrow=5, normalize=True, scale_each=True)
        self.writer.add_image('Visuals/EpochSamples', grid, epoch)
        
        save_path = os.path.join(self.save_dir, "samples", f"epoch_{epoch+1}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(grid, save_path)
