import os
from pathlib import Path
import logging
import random

logger = logging.getLogger(__name__)

class NiftiLoader:
    def __init__(self, data_path, seed=42, limit=None):
        self.data_path = Path(data_path)
        self.seed = seed
        self.limit = limit
        random.seed(self.seed)
        
    def scan_directory(self):
        """Recursively scans for Nifti files."""
        logger.info(f"Scanning directory for NIFTI files: {self.data_path}")
        
        file_list = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.lower().endswith('.nii') or file.lower().endswith('.nii.gz'):
                    file_list.append(os.path.join(root, file))
                    
        logger.info(f"Found {len(file_list)} NIFTI files.")
        
        # Sort to be deterministic
        file_list.sort()
        
        # Optional Shuffle before limit if desired, but for standard train/val dirs keeping it simple.
        if self.limit and len(file_list) > self.limit:
            random.shuffle(file_list)
            file_list = file_list[:self.limit]
            logger.info(f"Limited file list to {self.limit} NIFTI files.")
            
        return file_list
