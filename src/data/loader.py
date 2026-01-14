import os
import random
import json
from pathlib import Path
import pydicom
from collections import defaultdict
# from sklearn.model_selection import train_test_split
import logging

def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

configure_logging()
logger = logging.getLogger(__name__)

class DICOMLoader:
    def __init__(self, data_path, seed=42, split_ratios=None, limit=None, cache=True):
        self.data_path = Path(data_path)
        self.seed = seed
        self.split_ratios = split_ratios or {'train': 0.8, 'test': 0.1, 'val': 0.1}
        self.limit = limit
        self.cache = cache
        random.seed(self.seed)
        
    def scan_directory(self):
        """Recursively scans for DICOM files, with caching support."""
        cache_file = self.data_path / "dicom_file_cache.json"
        
        # Try loading from cache first
        if self.cache and cache_file.exists():
            logger.info(f"Loading DICOM list from cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
                # Convert lists back to sets for patient_registry
                patient_registry = {pid: set(sids) for pid, sids in cached_data['patient_registry'].items()}
                series_registry = cached_data['series_registry']
                
                logger.info(f"Loaded {len(patient_registry)} unique patients from cache.")
                return patient_registry, series_registry
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rescanning directory.")

        logger.info(f"Scanning directory: {self.data_path}")
        series_registry = defaultdict(list)
        patient_registry = defaultdict(set)
        
        count = 0
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.lower().endswith('.dcm') or '.' not in file: 
                    file_path = os.path.join(root, file)
                    try:
                        # Read only specific tags to be fast
                        ds = pydicom.dcmread(file_path, stop_before_pixels=True, specific_tags=['PatientID', 'SeriesInstanceUID'])
                        
                        pid = str(ds.get('PatientID', 'Unknown'))
                        sid = str(ds.get('SeriesInstanceUID', 'Unknown'))
                        
                        if sid == 'Unknown':
                            continue
                            
                        # Store file path under the series
                        series_registry[sid].append(file_path)
                        # Link series to patient
                        patient_registry[pid].add(sid)
                        
                        count += 1
                        
                    except Exception as e:
                        continue
                        
        logger.info(f"Found {len(patient_registry)} unique patients and {len(series_registry)} unique series.")
        
        # Save to cache
        if self.cache:
            try:
                # Convert sets to lists for JSON serialization
                serializable_registry = {pid: list(sids) for pid, sids in patient_registry.items()}
                cache_data = {
                    'patient_registry': serializable_registry,
                    'series_registry': series_registry
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=4)
                logger.info(f"Saved DICOM file cache to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
                
        return patient_registry, series_registry

    def create_splits(self, output_dir=None):
        """Creates Patient-wise splits."""
        patient_registry, series_registry = self.scan_directory()
        patient_ids = list(patient_registry.keys())
        
        # Shuffle patients
        random.shuffle(patient_ids)
        
        n_total = len(patient_ids)
        if n_total == 0:
            raise ValueError("No DICOM data found in the specified directory.")

        if self.limit:
            # If limiting, we want to limit total files but keep patient structure distribution if possible.
            # But simplest is to create splits normally then truncate? 
            # OR shuffle patients -> collect -> truncate.
            # But truncating might break 80/10/10 ratio.
            
            # Alternative: Collect ALL files first, then shuffle and split? 
            # But the requirement is patient-wise split.
            
            # Strategy: Split patients first, then limit each split proportionally.
            pass

        n_train = int(n_total * self.split_ratios['train'])
        n_test = int(n_total * self.split_ratios['test'])
        
        # Ensure at least 1 patient in train/val if low count
        if n_total > 1 and n_train == 0: n_train = 1
        
        train_pids = patient_ids[:n_train]
        test_pids = patient_ids[n_train:n_train + n_test]
        val_pids = patient_ids[n_train + n_test:]
        
        # Using a fallback if val/test are empty due to rounding
        if not val_pids and len(patient_ids) > 1:
             # Steal one from train or test
             if test_pids: 
                 val_pids = [test_pids.pop()]
             elif len(train_pids) > 1:
                 val_pids = [train_pids.pop()]

        splits = {
            'train': self._collect_files(train_pids, patient_registry, series_registry),
            'test': self._collect_files(test_pids, patient_registry, series_registry),
            'val': self._collect_files(val_pids, patient_registry, series_registry)
        }
        
        if self.limit:
            # Calculate limits per split
            l_train = int(self.limit * self.split_ratios['train'])
            l_test = int(self.limit * self.split_ratios['test'])
            l_val = int(self.limit * self.split_ratios['val'])
            
            # Since _collect_files returns sorted lists, we should shuffle before limiting to avoid bias
            random.shuffle(splits['train'])
            random.shuffle(splits['test'])
            random.shuffle(splits['val'])
            
            splits['train'] = splits['train'][:l_train]
            splits['test'] = splits['test'][:l_test]
            splits['val'] = splits['val'][:l_val]
            
            logger.info(f"Applied limit {self.limit} -> Train: {len(splits['train'])}, Test: {len(splits['test'])}, Val: {len(splits['val'])}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for split_name, files in splits.items():
                out_path = os.path.join(output_dir, f"{split_name}_files.json")
                with open(out_path, 'w') as f:
                    json.dump(files, f, indent=4)
                logger.info(f"Saved {split_name} split with {len(files)} files to {out_path}")
                
        return splits

    def _collect_files(self, patient_ids, patient_registry, series_registry):
        file_list = []
        for pid in patient_ids:
            sids = patient_registry[pid]
            for sid in sids:
                series_files = series_registry[sid]
                # Sort files to ensure slice order (dataset will handle actual collection, but good to be deterministic)
                series_files.sort() 
                file_list.extend(series_files)
        return file_list

if __name__ == "__main__":
    # Test execution
    import yaml
    config_path = "FMImaging_MRI_Denoise/configs/config_data.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        loader = DICOMLoader(
            data_path=config['data']['raw_path'],
            seed=config['data']['seed'],
            split_ratios=config['data']['split_ratios']
        )
        loader.create_splits(output_dir=config['data']['splits_path'])
    else:
        print("Config file not found, skipping standalone test.")
