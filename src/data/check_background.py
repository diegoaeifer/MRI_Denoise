import os
import pydicom
import numpy as np
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_batch(file_paths, threshold_range=1000):
    """
    Processes a list of files and returns those matching the criteria.
    """
    problematic = []
    for fp in file_paths:
        try:
            # Use force=True to handle missing headers noted in previous logs
            ds = pydicom.dcmread(fp, force=True)
            if not hasattr(ds, 'PixelData'):
                continue
                
            image = ds.pixel_array.astype(np.float32)
            
            p1, p99 = np.percentile(image, [1, 99])
            dynamic_range = p99 - p1
            
            if dynamic_range < threshold_range:
                problematic.append((fp, dynamic_range))
                
        except Exception as e:
            # logger.error(f"Error reading {fp}: {e}")
            pass
    return problematic

def check_background_batched(data_path, threshold_range=1000, batch_size=500, auto_delete=False, workers=None):
    logger.info(f"Scanning {data_path} (Batch Size: {batch_size}, Auto-Delete: {auto_delete}, Workers: {workers})")
    
    all_dcm_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                all_dcm_files.append(os.path.join(root, file))
    
    total_files = len(all_dcm_files)
    logger.info(f"Total DICOM files found: {total_files}")
    
    problematic_all = []
    deleted_count = 0
    
    batches = [all_dcm_files[i : i + batch_size] for i in range(0, total_files, batch_size)]

    # Process in batches in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_batch, batch, threshold_range): batch for batch in batches}
        
        with tqdm(total=len(batches)) as pbar:
            for future in as_completed(futures):
                found_in_batch = future.result()

                if found_in_batch:
                    problematic_all.extend(found_in_batch)

                    if auto_delete:
                        for fp, _ in found_in_batch:
                            try:
                                os.remove(fp)
                                deleted_count += 1
                            except Exception as e:
                                logger.error(f"Failed to delete {fp}: {e}")

                pbar.update(1)
        
    logger.info(f"Finished. Found {len(problematic_all)} problematic files.")
    if auto_delete:
        logger.info(f"Successfully deleted {deleted_count} files.")
        
    if problematic_all:
        problematic_all.sort(key=lambda x: x[1]) # Lowest range first
        
        # Save report
        with open("background_report.txt", "w") as f:
            f.write(f"Background Report (Range Threshold: {threshold_range}, Auto-Deleted: {auto_delete})\n")
            for path, dr in problematic_all:
                status = "Deleted" if auto_delete else "Flagged"
                f.write(f"[{status}] {path}: Range {dr:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=1000, help='Dynamic range threshold')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--delete', action='store_true', help='Delete without confirmation')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    args = parser.parse_args()
    
    check_background_batched(args.data_path, args.threshold, args.batch_size, args.delete, args.workers)
