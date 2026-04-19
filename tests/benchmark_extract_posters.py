import time
import os
import sys
from unittest.mock import patch, MagicMock

# Mock pymupdf before importing extract_posters
sys.modules['pymupdf'] = MagicMock()

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import extract_posters

def mock_extract_from_pdf(filepath):
    # Simulate processing time (0.05 seconds per file)
    time.sleep(0.05)
    return ["MockModel"], ["MockLoss"], [("MockParam", "Value")]

def run_benchmark(num_files=20):
    print(f"Benchmarking with {num_files} files...")

    # Create dummy files list
    dummy_files = [f"poster_{i}.pdf" for i in range(num_files)]

    # We need to patch extract_posters.extract_from_pdf directly
    # and also handle the file I/O

    with patch('os.path.exists', return_value=True), \
         patch('os.listdir', return_value=dummy_files), \
         patch('os.makedirs'), \
         patch('builtins.open', MagicMock()), \
         patch('extract_posters.extract_from_pdf', side_effect=mock_extract_from_pdf):

        start_time = time.time()
        extract_posters.main()
        end_time = time.time()

    duration = end_time - start_time
    print(f"Total time taken: {duration:.4f} seconds")
    return duration

if __name__ == "__main__":
    run_benchmark()
