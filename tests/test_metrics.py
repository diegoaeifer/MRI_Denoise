import unittest
import sys
from unittest.mock import MagicMock

# Mock monai to avoid missing dependency during tests
sys.modules['monai'] = MagicMock()
sys.modules['monai.metrics'] = MagicMock()

import numpy as np
import torch
import math

from src.utils.metrics import calculate_roi_snr

class TestMetrics(unittest.TestCase):
    def test_calculate_roi_snr_numpy_2d(self):
        # Create a 100x100 image
        image = np.zeros((100, 100))
        # central ROI is from cy-10 to cy+10, cx-10 to cx+10
        # cy, cx = 50, 50. ROI: 40:60, 40:60
        roi = np.zeros((20, 20))
        roi[:10, :] = 0
        roi[10:, :] = 2
        image[40:60, 40:60] = roi

        snr = calculate_roi_snr(image, box_size=20)
        # mu = 1, sigma = 1 (calculated using np.std which has ddof=0)
        expected_snr = 20 * np.log10(np.abs(1.0 / 1.0) + 1e-8)
        self.assertAlmostEqual(snr, expected_snr, places=5)

    def test_calculate_roi_snr_torch_tensor(self):
        image = torch.zeros((100, 100))
        roi = torch.zeros((20, 20))
        roi[:10, :] = 0
        roi[10:, :] = 2
        image[40:60, 40:60] = roi

        # When passed to the function it gets converted to numpy array first
        # Therefore np.std runs instead of torch.std, so std logic remains same
        snr = calculate_roi_snr(image, box_size=20)
        expected_snr = 20 * np.log10(np.abs(1.0 / 1.0) + 1e-8)
        self.assertAlmostEqual(snr, expected_snr, places=5)

    def test_calculate_roi_snr_zero_sigma(self):
        # constant image -> sigma=0
        image = np.ones((100, 100)) * 5
        snr = calculate_roi_snr(image, box_size=20)
        self.assertEqual(snr, float('inf'))

    def test_calculate_roi_snr_negative_mean(self):
        image = np.zeros((100, 100))
        roi = np.zeros((20, 20))
        roi[:10, :] = -3
        roi[10:, :] = -1
        # mu = -2, sigma = 1
        image[40:60, 40:60] = roi

        snr = calculate_roi_snr(image, box_size=20)

        # The code implementation explicitly uses `abs(mu / sigma)`
        # so testing to match that exact code implementation.
        expected_snr = 20 * np.log10(np.abs(-2.0 / 1.0) + 1e-8)
        self.assertAlmostEqual(snr, expected_snr, places=5)

if __name__ == '__main__':
    unittest.main()
