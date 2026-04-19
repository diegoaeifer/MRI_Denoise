import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import warnings

from src.losses.composite import CompositeLoss

class TestCompositeLoss(unittest.TestCase):
    def setUp(self):
        self.config = {
            'weights': {
                'l1': 1.0,
                'ssim': 1.0,
                'ms_ssim': 1.0,
                'psnr': 0.1,
                'haarpsi': 0.1,
                'epi': 0.1,
                'charbonnier': 0.1,
                'vgg': 0.0,
                'sure': 0.1,
                'lpips': 0.1,
                'dists': 0.1
            },
            'auxiliary': {
                'charbonnier_eps': 1e-3,
                'vgg_layer': 'relu3_3'
            }
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self.loss_fn = CompositeLoss(self.config)

    def test_forward_basic(self):
        """Test the basic forward pass with standard tensors."""
        # Using 128x128 because MS-SSIM requires at least 81x81
        pred = torch.rand((2, 1, 128, 128))
        target = torch.rand((2, 1, 128, 128))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            total_loss, metrics = self.loss_fn(pred, target)

        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(total_loss.dim(), 0) # Scalar

        expected_keys = ['l1', 'ssim', 'ms_ssim', 'psnr', 'haarpsi', 'epi', 'lpips', 'dists']
        for key in expected_keys:
            self.assertIn(key, metrics)

    def test_forward_with_sure(self):
        """Test the forward pass including SURE loss which requires model and input_tensor."""
        pred = torch.rand((2, 1, 128, 128))
        target = torch.rand((2, 1, 128, 128))
        model = nn.Identity()
        input_tensor = torch.rand((2, 2, 128, 128)) # Requires 2 channels for sigma_map

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            total_loss, metrics = self.loss_fn(pred, target, model=model, input_tensor=input_tensor)

        self.assertIsInstance(total_loss, torch.Tensor)

    def test_ms_ssim_exception_handling(self):
        """Test that the exception in ms_ssim computation is properly caught."""
        # MS-SSIM requires at least 81x81, using 64x64 should trigger the exception
        pred = torch.rand((2, 1, 64, 64))
        target = torch.rand((2, 1, 64, 64))

        with patch('logging.Logger.error') as mock_error:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                total_loss, metrics = self.loss_fn(pred, target)

            # Verify error was logged for MS-SSIM
            mock_error.assert_called()

            # The MS-SSIM loss should fallback to 1.0 when it errors
            self.assertEqual(metrics['ms_ssim'].item(), 1.0)

    @patch('src.losses.composite.piq.HaarPSILoss')
    def test_haarpsi_exception_handling(self, mock_haarpsi_class):
        """Test that the exception in haarpsi computation is properly caught."""
        # Create a mock instance that raises an Exception when called
        mock_instance = MagicMock()
        mock_instance.side_effect = Exception("Test exception")

        # We need a new CompositeLoss where HaarPSILoss is the mocked one
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            loss_fn = CompositeLoss(self.config)
            loss_fn.haarpsi = mock_instance

        pred = torch.rand((2, 1, 128, 128))
        target = torch.rand((2, 1, 128, 128))

        with patch('logging.Logger.error') as mock_error:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                total_loss, metrics = loss_fn(pred, target)

            # Verify error was logged
            mock_error.assert_called()

            # The HaarPSI loss should fallback to 1.0 when it errors
            self.assertEqual(metrics['haarpsi'].item(), 1.0)

    def test_identical_tensors(self):
        """Test that identical tensors yield minimal loss for difference metrics."""
        pred = torch.rand((2, 1, 128, 128))
        target = pred.clone()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            total_loss, metrics = self.loss_fn(pred, target)

        # L1 should be exactly 0
        self.assertAlmostEqual(metrics['l1'].item(), 0.0, places=4)

        # SSIM loss is 1 - SSIM. Perfect SSIM is 1, so loss should be 0
        self.assertAlmostEqual(metrics['ssim'].item(), 0.0, places=4)

if __name__ == '__main__':
    unittest.main()
