import torch
import unittest
from src.losses.auxiliary import VGGPerceptualLoss

class TestVGGPerceptualLoss(unittest.TestCase):
    def test_vgg_perceptual_loss_forward(self):
        """Test that VGGPerceptualLoss forward pass runs successfully without shape errors for a 1-channel image."""
        loss_fn = VGGPerceptualLoss()

        # Create dummy 1-channel grayscale inputs (B, C, H, W)
        x = torch.rand(2, 1, 64, 64)
        y = torch.rand(2, 1, 64, 64)

        # Run forward pass
        loss = loss_fn(x, y)

        # Verify loss is a scalar tensor
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertEqual(loss.dim(), 0)
        self.assertFalse(torch.isnan(loss))

        # Verify identical tensors yield zero loss
        loss_identical = loss_fn(x, x)
        self.assertAlmostEqual(loss_identical.item(), 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
