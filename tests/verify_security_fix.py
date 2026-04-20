import sys
import os
from unittest.mock import MagicMock, patch

# Mock dependencies before importing DenoisePipeline
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pydicom'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['models'] = MagicMock()
sys.modules['models.factory'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.ndimage'] = MagicMock()

# Ensure src is in sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import unittest
import torch
from pipeline import DenoisePipeline

class TestSecurityFix(unittest.TestCase):
    @patch('os.path.exists')
    def test_torch_load_weights_only(self, mock_exists):
        mock_exists.return_value = True
        mock_config = {'models': {}}
        checkpoint_path = 'dummy_checkpoint.pth'

        # Mocking the return value of torch.load
        torch.load.return_value = {'model_state_dict': {}}

        # Initialize pipeline
        # Note: We need to mock get_model since it's used in __init__
        with patch('pipeline.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            pipeline = DenoisePipeline('drunet', mock_config, checkpoint_path=checkpoint_path)

            # Verify torch.load was called with weights_only=True
            torch.load.assert_called_once()
            args, kwargs = torch.load.call_args
            self.assertEqual(args[0], checkpoint_path)
            self.assertTrue(kwargs.get('weights_only'), "torch.load must be called with weights_only=True")

if __name__ == '__main__':
    unittest.main()
