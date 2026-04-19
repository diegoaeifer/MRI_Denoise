import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Mock pymupdf before importing extract_posters
sys.modules['pymupdf'] = MagicMock()

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import extract_posters

class TestExtractPosters(unittest.TestCase):

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.makedirs')
    @patch('builtins.open')
    @patch('extract_posters.extract_from_pdf')
    def test_main_parallel(self, mock_extract, mock_open, mock_makedirs, mock_listdir, mock_exists):
        # Setup
        mock_exists.return_value = True
        mock_listdir.return_value = ['test1.pdf', 'test2.pdf']
        mock_extract.side_effect = [
            (['Model1'], ['Loss1'], [('Param1', 'Val1')]),
            (['Model2'], ['Loss2'], [('Param2', 'Val2')])
        ]

        # We need a real file-like object for open
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Execute with use_threads=True so mocks work
        extract_posters.main(use_threads=True)

        # Verify
        self.assertEqual(mock_extract.call_count, 2)
        mock_makedirs.assert_called_once()

        # Check if output contains expected info (roughly)
        calls = [call[0][0] for call in mock_file.write.call_args_list]
        combined_output = "".join(calls)
        self.assertIn("Total PDFs found: 2", combined_output)
        self.assertIn("--- File: test1.pdf ---", combined_output)
        self.assertIn("--- File: test2.pdf ---", combined_output)
        self.assertIn("Model1", combined_output)
        self.assertIn("Model2", combined_output)

    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('builtins.open')
    def test_main_dir_not_found(self, mock_open, mock_makedirs, mock_exists):
        # Setup
        mock_exists.return_value = False
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Execute
        extract_posters.main()

        # Verify
        mock_file.write.assert_any_call("Directory not found!\n")

if __name__ == "__main__":
    unittest.main()
