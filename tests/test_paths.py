import os


def test_script_paths():
    assert os.path.exists("src/mri_denoise/train.py")
