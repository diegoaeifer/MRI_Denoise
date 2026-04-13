import sys
import pytest

def test_tqdm_fallback():
    """
    Test the fallback implementation of tqdm in train.py when tqdm is not installed.
    """
    original_modules = sys.modules.copy()

    # Pre-import heavy dependencies so they don't break when we modify sys.modules
    import torch
    import torchio
    import torchvision
    import yaml
    import argparse
    import logging

    # Ensure src.trainer isn't cached
    if 'trainer' in sys.modules:
        del sys.modules['trainer']
    if 'src.trainer' in sys.modules:
        del sys.modules['src.trainer']

    # By setting sys.modules['tqdm'] = None, we force Python's import system
    # to raise a ModuleNotFoundError when anything tries to import tqdm
    sys.modules['tqdm'] = None

    try:
        # We need to test the file in `src.trainer`
        # Adding src to sys.path helps it resolve local imports like `data.loader`
        if 'src' not in sys.path:
            sys.path.insert(0, 'src')

        import trainer

        # Test fallback function signature and return value
        iterable = [1, 2, 3]
        result = trainer.tqdm(iterable)

        # Should return exactly the iterable
        assert result == iterable
        assert list(result) == [1, 2, 3]

        # Should accept *args and **kwargs without error
        result2 = trainer.tqdm(iterable, "desc", total=10)
        assert result2 == iterable
    finally:
        # Clean up modifications
        if 'src' in sys.path:
            sys.path.remove('src')

        # Restore sys.modules carefully
        for key in list(sys.modules.keys()):
            if key not in original_modules:
                del sys.modules[key]
        sys.modules.update(original_modules)
