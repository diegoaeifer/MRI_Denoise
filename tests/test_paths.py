import os
import yaml


def test_config_paths():
    config_path = "configs/config_data.yaml"
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assert config["data"]["raw_path"] == "data/IXI"
    assert config["data"]["processed_path"] == "data/processed"
    assert config["data"]["splits_path"] == "data/splits"


def test_script_paths():
    assert os.path.exists("src/train.py")
    assert os.path.exists("src/inference.py")
    assert os.path.exists("run_trials.py")


if __name__ == "__main__":
    test_config_paths()
    test_script_paths()
    print("Path tests passed!")
