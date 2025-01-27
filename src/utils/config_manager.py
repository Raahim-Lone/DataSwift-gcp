import yaml
import os

def load_config(config_path="config/default_config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    required_keys = ["database", "training", "paths", "logging"]
    for rk in required_keys:
        if rk not in config:
            raise ValueError(f"Missing '{rk}' section in config")

    return config
