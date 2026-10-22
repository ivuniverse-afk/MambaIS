"""Load experiment settings from YAML files."""

from pathlib import Path
import yaml


def load_config(path):
    with Path(path).open() as f:
        return yaml.safe_load(f)
