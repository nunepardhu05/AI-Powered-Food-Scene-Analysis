from pathlib import Path
from functools import lru_cache
import yaml

@lru_cache(maxsize=1)
def settings():
    path=Path(__file__).resolve().parents[1]/'config.yaml'
    config=yaml.safe_load(path.read_text())
    if not 0<config['recognition_threshold']<1:raise ValueError('Recognition threshold must be between 0 and 1')
    return config
