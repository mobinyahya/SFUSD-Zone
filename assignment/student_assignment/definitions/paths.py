import os
from pathlib import Path

# STATIC PATHS
ROOT_DIR = os.path.dirname(Path(__file__).parent)
CONFIGS_DIR = f"{ROOT_DIR}/../configs/"
SUBCONFIGS_DIR = f"{CONFIGS_DIR}policy_configs/"
