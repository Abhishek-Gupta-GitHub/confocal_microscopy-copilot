# copilot/io_utils.py

import json
from pathlib import Path

import pandas as pd
import tifffile

def load_stack(path: str):
    """Load a TIFF stack as numpy array."""
    return tifffile.imread(path)

def save_stack(path: str, stack):
    """Save a 3D/4D stack as float32 TIFF."""
    tifffile.imwrite(path, stack.astype("float32"))

def load_metadata(path: str):
    """Load JSON metadata dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with open(p, "r") as f:
        return json.load(f)

def save_metadata(path: str, meta: dict):
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_trajectories_csv(path: str, df: pd.DataFrame):
    df.to_csv(path, index=False)

