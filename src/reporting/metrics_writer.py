import json
import numpy as np
import os
from pathlib import Path
import subprocess

def numpy_encoder(obj):
    """Casts numpy types to native Python types for JSON serialization."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    return obj

def write_latest_metrics(metrics_dict, file_path="reports/latest_metrics.json"):
    """
    Safely writes metrics to a JSON file with numpy support and atomic rename.
    Includes a round-trip check to ensure validity.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    # Add metadata
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        git_hash = "unknown"
    
    metrics_dict["generated_at"] = np.datetime64('now').astype(str)
    metrics_dict["git_hash"] = git_hash

    # Write to temp file
    with open(temp_path, "w") as f:
        json.dump(metrics_dict, f, default=numpy_encoder, indent=4)

    # Round-trip check
    with open(temp_path, "r") as f:
        try:
            json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ JSON Round-trip check failed: {e}")
            return False

    # Atomic rename
    if path.exists():
        path.unlink()
    temp_path.rename(path)
    print(f"✅ Metrics successfully written to {path}")
    return True
