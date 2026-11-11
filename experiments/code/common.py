"""Shared provenance and strict final-state validation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha256(root / name) != expected:
            raise RuntimeError(f"Snapshot changed: {name}. Prepare a new campaign.")
    return manifest


def compare_states(reference, candidate):
    a = pd.read_csv(reference, float_precision="round_trip")
    b = pd.read_csv(candidate, float_precision="round_trip")
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        raise RuntimeError("Final-state columns or particle counts differ")
    max_difference = 0.0
    for col in a:
        if pd.api.types.is_float_dtype(a[col]):
            if not np.isfinite(a[col]).all() or not np.isfinite(b[col]).all():
                raise RuntimeError(f"Non-finite final state: {col}")
            np.testing.assert_allclose(a[col], b[col], rtol=1e-12, atol=1e-12,
                                       err_msg=f"Final-state mismatch: {col}")
            max_difference = max(max_difference, float(np.abs(a[col] - b[col]).max()))
        elif not a[col].equals(b[col]):
            raise RuntimeError(f"Final-state mismatch: {col}")
    return {"passed": True, "exact": a.equals(b), "max_abs_difference": max_difference,
            "rtol": 1e-12, "atol": 1e-12, "n_final": len(a)}
