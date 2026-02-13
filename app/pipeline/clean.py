from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd


FEATURE_COLS = ["fWHR", "EFR", "ESI", "Smile_Angle", "Mouth_Width"]


def _iqr_bounds(s: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return float(lo), float(hi)


def run_cleaning(
    features_csv: str,
    out_dir: str,
    iqr_k: float = 1.5,
) -> Dict:
    """
    Minimal cleaning rules:
      1) keep only status==ok
      2) drop rows with NaN in any feature columns
      3) IQR filter per feature (remove rows outside bounds)
    Writes:
      - outputs/cleaned.csv (contract)
      - outputs/cleaning_log.json (optional but recommended)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)

    log = {
        "input": features_csv,
        "feature_cols": FEATURE_COLS,
        "steps": [],
        "iqr_k": iqr_k,
    }

    n0 = len(df)
    log["steps"].append({"name": "load", "n": n0})

    # 1) status ok
    df1 = df[df["status"] == "ok"].copy()
    log["steps"].append({"name": "filter_status_ok", "before": n0, "after": len(df1)})

    # 2) drop NaN
    before = len(df1)
    df2 = df1.dropna(subset=FEATURE_COLS).copy()
    log["steps"].append({"name": "drop_nan", "before": before, "after": len(df2)})

    # 3) IQR filter (row-wise mask)
    before = len(df2)
    mask = np.ones(len(df2), dtype=bool)

    bounds = {}
    for col in FEATURE_COLS:
        lo, hi = _iqr_bounds(df2[col], k=iqr_k)
        bounds[col] = {"lo": lo, "hi": hi}
        mask &= (df2[col] >= lo) & (df2[col] <= hi)

    df3 = df2.loc[mask].copy()
    log["steps"].append({"name": "iqr_filter", "before": before, "after": len(df3), "bounds": bounds})

    # output
    cleaned_path = out / "cleaned.csv"
    df3.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    log_path = out / "cleaning_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "n_input": n0,
        "n_cleaned": len(df3),
        "cleaned_csv": str(cleaned_path),
        "log_json": str(log_path),
    }
