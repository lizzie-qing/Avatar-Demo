from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error


FEATURE_COLS = ["fWHR", "EFR", "ESI", "Mouth_Width"]
TARGET_COL = "Smile_Angle"   # demo baseline target


def run_regression(
    cleaned_csv: str,
    out_dir: str,
    seed: int = 42,
    alpha: float = 1.0,
) -> Dict:
    """
    Baseline regression for demo:
      y = Smile_Angle
      X = fWHR, EFR, ESI, Mouth_Width
    Writes:
      - outputs/regression_summary.txt (contract)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cleaned_csv)

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    r2_tr = r2_score(y_train, pred_train)
    r2_te = r2_score(y_test, pred_test)
    mae_te = mean_absolute_error(y_test, pred_test)

    # coefficient importance (abs)
    coefs = model.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1]
    topk = [(FEATURE_COLS[i], float(coefs[i])) for i in order]

    lines = []
    lines.append("=== Regression Summary (Demo Baseline) ===")
    lines.append(f"Target (y): {TARGET_COL}")
    lines.append(f"Features (X): {', '.join(FEATURE_COLS)}")
    lines.append("")
    lines.append(f"N total: {len(df)}")
    lines.append(f"N train: {len(X_train)} | N test: {len(X_test)}")
    lines.append("")
    lines.append("Model: Ridge Regression")
    lines.append(f"alpha: {alpha}")
    lines.append("")
    lines.append(f"R^2 (train): {r2_tr:.4f}")
    lines.append(f"R^2 (test):  {r2_te:.4f}")
    lines.append(f"MAE (test):  {mae_te:.4f}")
    lines.append("")
    lines.append("Coefficients (sorted by |coef|):")
    for name, coef in topk:
        lines.append(f"  {name:12s}  {coef:+.6f}")

    out_txt = out / "regression_summary.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "n": int(len(df)),
        "target": TARGET_COL,
        "features": FEATURE_COLS,
        "r2_test": float(r2_te),
        "mae_test": float(mae_te),
        "output": str(out_txt),
    }
