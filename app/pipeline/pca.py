from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


FEATURE_COLS = ["fWHR", "EFR", "ESI", "Smile_Angle", "Mouth_Width"]


def run_pca(
    cleaned_csv: str,
    out_dir: str,
    n_components: int = 2,
    seed: int = 42,
) -> Dict:
    """
    Reads cleaned.csv, runs PCA on FEATURE_COLS, writes:
      - outputs/pca.png (contract)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cleaned_csv)
    X = df[FEATURE_COLS].to_numpy(dtype=float)

    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X)

    # scatter PC1-PC2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Z[:, 0], Z[:, 1], s=12)

    evr = pca.explained_variance_ratio_
    ax.set_title(f"PCA Scatter (PC1={evr[0]:.2%}, PC2={evr[1]:.2%})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    out_png = out / "pca.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return {
        "n": int(len(df)),
        "feature_cols": FEATURE_COLS,
        "explained_variance_ratio": [float(e) for e in evr[:2]],
        "output": str(out_png),
    }
