from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from app.pipeline.extract import extract_features_one
from app.pipeline.regress import FEATURE_COLS, TARGET_COL  # X cols and y name


app = FastAPI(title="Avatar Demo API", version="0.1.0")


def _load_training_table() -> pd.DataFrame:
    # 使用 pipeline 产物作为“训练数据”（demo用）
    cleaned_path = Path("outputs/cleaned.csv")
    if not cleaned_path.exists():
        raise FileNotFoundError("outputs/cleaned.csv not found. Run pipeline first.")
    return pd.read_csv(cleaned_path)


def _fit_demo_model(df: pd.DataFrame):
    # 用 cleaned.csv 拟合一个 Ridge（与 pipeline 一致的 demo 思路）
    from sklearn.linear_model import Ridge

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)
    return model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) 读图到临时文件（extract_features_one 走 cv2.imread 路线最省事）
    suffix = Path(file.filename).suffix.lower() if file.filename else ".png"
    tmp_path = Path("outputs") / f"_upload{suffix}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    tmp_path.write_bytes(content)

    feats, err = extract_features_one(str(tmp_path))
    if err is not None or feats is None:
        return JSONResponse(
            status_code=400,
            content={"error": err or "feature_extraction_failed"},
        )

    # 2) 载入训练表并拟合 demo 模型（最小实现：每次请求拟合一次；后面可优化成启动时加载）
    df = _load_training_table()
    model = _fit_demo_model(df)

    x = np.array([[feats[c] for c in FEATURE_COLS]], dtype=float)
    pred = float(model.predict(x)[0])

    return {
        "features": feats,
        "model": {"type": "ridge", "alpha": 1.0, "target": TARGET_COL, "features": FEATURE_COLS},
        "prediction": {TARGET_COL: pred},
    }
