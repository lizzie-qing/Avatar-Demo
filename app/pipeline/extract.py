from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from .io import Sample


# ---- Mediapipe init (module-level singleton) ----
_mp_face_mesh = mp.solutions.face_mesh
_face_mesh = _mp_face_mesh.FaceMesh(static_image_mode=True)


def _polygon_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _fit_ellipse_ratio(eye_pts: np.ndarray) -> float:
    if len(eye_pts) < 5:
        return float("nan")
    (x, y), (MA, ma), angle = cv2.fitEllipse(eye_pts.astype(np.float32))
    return (MA / ma) if ma > 0 else float("nan")


def extract_features_one(image_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Returns:
      - features dict if success else None
      - error_reason if failed else None
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, "cv2_imread_failed"

    h, w = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb_img)

    if not results.multi_face_landmarks:
        return None, "no_face_detected"

    landmarks = results.multi_face_landmarks[0].landmark
    points = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

    # basic points
    left_cheek = points[234]
    right_cheek = points[454]
    chin = points[152]
    forehead = points[10]

    # eye contour indices
    left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263]
    left_eye = points[left_eye_idx]
    right_eye = points[right_eye_idx]

    # mouth points
    left_mouth = points[61]
    right_mouth = points[291]

    # feature calc
    face_width = float(np.linalg.norm(left_cheek - right_cheek))
    face_height = float(np.linalg.norm(forehead - chin))
    fWHR = (face_width / face_height) if face_height > 0 else float("nan")

    left_eye_area = _polygon_area(left_eye)
    right_eye_area = _polygon_area(right_eye)
    eye_area_total = left_eye_area + right_eye_area
    face_area = face_width * face_height
    EFR = (eye_area_total / face_area) if face_area > 0 else float("nan")

    ESI_left = _fit_ellipse_ratio(left_eye)
    ESI_right = _fit_ellipse_ratio(right_eye)
    ESI = (ESI_left + ESI_right) / 2.0

    mouth_width = float(np.linalg.norm(left_mouth - right_mouth))
    mouth_slope = float(
        np.degrees(
            np.arctan2(
                float(right_mouth[1] - left_mouth[1]),
                float(right_mouth[0] - left_mouth[0]),
            )
        )
    )

    feats = {
        "fWHR": fWHR,
        "EFR": EFR,
        "ESI": ESI,
        "Smile_Angle": mouth_slope,
        "Mouth_Width": mouth_width,
    }
    return feats, None


def run_feature_extraction(
    samples: List[Sample],
    out_dir: str,
) -> Dict:
    """
    Writes:
      - outputs/features.csv   (contract)
    Returns:
      meta dict with counts
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    num_ok = 0
    num_fail = 0

    for s in samples:
        feats, err = extract_features_one(s.path)
        row = {
            "sample_id": s.sample_id,
            "path": s.path,
            "status": "ok" if err is None else "fail",
            "error": "" if err is None else err,
        }
        if feats is not None:
            row.update(feats)
            num_ok += 1
        else:
            # keep feature columns but empty (NaN) —方便后续清洗统计
            row.update(
                {
                    "fWHR": np.nan,
                    "EFR": np.nan,
                    "ESI": np.nan,
                    "Smile_Angle": np.nan,
                    "Mouth_Width": np.nan,
                }
            )
            num_fail += 1
        rows.append(row)

    df = pd.DataFrame(rows)
    # 固定列顺序（契约稳定）
    cols = [
        "sample_id",
        "path",
        "status",
        "error",
        "fWHR",
        "EFR",
        "ESI",
        "Smile_Angle",
        "Mouth_Width",
    ]
    df = df[cols]
    out_csv = out / "features.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return {
        "num_samples": len(samples),
        "num_ok": num_ok,
        "num_fail": num_fail,
        "output": str(out_csv),
    }
