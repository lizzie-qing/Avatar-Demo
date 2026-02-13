from pathlib import Path
import pandas as pd


OUT_DIR = Path("outputs")


def test_features_csv_exists_and_has_columns():
    f = OUT_DIR / "features.csv"
    assert f.exists(), "outputs/features.csv not found"

    df = pd.read_csv(f)
    expected_cols = [
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
    for c in expected_cols:
        assert c in df.columns, f"missing column: {c}"


def test_cleaned_csv_no_nan_in_features():
    f = OUT_DIR / "cleaned.csv"
    assert f.exists(), "outputs/cleaned.csv not found"

    df = pd.read_csv(f)
    feat_cols = ["fWHR", "EFR", "ESI", "Smile_Angle", "Mouth_Width"]
    assert df[feat_cols].isna().sum().sum() == 0, "NaN found in cleaned feature columns"


def test_pca_png_exists():
    f = OUT_DIR / "pca.png"
    assert f.exists(), "outputs/pca.png not found"
    assert f.stat().st_size > 0, "outputs/pca.png is empty"
