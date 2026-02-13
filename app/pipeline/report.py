from __future__ import annotations

from pathlib import Path
from typing import Dict
import datetime as dt
import os


def run_report(
    out_dir: str,
    input_dir: str,
    n_samples: int,
    n_cleaned: int,
) -> Dict:
    """
    Writes:
      - outputs/report.md (contract)
    Assumes these files already exist in out_dir:
      - features.csv
      - cleaned.csv
      - pca.png
      - regression_summary.txt
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    features_csv = out / "features.csv"
    cleaned_csv = out / "cleaned.csv"
    pca_png = out / "pca.png"
    reg_txt = out / "regression_summary.txt"

    # read regression summary text (small)
    reg_text = reg_txt.read_text(encoding="utf-8") if reg_txt.exists() else "(missing regression_summary.txt)"

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# Auto Report")
    md.append("")
    md.append(f"- Generated at: `{now}`")
    md.append("")
    md.append("## How to run")
    md.append("")
    md.append("```bash")
    md.append(f"python -m app run --input {input_dir} --out {out_dir}")
    md.append("```")
    md.append("")
    md.append("## Data overview")
    md.append("")
    md.append(f"- Input samples scanned: **{n_samples}**")
    md.append(f"- Cleaned samples kept: **{n_cleaned}**")
    md.append("")
    md.append("## Cleaning rules (minimal)")
    md.append("")
    md.append("- Keep `status == ok`")
    md.append("- Drop rows with NaN in any feature columns")
    md.append("- IQR filter (k=1.5) per feature column")
    md.append("")
    md.append("## PCA")
    md.append("")
    md.append(f"![pca](pca.png)")
    md.append("")
    md.append("## Regression summary")
    md.append("")
    md.append("```text")
    md.append(reg_text.strip())
    md.append("```")
    md.append("")
    md.append("## Artifacts")
    md.append("")
    md.append(f"- `{features_csv.name}`")
    md.append(f"- `{cleaned_csv.name}`")
    md.append(f"- `{pca_png.name}`")
    md.append(f"- `{reg_txt.name}`")
    md.append("")

    report_path = out / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    return {"output": str(report_path)}
