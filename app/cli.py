from app.pipeline.io import read_samples, write_io_log
from app.pipeline.extract import run_feature_extraction
from app.pipeline.clean import run_cleaning
from app.pipeline.pca import run_pca
from app.pipeline.regress import run_regression
from app.pipeline.report import run_report
import os
import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the full pipeline")
    run.add_argument("--input", required=True, help="Input data directory, e.g. data/")
    run.add_argument("--out", required=True, help="Output directory, e.g. outputs/")
    run.add_argument("--config", default="configs/default.yaml", help="Config yaml path")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        from app.config import load_config
        cfg = load_config(args.config)

        # step2：IO
        samples, meta = read_samples(args.input)
        log_path = write_io_log(args.out, meta)
        print(f"[io] samples_ok={len(samples)} skipped={meta['num_skipped']} log={log_path}")

        # step3：features
        feat_meta = run_feature_extraction(samples, args.out)
        print(f"[features] ok={feat_meta['num_ok']} fail={feat_meta['num_fail']} -> {feat_meta['output']}")

        # step4：cleaning
        features_path = os.path.join(args.out, "features.csv")
        iqr_k = float(cfg.get("cleaning", {}).get("iqr_k", 1.5))
        clean_meta = run_cleaning(features_path, args.out, iqr_k=iqr_k)
        print(f"[clean] n={clean_meta['n_cleaned']} -> {clean_meta['cleaned_csv']}")

        # Step5: PCA
        cleaned_path = os.path.join(args.out, "cleaned.csv")
        seed = int(cfg.get("seed", 42))
        n_components = int(cfg.get("pca", {}).get("n_components", 2))
        pca_meta = run_pca(cleaned_path, args.out, n_components=n_components, seed=seed)
        print(f"[pca] n={pca_meta['n']} -> {pca_meta['output']}")

        # Step6: Regression
        cleaned_path = os.path.join(args.out, "cleaned.csv")
        alpha = float(cfg.get("regression", {}).get("alpha", 1.0))
        reg_meta = run_regression(cleaned_path, args.out, seed=seed, alpha=alpha)
        print(f"[regress] n={reg_meta['n']} -> {reg_meta['output']}")

        # Step7: Report
        rep_meta = run_report(
            out_dir=args.out,
            input_dir=args.input,
            n_samples=len(samples),
            n_cleaned=clean_meta["n_cleaned"],
        )
        print(f"[report] -> {rep_meta['output']}")

        import json, time
        from pathlib import Path

        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": args.config,
            "config": cfg,
            "counts": {
                "samples_scanned": len(samples),
                "cleaned": clean_meta["n_cleaned"],
            },
            "artifacts": {
                "features_csv": str(Path(args.out) / "features.csv"),
                "cleaned_csv": str(Path(args.out) / "cleaned.csv"),
                "pca_png": str(Path(args.out) / "pca.png"),
                "regression_summary": str(Path(args.out) / "regression_summary.txt"),
                "report_md": str(Path(args.out) / "report.md"),
            },
        }
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
        print(f"[meta] -> {Path(args.out) / 'run_metadata.json'}")







