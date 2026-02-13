from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json
import time

@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: str

def read_samples(input_dir: str) -> Tuple[List[Sample], dict]:
    t0 = time.time()
    p = Path(input_dir)

    meta = {
        "input_dir": str(p),
        "pattern": "*.png",
        "num_found": 0,
        "num_ok": 0,
        "num_skipped": 0,
        "skipped": [],
        "elapsed_sec": None,
    }

    if not p.exists() or not p.is_dir():
        meta["skipped"].append({"file": str(p), "reason": "input_dir_not_found_or_not_dir"})
        meta["num_skipped"] = 1
        meta["elapsed_sec"] = round(time.time() - t0, 4)
        return [], meta

    files = sorted(p.glob("*.png"))
    meta["num_found"] = len(files)

    samples: List[Sample] = []
    for f in files:
        try:
            if (not f.exists()) or f.stat().st_size == 0:
                meta["skipped"].append({"file": str(f), "reason": "empty_or_missing"})
                continue
            samples.append(Sample(sample_id=f.stem, path=str(f)))
        except Exception as e:
            meta["skipped"].append({"file": str(f), "reason": f"exception:{type(e).__name__}"})

    meta["num_ok"] = len(samples)
    meta["num_skipped"] = len(meta["skipped"])
    meta["elapsed_sec"] = round(time.time() - t0, 4)
    return samples, meta

def write_io_log(out_dir: str, meta: dict) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "io_log.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
