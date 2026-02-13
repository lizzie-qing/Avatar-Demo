# AvatarDemo — Face geometry → impression prediction (demo)

A small end-to-end demo project:

**PNG avatars → facial geometry features (MediaPipe) → cleaning → PCA → regression → report + API + dashboard**

## Output contract (fixed artifacts)
Running the pipeline will generate the following files under `outputs/`:

- `features.csv` — extracted features per image (fWHR, EFR, ESI, Smile_Angle, Mouth_Width)
- `cleaned.csv` — cleaned feature table after simple rules
- `pca.png` — PC1–PC2 scatter plot
- `regression_summary.txt` — baseline regression summary (Ridge/OLS)
- `report.md` — auto-generated readable report (includes PCA figure + key stats)

(Extra) `io_log.json`, `cleaning_log.json`, `run_metadata.json` are also generated for traceability.

## Project structure

```text
AvatarDemo/
├── app/
│   ├── __init__.py
│   ├── __main__.py          # `python -m app ...` entry
│   ├── cli.py               # CLI: `python -m app run ...`
│   ├── config.py            # load YAML config + defaults
│   ├── api.py               # FastAPI app (Swagger: /docs)
│   ├── dashboard.py         # Streamlit UI
│   └── pipeline/
│       ├── __init__.py
│       ├── io.py            # scan PNGs, basic checks, io_log.json
│       ├── extract.py       # MediaPipe feature extraction -> features.csv
│       ├── clean.py         # simple cleaning rules -> cleaned.csv + cleaning_log.json
│       ├── pca.py           # PCA + plot -> pca.png
│       ├── regress.py       # baseline regression -> regression_summary.txt
│       └── report.py        # report.md + run_metadata.json
├── configs/
│   └── default.yaml         # demo configuration (target, model, etc.)
├── assets/
│   ├── pca.png
│   ├── api_predict.png
│   └── dashboard.png
├── data/                    # sample PNG avatars (keep small for repo)
├── outputs/                 # generated artifacts (ignored by git)
├── tests/
│   └── test_pipeline.py
├── demo.sh                  # one-command local demo script
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Quick start (30s)
### Option A: local Python
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m app run --input data --out outputs --config configs/default.yaml**.
```

### Option B: Docker
```bash
docker build -t avatardemo .
docker run --rm -v "$PWD/outputs:/app/outputs" avatardemo
```

## API (FastAPI)
### Start
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8001
```
### Open Swagger:
http://127.0.0.1:8001/docs
### Endpoints:
- `GET /health`
- `POST /predict` (multipart image upload)

## Dashboard (Streamlit)
`
streamlit run app/dashboard.py
`
### Open:

` http://127.0.0.1:8501`

### Screenshots
- PCA: `assets/pca.png`

- API /predict response: ` assets/api_predict.png` 

- Dashboard: ` assets/dashboard.png` 

# Notes / limitations
- This is a demo baseline. Feature extraction uses MediaPipe FaceMesh and simple geometric heuristics.

- The regression target in this demo is configurable; current default is defined in ` configs/default.yaml.` 

- For real research / production, we would add dataset versioning, stronger validation, and a proper model training pipeline.

## Makefile shortcuts (recommended)

```bash
make setup
make run
make test
make clean
```

## Demo script 
` bash demo.sh` 


