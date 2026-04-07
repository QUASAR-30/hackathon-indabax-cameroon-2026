"""Upload files to HuggingFace Space via huggingface_hub (handles xet/LFS automatically)."""
import os
from huggingface_hub import HfApi
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

ROOT  = Path(__file__).parent
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    raise EnvironmentError("HF_TOKEN manquant — ajoutez-le dans .env ou exportez-le dans le shell")
api   = HfApi(token=token)
REPO  = "QUASAR-30/pm25-cameroun"

files = {
    "app.py":                             "app.py",
    "README_HF.md":                       "README.md",
    "requirements_hf.txt":                "requirements.txt",
    "notebooks/07_dashboard.py":          "notebooks/07_dashboard.py",
    "models/xgboost_final.pkl":           "models/xgboost_final.pkl",
    "models/lightgbm_final.pkl":          "models/lightgbm_final.pkl",
    "models/xgboost_coldstart.pkl":       "models/xgboost_coldstart.pkl",
    "data/pm25_with_uncertainty.parquet": "data/pm25_with_uncertainty.parquet",
    "data/pm25_proxy_era5.parquet":       "data/pm25_proxy_era5.parquet",
    "data/predictions_latest.parquet":    "data/predictions_latest.parquet",
    "data/alerts_latest.json":            "data/alerts_latest.json",
}

for local, remote in files.items():
    print(f"Uploading {local} ...", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=str(ROOT / local),
        path_in_repo=remote,
        repo_id=REPO,
        repo_type="space",
    )
    print("OK")

print(f"\nDone! https://huggingface.co/spaces/{REPO}")
