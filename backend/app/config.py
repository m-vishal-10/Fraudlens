from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("FRAUDLENS_DATA_PATH", ROOT / "data" / "raw" / "creditcard.csv"))
ARTIFACT_DIR = Path(os.getenv("FRAUDLENS_ARTIFACT_DIR", ROOT / "artifacts"))
MAX_UPLOAD_BYTES = int(os.getenv("FRAUDLENS_MAX_UPLOAD_BYTES", "52428800"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRAUDLENS_CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
