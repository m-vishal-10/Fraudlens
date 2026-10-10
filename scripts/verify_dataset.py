import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.eda import write_summary


if __name__ == "__main__":
    print(json.dumps(write_summary("data/raw/creditcard.csv", "artifacts/eda_summary.json"), indent=2))
