import json
from pathlib import Path
import py_compile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

print("Running project validation...")

py_compile.compile(str(ROOT / "app.py"), doraise=True)
print("[PASS] app.py syntax")

with (ROOT / "person_details.json").open("r", encoding="utf-8") as f:
    details = json.load(f)
assert isinstance(details, dict)
print(f"[PASS] person_details.json ({len(details)} persons)")

metadata_path = PROJECT_ROOT / "Data" / "Embeddings" / "metadata_clean.csv"
embeddings_path = PROJECT_ROOT / "Data" / "Embeddings" / "embeddings_clean.npy"

if metadata_path.exists() and embeddings_path.exists():
    metadata = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path, allow_pickle=False)

    assert "person" in metadata.columns
    assert len(metadata) == len(embeddings)

    print(f"[PASS] metadata records: {len(metadata)}")
    print(f"[PASS] unique persons: {metadata['person'].nunique()}")
    print("[PASS] embedding/metadata row count matches")
else:
    print("[WARN] Data/Embeddings files not found from expected project root.")

print("Validation complete.")
