
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "documents_combined.csv"

df = pd.read_csv(DATA_PATH)

for category in ["invoice", "bill", "receipt"]:
    print(f"\n===== {category.upper()} =====")

    samples = df[df["category"] == category]["text"].dropna().head(3)

    for i, sample in enumerate(samples, start=1):
        print(f"\nSample {i}:")
        print(str(sample)[:1000])

