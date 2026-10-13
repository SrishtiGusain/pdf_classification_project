
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "documents_combined.csv"

df = pd.read_csv(DATA_PATH)

# Remove missing and empty text
df = df.dropna(subset=["text", "category"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# Normalize text for duplicate checking
df["normalized_text"] = (
    df["text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Find identical text assigned to multiple categories
category_counts = df.groupby("normalized_text")["category"].nunique()
cross_category_texts = category_counts[category_counts > 1]

print("===== DUPLICATE CHECK =====")
print("Total usable records:", len(df))
print("Texts appearing in multiple categories:", len(cross_category_texts))

if len(cross_category_texts) > 0:
    print("\nExamples:")
    examples = df[
        df["normalized_text"].isin(cross_category_texts.index)
    ][["category", "file_name", "text"]]

    print(examples.head(10).to_string(index=False))
else:
    print("No identical normalized text found across categories.")

