
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent
csv_path = project_dir / "data" / "processed" / "documents.csv"

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nDocuments per category:")
print(df["category"].value_counts())

print("\nExtraction status:")
print(df["status"].value_counts())

# Keep only successfully extracted documents
df = df[
    (df["status"] == "ok")
    & df["text"].notna()
    & (df["text"].str.strip() != "")
].copy()

print("\nUsable documents:", len(df))
print("\nUsable documents per category:")
print(df["category"].value_counts())

if df["category"].nunique() < 3:
    print("\nWARNING: Some categories have no usable documents.")

if df["category"].value_counts().min() < 2:
    print("WARNING: Collect more documents before splitting the dataset.")