

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OLD_CSV = PROCESSED_DIR / "documents.csv"
RECEIPT_CSV = PROCESSED_DIR / "sroie_receipts.csv"
OUTPUT_CSV = PROCESSED_DIR / "documents_combined.csv"

records = []

# 1. Load invoice OCR text from the existing CSV files.
for csv_path in (RAW_DIR / "invoice").rglob("*.csv"):
    try:
        df = pd.read_csv(csv_path)

        if "OCRed Text" not in df.columns:
            continue

        for _, row in df.iterrows():
            text = str(row["OCRed Text"]).strip()

            if text and text.lower() != "nan":
                records.append({
                    "file_name": str(row.get("File Name", csv_path.name)),
                    "category": "invoice",
                    "text": text,
                    "status": "ok"
                })

        print(f"Loaded invoice CSV: {csv_path.name}")

    except Exception as e:
        print(f"Could not read {csv_path}: {e}")

# 2. Keep successfully extracted bills.
if OLD_CSV.exists():
    old_df = pd.read_csv(OLD_CSV)

    for _, row in old_df.iterrows():
        text = str(row.get("text", "")).strip()
        category = str(row.get("category", "")).lower()

        if (
            category == "bill"
            and str(row.get("status", "")) == "ok"
            and text and text.lower() != "nan"
        ):
            records.append({
                "file_name": str(row["file_name"]),
                "category": "bill",
                "text": text,
                "status": "ok"
            })

# 3. Import SROIE receipt annotations.
if RECEIPT_CSV.exists():
    receipt_df = pd.read_csv(RECEIPT_CSV)

    for _, row in receipt_df.iterrows():
        text = str(row.get("text", "")).strip()

        if text and text.lower() != "nan":
            records.append({
                "file_name": str(row["file_name"]),
                "category": "receipt",
                "text": text,
                "status": "ok"
            })
else:
    print("Receipt CSV not found:", RECEIPT_CSV)

# 4. Create the combined dataset.
combined = pd.DataFrame(records)

# Remove exact duplicate text within each category.
if not combined.empty:
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined.drop_duplicates(
        subset=["category", "text"]
    ).reset_index(drop=True)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print("\n===== COMBINED DATASET =====")
print("Total unique records:", len(combined))

if not combined.empty:
    print("\nDocuments per category:")
    print(combined["category"].value_counts())

print("\nSaved to:", OUTPUT_CSV)