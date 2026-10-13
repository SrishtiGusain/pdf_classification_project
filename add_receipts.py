
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SROIE_DIR = BASE_DIR / "data" / "raw" / "receipt" / "SROIE2019"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "sroie_receipts.csv"


def read_annotation(txt_path):
    lines = []

    with open(txt_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # SROIE box annotations contain 8 coordinates, then text.
            parts = line.split(",", 8)

            if len(parts) == 9:
                text = parts[8].strip()

                if text:
                    lines.append(text)

    return " ".join(lines)


def main():
    records = []

    # Use training annotations for now.
    # Keep the official test set separate for later evaluation.
    annotation_dir = SROIE_DIR / "train" / "box"

    if not annotation_dir.exists():
        print("Annotation folder not found:", annotation_dir)
        return

    for txt_path in annotation_dir.glob("*.txt"):
        text = read_annotation(txt_path)

        if text:
            records.append({
                "file_name": txt_path.stem,
                "category": "receipt",
                "text": text,
                "status": "ok"
            })

    df = pd.DataFrame(records)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n===== SROIE RECEIPTS =====")
    print("Usable receipt annotations:", len(df))
    print("Saved to:", OUTPUT_FILE)

    if not df.empty:
        print("\nSample receipt text:")
        print(df.iloc[0]["text"][:500])


if __name__ == "__main__":
    main()