


from pathlib import Path

import pandas as pd
import pymupdf
import pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "documents.csv"

CATEGORIES = ["invoice", "bill", "receipt"]


def extract_pdf_text(pdf_path):
    text_parts = []
    ocr_pages = 0

    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            text = page.get_text().strip()

            if len(text) < 30:
                pix = page.get_pixmap(dpi=200)
                image = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples
                )
                text = pytesseract.image_to_string(image).strip()
                ocr_pages += 1

            text_parts.append(text)

    return "\n".join(text_parts).strip(), ocr_pages


def main():
    records = []

    for category in CATEGORIES:
        folder = RAW_DIR / category

        # Search the category folder and every subfolder.
        pdf_files = list(folder.rglob("*.pdf"))

        print(f"\n{category.upper()}: {len(pdf_files)} PDF(s) found")

        for pdf_path in pdf_files:
            try:
                text, ocr_pages = extract_pdf_text(pdf_path)

                records.append({
                    "file_name": str(pdf_path.relative_to(RAW_DIR)),
                    "category": category,
                    "text": text,
                    "ocr_pages": ocr_pages,
                    "status": "ok" if text else "empty",
                    "error": ""
                })

                print(
                    f"  {pdf_path.name}: "
                    f"{len(text)} characters extracted"
                )

            except Exception as e:
                records.append({
                    "file_name": str(pdf_path.relative_to(RAW_DIR)),
                    "category": category,
                    "text": "",
                    "ocr_pages": 0,
                    "status": "error",
                    "error": str(e)
                })

                print(f"  Error with {pdf_path.name}: {e}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records, columns=[
        "file_name", "category", "text",
        "ocr_pages", "status", "error"
    ])

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== DATASET SUMMARY =====")
    print(f"Total PDFs processed: {len(df)}")

    if not df.empty:
        print("\nDocuments per category:")
        print(df["category"].value_counts())

        print("\nExtraction status:")
        print(df["status"].value_counts())

    print(f"\nDataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()