

from pathlib import Path
import re
import hashlib

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "documents_combined.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "document_classifier.joblib"
MATRIX_PATH = BASE_DIR / "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\n========== PDF DOCUMENT CLASSIFICATION ==========")
print("\nLoading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please run your dataset preparation script first."
    )

df = pd.read_csv(DATA_PATH)

required_columns = {"text", "category"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "Dataset must contain 'text' and 'category' columns. "
        f"Found columns: {list(df.columns)}"
    )

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 3. CLEAN DATA
# --------------------------------------------------

df = df[["text", "category"]].copy()

df["text"] = df["text"].fillna("").astype(str)
df["category"] = df["category"].fillna("").astype(str).str.strip().str.lower()

# Remove empty text and missing labels
df["text"] = df["text"].str.strip()
df = df[(df["text"] != "") & (df["category"] != "")].copy()

# Normalize whitespace and case for duplicate detection
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["normalized_text"] = df["text"].apply(normalize_text)

# Remove exact normalized duplicate texts.
# Conflicting labels for the same text are reported first.
label_counts = df.groupby("normalized_text")["category"].nunique()
conflicting_texts = label_counts[label_counts > 1]

if len(conflicting_texts) > 0:
    print(
        "\nWARNING:",
        len(conflicting_texts),
        "normalized text entries have conflicting labels."
    )
    print(
        "These entries will be excluded because their labels disagree."
    )
    df = df[
        ~df["normalized_text"].isin(conflicting_texts.index)
    ].copy()

before_deduplication = len(df)

df = df.drop_duplicates(
    subset=["normalized_text"],
    keep="first"
).copy()

duplicates_removed = before_deduplication - len(df)

df = df.drop(columns=["normalized_text"]).reset_index(drop=True)

print("Duplicate texts removed:", duplicates_removed)
print("Cleaned dataset shape:", df.shape)

print("\nClass distribution:")
print(df["category"].value_counts())


# --------------------------------------------------
# 4. CHECK DATASET SIZE
# --------------------------------------------------

if df["category"].nunique() < 2:
    raise ValueError(
        "At least two different document categories are required."
    )

if df["category"].value_counts().min() < 2:
    raise ValueError(
        "Each category must contain at least two examples "
        "for a stratified train/test split."
    )


# --------------------------------------------------
# 5. SPLIT DATA BEFORE TRAINING
# --------------------------------------------------

X = df["text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n========== DATA SPLIT ==========")
print("Training documents:", len(X_train))
print("Testing documents:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 6. BUILD MACHINE LEARNING PIPELINE
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
])


# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

print("\n========== TRAINING MODEL ==========")

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 8. EVALUATE ON HELD-OUT TEST DATA
# --------------------------------------------------

print("\n========== MODEL EVALUATION ==========")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)

class_names = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names,
)

print("\nConfusion matrix (actual rows, predicted columns):")
print("Class order:", class_names)
print(cm)


# --------------------------------------------------
# 9. SAVE CONFUSION MATRIX IMAGE
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(
    xticks_rotation="vertical",
    values_format="d",
)

plt.title("PDF Document Classification - Test Results")
plt.tight_layout()
plt.savefig(MATRIX_PATH, dpi=150)
plt.close()

print("\nConfusion matrix saved to:", MATRIX_PATH)


# --------------------------------------------------
# 10. SHOW INCORRECT PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "text": X_test.to_numpy(),
    "actual_category": y_test.to_numpy(),
    "predicted_category": y_pred,
})

errors = results[
    results["actual_category"] != results["predicted_category"]
]

print("\n========== INCORRECT PREDICTIONS ==========")
print("Incorrect predictions:", len(errors))

if errors.empty:
    print(
        "No mistakes on this test split. "
        "This does not guarantee correct predictions on new documents."
    )
else:
    for _, row in errors.head(5).iterrows():
        print("\nActual:", row["actual_category"])
        print("Predicted:", row["predicted_category"])
        print("Text sample:", row["text"][:300])


# --------------------------------------------------
# 11. SAVE TRAINED MODEL
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\n========== MODEL SAVED ==========")
print("Model location:", MODEL_PATH)

print("\nTraining and evaluation completed.")
print(
    "\nIMPORTANT: Validate this model on genuinely unseen PDFs "
    "before relying on its predictions."
)