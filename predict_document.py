
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "document_classifier.joblib"

# Load the saved model
model = joblib.load(MODEL_PATH)

print("===== DOCUMENT CLASSIFIER =====")
print("Enter document text to classify.")
print("Type 'exit' to stop.\n")


while True:
    print("\nPaste your document text below.")
    print("Type END on a separate line when finished.")
    print("Type EXIT to quit.")

    lines = []

    while True:
        line = input()

        if line.strip().upper() == "EXIT":
            print("Program closed.")
            raise SystemExit

        if line.strip().upper() == "END":
            break

        lines.append(line)

    text = "\n".join(lines).strip()

    if not text:
        print("Please enter some document text.")
        continue

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    print("\n===== PREDICTION RESULTS =====")
    print("Predicted category:", prediction)
    print("\nScores for each category:")

    for category, probability in zip(
        model.classes_, probabilities
    ):
        print(f"{category}: {probability * 100:.2f}%")