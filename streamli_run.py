

import streamlit as st
import pymupdf
import pytesseract
from PIL import Image
from io import BytesIO
from pathlib import Path
import joblib

# ---------------------------------------
# 1. CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="PDF Document Classifier",
    page_icon="📄",
    layout="wide"
)

MODEL_PATH = Path("models/document_classifier.joblib")

# Tesseract OCR path for Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ---------------------------------------
# 2. LOAD TRAINED MODEL
# ---------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------------------------------
# 3. EXTRACT TEXT FROM PDF
# ---------------------------------------

def extract_pdf_text(pdf_bytes):
    extracted_pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as document:

        for page_number, page in enumerate(document, start=1):

            # First, extract embedded PDF text
            text = page.get_text().strip()

            # If little or no text is found, use OCR
            if len(text) < 20:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    BytesIO(pix.tobytes("png"))
                )

                text = pytesseract.image_to_string(
                    image
                ).strip()

            extracted_pages.append(
                f"--- Page {page_number} ---\n{text}"
            )

    return "\n\n".join(extracted_pages).strip()


# ---------------------------------------
# 4. APPLICATION HEADER
# ---------------------------------------

st.title("📄 PDF Document Classification")
st.write(
    "Upload a PDF to classify it as an Invoice, Bill, "
    "or Receipt using OCR and Machine Learning."
)

st.caption(
    "Workflow: PDF Upload → Text Extraction → "
    "TF-IDF Features → ML Prediction"
)

# ---------------------------------------
# 5. CHECK MODEL
# ---------------------------------------

if model is None:
    st.error(
        "Trained model not found. Please run "
        "'python train_model.py' first."
    )

    st.info(
        "Expected model path: "
        "models/document_classifier.joblib"
    )

# ---------------------------------------
# 6. PDF UPLOAD
# ---------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"],
    help="Choose an invoice, bill, or receipt PDF."
)

if uploaded_file is not None:

    st.success(f"Selected file: {uploaded_file.name}")

    if st.button(
        "Extract Text and Classify",
        type="primary"
    ):

        if model is None:
            st.error(
                "Cannot classify the document because "
                "the trained model is unavailable."
            )

        else:
            try:
                # Extract text
                with st.spinner(
                    "Extracting text from your PDF..."
                ):
                    pdf_bytes = uploaded_file.getvalue()
                    text = extract_pdf_text(pdf_bytes)

                if not text.strip():
                    st.error(
                        "No readable text was found. "
                        "Try a clearer PDF or a better scan."
                    )

                else:
                    # Display extracted text
                    st.subheader("1. Extracted Text")

                    st.text_area(
                        "Review the extracted document text",
                        value=text,
                        height=250
                    )

                    # Make prediction
                    with st.spinner(
                        "Classifying your document..."
                    ):
                        prediction = model.predict([text])[0]

                        if hasattr(model, "predict_proba"):
                            probabilities = (
                                model.predict_proba([text])[0]
                            )
                            classes = model.classes_
                        else:
                            probabilities = None
                            classes = []

                    # Display result
                    st.subheader("2. Classification Result")

                    st.success(
                        f"Predicted Category: "
                        f"{str(prediction).title()}"
                    )

                    # Display probability scores
                    if probabilities is not None:
                        st.subheader("3. Category Scores")

                        for category, probability in zip(
                            classes, probabilities
                        ):
                            percentage = probability * 100

                            st.write(
                                f"**{str(category).title()}:** "
                                f"{percentage:.2f}%"
                            )

                            st.progress(
                                float(probability),
                                text=f"{percentage:.2f}%"
                            )

                        # Confidence warning
                        confidence = max(probabilities)

                        if confidence < 0.50:
                            st.warning(
                                "Prediction uncertain. "
                                "Please verify the document manually."
                            )
                        else:
                            st.info(
                                "This is the model's highest-scoring "
                                "category, not a guarantee of correctness. "
                                "Please verify important documents."
                            )

                    else:
                        st.warning(
                            "This model does not provide probability "
                            "scores. Verify the prediction manually."
                        )

            except Exception as error:
                st.error(
                    "The PDF could not be processed. "
                    "Check that it is a valid PDF and that "
                    "OCR dependencies are installed correctly."
                )

                st.caption(
                    f"Technical details: {error}"
                )

# ---------------------------------------
# 7. FOOTER
# ---------------------------------------

st.divider()

st.caption(
    "PDF Document Classification Using OCR and Machine Learning | "
    "Categories: Invoice, Bill, Receipt"
)

st.caption(
    "This application assists with classification. "
    "Predictions may be incorrect and should be verified."
)