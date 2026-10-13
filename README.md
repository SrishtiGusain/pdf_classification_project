PDF Document Classification Using OCR and Machine Learning
Project Overview

This project is an AI-powered application that extracts text from PDF documents using Optical Character Recognition (OCR) and classifies them into three categories: Invoices, Bills, and Receipts.

The application uses Python, PyMuPDF, Tesseract OCR, TF-IDF, and Logistic Regression. A Streamlit web interface allows users to upload PDF files, extract their text, and view the predicted document category.

Features
Upload PDF documents through a web interface.
Extract text from digital and scanned PDFs using OCR when required.
Classify documents as invoices, bills, or receipts.
Display the extracted text and prediction probabilities.
Show a warning when the prediction is uncertain.
Use a trained machine learning model for classification.
Technologies Used
Python — Programming language
PyMuPDF — PDF text extraction
Tesseract OCR and Pytesseract — Text recognition from scanned documents
Pandas — Dataset handling and preprocessing
Scikit-learn — TF-IDF vectorization, Logistic Regression, and model evaluation
Joblib — Saving and loading the trained model
Streamlit — Web application interface
Project Structure
AiML/
├── streamli_run.py
├── extract_text.py
├── build_dataset.py
├── add_receipts.py
├── train_model.py
├── predict_document.py
├── check_dataset.py
├── check_duplicates.py
├── inspect_training_data.py
├── add_training_examples.py
├── requirements.txt
├── models/
│   └── document_classifier.joblib
├── data/
│   └── processed/
│       └── documents_combined.csv
└── docs/
    └── project_report.docx

This is an example structure. Adjust the filenames and folders to match the files actually included in your repository.

Installation and Setup
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd AiML

Replace YOUR_GITHUB_REPOSITORY_URL with your repository's actual URL.

2. Create and activate a virtual environment
python -m venv .venv

On Windows PowerShell:

.\.venv\Scripts\Activate.ps1
3. Install Python dependencies
python -m pip install -r requirements.txt
4. Install Tesseract OCR

Install Tesseract OCR separately and configure its executable path in the application if it is not automatically detected.

Example Windows path:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
5. Run the application
python -m streamlit run streamli_run.py

Open the local URL displayed in the terminal, usually http://localhost:8501.

Machine Learning Workflow
Collect and organize labeled invoice, bill, and receipt documents.
Extract text from PDFs using direct PDF extraction or OCR.
Clean and preprocess the extracted text.
Convert text into numerical features using TF-IDF.
Train a Logistic Regression classifier.
Evaluate predictions using accuracy, precision, recall, F1-score, and a confusion matrix.
Save the trained model and use it to classify uploaded documents.
Dataset

The project uses labeled invoice, bill, and receipt examples. One receipt data source investigated is the SROIE dataset.

Official SROIE dataset and competition page
SROIE dataset on Hugging Face

Please check the licenses and redistribution conditions of all datasets before sharing them publicly.

Limitations
OCR may produce incorrect text when a PDF is blurry, rotated, or low quality.
Similar document layouts may cause incorrect classifications.
Prediction probabilities are not guaranteed to represent true confidence.
Performance on a random test split may not reflect performance on new documents.
The model should be tested on independent, unseen PDFs before real-world use.
Future Improvements
Add more diverse and independently labeled training documents.
Improve OCR preprocessing for low-quality scans.
Test generalization on unseen documents and different templates.
Explore additional classification algorithms and confidence calibration.
Extend the system to support more document categories.
