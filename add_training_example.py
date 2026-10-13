
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "documents_combined.csv"

df = pd.read_csv(DATA_PATH)

# Additional examples for improving coverage of English document formats
examples = [
    {
        "file_name": "extra_invoice_1",
        "category": "invoice",
        "text": "TAX INVOICE Invoice Number INV-102 Seller ABC Technologies Customer XYZ Company Subtotal 5000 Tax 900 Grand Total 5900 Payment Terms Net 30 days",
        "status": "ok",
    },
    {
        "file_name": "extra_invoice_2",
        "category": "invoice",
        "text": "INVOICE Invoice No WEB-2045 Bill To Sunrise Pvt Ltd Website Development 15000 Hosting Charges 2000 Subtotal 17000 Tax 3060 Total Due 20060",
        "status": "ok",
    },
    {
        "file_name": "extra_invoice_3",
        "category": "invoice",
        "text": "COMMERCIAL INVOICE Invoice ID C-501 Date 12 October 2026 Supplier ABC Traders Product quantity 10 Unit price 250 Tax 450 Total invoice amount 2950",
        "status": "ok",
    },
    {
        "file_name": "extra_bill_1",
        "category": "bill",
        "text": "ELECTRICITY BILL Consumer Number 123456 Billing Period September 2026 Units Consumed 250 Current Charges 1100 Amount Due 1300 Due Date 25 October",
        "status": "ok",
    },
    {
        "file_name": "extra_bill_2",
        "category": "bill",
        "text": "INTERNET BILL Account Number NET789 Monthly Broadband Charges 799 Router Rental 100 Total Amount Payable 899 Payment Due Date 20 October",
        "status": "ok",
    },
    {
        "file_name": "extra_bill_3",
        "category": "bill",
        "text": "TELEPHONE BILL Account Number 789012 Previous Balance 300 Current Monthly Charges 500 Taxes 90 Total Balance Due 890",
        "status": "ok",
    },
    {
        "file_name": "extra_receipt_1",
        "category": "receipt",
        "text": "PAYMENT RECEIPT Restaurant Food Corner Transaction ID TXN-9921 Food Items 650 GST 32.50 Total Paid 682.50 Payment Method UPI Payment Status SUCCESS Thank you Visit again",
        "status": "ok",
    },
    {
        "file_name": "extra_receipt_2",
        "category": "receipt",
        "text": "CASH RECEIPT Store Fresh Mart Receipt No RC-501 Rice 250 Milk 60 Bread 40 Total 350 Cash Received 500 Change 150 Thank you for shopping",
        "status": "ok",
    },
    {
        "file_name": "extra_receipt_3",
        "category": "receipt",
        "text": "PAYMENT RECEIPT Transaction ID PAY502 Amount Received 1200 Payment Method Credit Card Payment Status Completed Date 12 October 2026 Thank you",
        "status": "ok",
    },
]

extra_df = pd.DataFrame(examples)

# Avoid adding any exact text already present
existing_texts = set(
    df["text"].dropna().astype(str).str.strip().str.lower()
)

extra_df = extra_df[
    ~extra_df["text"].str.strip().str.lower().isin(existing_texts)
]

df = pd.concat([df, extra_df], ignore_index=True)

df.to_csv(DATA_PATH, index=False)

print("Additional examples added:", len(extra_df))
print("\nUpdated category counts:")
print(df["category"].value_counts())
print("\nDataset saved to:", DATA_PATH)

