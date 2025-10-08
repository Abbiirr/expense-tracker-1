import google.generativeai as genai
import json
import sys
from pathlib import Path
from typing import List, Dict
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)


def parse_pdf_with_gemini(pdf_path: str) -> List[Dict]:
    """
    Parse bKash PDF statement using Gemini API
    """
    # Use Gemini 1.5 Flash for better free quota
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Upload the file
    uploaded_file = genai.upload_file(pdf_path, mime_type="application/pdf")

    # Prompt for extraction
    prompt = """
    Extract ALL transactions from this bKash bank statement and return as JSON array.

    For EACH transaction line in the statement, create a JSON object with:
    - date: in DD-MMM-YY format (e.g., "01-Feb-25")
    - description: combine Transaction Type and Transaction Details into one string
    - withdrawal: the "Out" amount as float (0.0 if empty)
    - deposit: the "In" amount as float (0.0 if empty)  
    - balance: always 0.0

    Important rules:
    - Include ALL transactions, don't skip any
    - For withdrawals, use the TOTAL amount including fees (e.g., if transaction shows "1,000.00" with fee "-18.50", withdrawal should be 1018.50)
    - Mobile Recharge transactions should have withdrawal as 0.0 (they don't deduct from balance)
    - Pay Bill transactions should have withdrawal as 0.0
    - Ignore header rows and summary information
    - Return ONLY the JSON array, no other text

    Example output format:
    [
        {
            "date": "01-Feb-25",
            "description": "Make Payment Breadsmith Bakery-RM49603 / 651.00 8,751.86",
            "withdrawal": 651.0,
            "deposit": 0.0,
            "balance": 0.0
        }
    ]
    """

    # Generate response
    response = model.generate_content([uploaded_file, prompt])

    # Clean and parse response
    json_text = response.text.strip()

    # Remove markdown code blocks if present
    if json_text.startswith("```"):
        json_text = json_text.split("```")[1]
        if json_text.startswith("json"):
            json_text = json_text[4:]
    if json_text.endswith("```"):
        json_text = json_text.rsplit("```", 1)[0]

    # Parse JSON
    try:
        transactions = json.loads(json_text)
        return transactions
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {json_text[:500]}...")
        return []


def post_process_transactions(transactions: List[Dict]) -> List[Dict]:
    """
    Clean and standardize the extracted transactions
    """
    processed = []

    for trans in transactions:
        # Ensure all required fields exist
        processed_trans = {
            "date": trans.get("date", ""),
            "description": trans.get("description", ""),
            "withdrawal": float(trans.get("withdrawal", 0.0)),
            "deposit": float(trans.get("deposit", 0.0)),
            "balance": 0.0  # Always 0.0 as per requirement
        }

        # Clean up description - remove extra spaces
        processed_trans["description"] = " ".join(processed_trans["description"].split())

        # Special handling for certain transaction types
        desc = processed_trans["description"].lower()

        # Mobile Recharge and Pay Bill shouldn't have withdrawals
        if "mobile recharge" in desc or "pay bill" in desc:
            # These might show amounts but don't actually withdraw
            if "reversal" not in desc:
                processed_trans["withdrawal"] = 0.0

        # Cash Out transactions include fees
        if "cash out" in desc and "-" in processed_trans["description"]:
            # Fee is already included in the withdrawal amount from Gemini
            pass

        processed.append(processed_trans)

    return processed


def save_to_json(transactions: List[Dict], output_path: str = "bkash_transactions.json"):
    """
    Save transactions to JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transactions, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(transactions)} transactions to {output_path}")


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python bkash_parser.py <pdf_file_path> [output_json_path]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "bkash_transactions.json"

    # Verify PDF exists
    if not Path(pdf_path).exists():
        print(f"Error: PDF file '{pdf_path}' not found")
        sys.exit(1)

    print(f"Processing PDF: {pdf_path}")

    try:
        # Extract transactions using Gemini
        transactions = parse_pdf_with_gemini(pdf_path)

        if not transactions:
            print("No transactions extracted")
            sys.exit(1)

        # Post-process transactions
        transactions = post_process_transactions(transactions)

        # Save to JSON
        save_to_json(transactions, output_path)

        # Print summary
        total_withdrawals = sum(t["withdrawal"] for t in transactions)
        total_deposits = sum(t["deposit"] for t in transactions)

        print(f"\nSummary:")
        print(f"Total transactions: {len(transactions)}")
        print(f"Total withdrawals: {total_withdrawals:,.2f}")
        print(f"Total deposits: {total_deposits:,.2f}")
        print(f"Date range: {transactions[-1]['date']} to {transactions[0]['date']}")

    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()