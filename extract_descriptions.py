# expense-categorizer/extract_descriptions.py

import json

# Load the bKash data
with open('bkash_extracted_transactions_text.json', 'r') as f:
    transactions = json.load(f)

# Extract just descriptions
descriptions = [t['description'] for t in transactions]

# Save as test input
with open('test_descriptions.json', 'w') as f:
    json.dump(descriptions, f, indent=2)

print(f"Extracted {len(descriptions)} transaction descriptions")

# Also create a smaller sample for quick testing
sample = descriptions[:20]
with open('test_sample.json', 'w') as f:
    json.dump(sample, f, indent=2)

print(f"Created sample with 20 transactions")