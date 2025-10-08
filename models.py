# expense-categorizer/models.py
"""
Data models and category definitions for expense classification
"""

from dataclasses import dataclass
from typing import Optional, Dict, List

# Default expense categories
DEFAULT_CATEGORIES = [
    "food",
    "cashout",
    "utility bill",
    "shopping",
    "cashback",
    "mobile recharge",
    "transportation",
    "entertainment",
    "healthcare",
    "education",
    "rent",
    "insurance",
    "others"
]

# Subcategory mappings for utility bills
UTILITY_SUBCATEGORIES = {
    "electricity": ["desco", "electric", "dpdc", "bpdb", "power"],
    "gas": ["titas", "gas bill", "gas transmission"],
    "water": ["water", "dwasa", "wasa"],
    "internet": ["internet", "broadband", "wifi", "isp"]
}

# Rule-based keyword mappings
KEYWORD_MAPPINGS = {
    "cashback": ["cashback", "cash back", "reward"],
    "cashout": ["cash out", "cash-out", "cashout", "atm withdrawal"],
    "mobile recharge": ["recharge", "top up", "top-up", "mobile", "flexiload"],
    "food": ["restaurant", "cafe", "food", "meal", "lunch", "dinner", "breakfast"],
    "transportation": ["uber", "pathao", "taxi", "bus", "train", "fuel", "petrol"],
    "healthcare": ["doctor", "hospital", "medicine", "pharmacy", "clinic"],
    "entertainment": ["movie", "cinema", "netflix", "spotify", "game"],
}


@dataclass
class TransactionResult:
    """Result of transaction classification"""
    description: str
    category: str
    subcategory: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "category": self.category,
            "subcategory": self.subcategory,
            "confidence": self.confidence
        }