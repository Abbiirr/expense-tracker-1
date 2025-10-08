# expense-categorizer/classifier.py
"""
Main classification engine for expense categorization using Ollama
"""

import logging
import json
from typing import List, Optional
import requests

from models import (
    DEFAULT_CATEGORIES,
    UTILITY_SUBCATEGORIES,
    KEYWORD_MAPPINGS,
    TransactionResult
)

LOG = logging.getLogger(__name__)


class ExpenseClassifier:
    """Expense classifier using Ollama LLM with rule-based refinements"""

    def __init__(
            self,
            model_name: str = "qwen3:8b",
            categories: Optional[List[str]] = None,
            ollama_host: str = "http://10.112.30.10:11434"
    ):
        """
        Initialize the expense classifier

        Args:
            model_name: Ollama model name (default: qwen3:8b)
            categories: List of expense categories (uses DEFAULT_CATEGORIES if None)
            ollama_host: Ollama API host URL
        """
        self.categories = categories or DEFAULT_CATEGORIES
        self.model_name = model_name
        self.ollama_host = ollama_host
        self._verify_ollama_connection()

    def _verify_ollama_connection(self):
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model_name not in model_names:
                    LOG.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                    LOG.info(f"Pulling model {self.model_name}...")
                    self._pull_model()
            else:
                raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_host}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Ollama is not running. Start it with: ollama serve")

    def _pull_model(self):
        """Pull the model if not available"""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json={"name": self.model_name}
            )
            if response.status_code == 200:
                LOG.info(f"Successfully pulled model {self.model_name}")
        except Exception as e:
            LOG.error(f"Failed to pull model: {e}")

    def classify(self, description: str, threshold: float = 0.0) -> TransactionResult:
        """
        Classify a single transaction description

        Args:
            description: Transaction description text
            threshold: Minimum confidence threshold (0-1)

        Returns:
            TransactionResult with category and optional subcategory
        """
        # First apply rule-based classification
        rule_category = self._apply_rules(description.lower())

        if rule_category:
            category = rule_category
            confidence = 0.95  # High confidence for rule-based matches
        else:
            # Use Ollama for classification
            category, confidence = self._classify_with_ollama(description)

            # Refine with partial rules
            category = self._refine_category(category, description.lower())

        # Check for subcategory
        subcategory = self._get_subcategory(category, description.lower())

        # Apply threshold
        if confidence < threshold:
            category = "others"
            subcategory = None

        return TransactionResult(
            description=description,
            category=category,
            subcategory=subcategory,
            confidence=confidence
        )

    def _classify_with_ollama(self, description: str) -> tuple[str, float]:
        """Use Ollama to classify the transaction"""
        categories_str = ", ".join(self.categories)

        prompt = f"""Classify the following transaction into EXACTLY ONE of these categories: {categories_str}

Transaction: {description}

Respond with ONLY the category name and confidence score (0-1) in JSON format:
{{"category": "...", "confidence": 0.X}}"""

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,  # Low temperature for consistent results
                    "top_p": 0.9
                }
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()

                # Try to parse JSON response
                try:
                    # Extract JSON from response if wrapped in other text
                    import re
                    json_match = re.search(r'\{[^}]+\}', response_text)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        category = parsed.get('category', 'others').lower()
                        confidence = float(parsed.get('confidence', 0.5))
                    else:
                        # Fallback: try to find category name in response
                        category = self._extract_category_from_text(response_text)
                        confidence = 0.7
                except:
                    # Fallback if JSON parsing fails
                    category = self._extract_category_from_text(response_text)
                    confidence = 0.6

                # Ensure category is valid
                if category not in [c.lower() for c in self.categories]:
                    category = "others"
                    confidence = 0.5

                return category, confidence
            else:
                LOG.error(f"Ollama API error: {response.status_code}")
                return "others", 0.5

        except Exception as e:
            LOG.error(f"Error calling Ollama: {e}")
            return "others", 0.5

    def _extract_category_from_text(self, text: str) -> str:
        """Extract category from non-JSON response"""
        text_lower = text.lower()
        for category in self.categories:
            if category.lower() in text_lower:
                return category.lower()
        return "others"

    def classify_batch(
            self,
            descriptions: List[str],
            threshold: float = 0.0
    ) -> List[TransactionResult]:
        """
        Classify multiple transaction descriptions

        Args:
            descriptions: List of transaction descriptions
            threshold: Minimum confidence threshold

        Returns:
            List of TransactionResult objects
        """
        results = []
        for desc in descriptions:
            results.append(self.classify(desc, threshold))
        return results

    def _apply_rules(self, description: str) -> Optional[str]:
        """Apply keyword-based rules for direct classification"""
        for category, keywords in KEYWORD_MAPPINGS.items():
            if any(keyword in description for keyword in keywords):
                return category
        return None

    def _refine_category(self, category: str, description: str) -> str:
        """Refine category based on specific keywords"""
        # Override model prediction if strong keywords are present
        if "cashback" in description or "cash back" in description:
            return "cashback"
        elif any(kw in description for kw in ["cash out", "cash-out", "cashout"]):
            return "cashout"
        elif any(kw in description for kw in ["recharge", "top up", "top-up"]):
            return "mobile recharge"
        return category

    def _get_subcategory(self, category: str, description: str) -> Optional[str]:
        """Determine subcategory for certain categories"""
        if category == "utility bill":
            for subcat, keywords in UTILITY_SUBCATEGORIES.items():
                if any(kw in description for kw in keywords):
                    return subcat
        return None


class SimpleClassifier:
    """Lightweight rule-based classifier for quick categorization"""

    def __init__(self, categories: Optional[List[str]] = None):
        self.categories = categories or DEFAULT_CATEGORIES

    def classify(self, description: str, threshold: float = 0.0) -> TransactionResult:
        """
        Simple rule-based classification without ML model

        Args:
            description: Transaction description
            threshold: Minimum confidence threshold

        Returns:
            TransactionResult with category
        """
        desc_lower = description.lower()

        # Check keyword mappings
        for category, keywords in KEYWORD_MAPPINGS.items():
            if any(keyword in desc_lower for keyword in keywords):
                subcategory = self._get_subcategory(category, desc_lower)
                return TransactionResult(
                    description=description,
                    category=category,
                    subcategory=subcategory,
                    confidence=0.9
                )

        # Default to others
        return TransactionResult(
            description=description,
            category="others",
            subcategory=None,
            confidence=0.5
        )

    def classify_batch(
            self,
            descriptions: List[str],
            threshold: float = 0.0
    ) -> List[TransactionResult]:
        """
        Classify multiple transaction descriptions

        Args:
            descriptions: List of transaction descriptions
            threshold: Minimum confidence threshold

        Returns:
            List of TransactionResult objects
        """
        results = []
        for desc in descriptions:
            results.append(self.classify(desc, threshold))
        return results

    def _get_subcategory(self, category: str, description: str) -> Optional[str]:
        """Determine subcategory for certain categories"""
        if category == "utility bill":
            for subcat, keywords in UTILITY_SUBCATEGORIES.items():
                if any(kw in description for kw in keywords):
                    return subcat
        return None