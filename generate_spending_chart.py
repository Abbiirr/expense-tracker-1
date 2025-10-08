#!/usr/bin/env python3
# generate_spending_chart.py
"""
Generate spending pie chart from bKash transactions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Default categories and keyword mappings
DEFAULT_CATEGORIES = [
    "food", "cashout", "utility bill", "shopping", "cashback",
    "mobile recharge", "transportation", "entertainment", "healthcare",
    "education", "rent", "insurance", "others"
]

KEYWORD_MAPPINGS = {
    "cashback": ["cashback", "cash back", "reward"],
    "cashout": ["cash out", "cash-out", "cashout", "atm withdrawal"],
    "mobile recharge": ["recharge", "top up", "top-up", "mobile", "flexiload"],
    "food": ["restaurant", "cafe", "food", "meal", "lunch", "dinner", "breakfast",
             "breadsmith", "emerald", "woodhouse", "sahibaan", "kasundi", "arabika",
             "coal & coffee", "dash dine", "shumis", "lantern", "meat bar", "foodpanda"],
    "transportation": ["uber", "pathao", "taxi", "bus", "train", "fuel", "petrol"],
    "healthcare": ["doctor", "hospital", "medicine", "pharmacy", "clinic"],
    "entertainment": ["movie", "cinema", "netflix", "spotify", "game", "vibe gaming"],
    "utility bill": ["desco", "titas", "gas", "electric", "water", "dwasa"],
    "shopping": ["shwapno", "meena bazar", "star tech", "ryans it", "walton",
                 "apex", "easy fashion", "paragon agro"],
    "education": ["university", "college", "school", "admission", "jagannath"]
}


class SimpleClassifier:
    """Fallback rule-based classifier"""

    def classify(self, description: str) -> tuple[str, float]:
        desc_lower = description.lower()

        # Check keyword mappings
        for category, keywords in KEYWORD_MAPPINGS.items():
            if any(keyword in desc_lower for keyword in keywords):
                return category, 0.9

        # Special checks for common patterns
        if "pay bill" in desc_lower:
            return "utility bill", 0.85
        elif "send money" in desc_lower:
            return "cashout", 0.8
        elif "make payment" in desc_lower:
            # Try to guess based on merchant name
            if any(food in desc_lower for food in ["restaurant", "cafe", "grill", "bakery"]):
                return "food", 0.75
            return "shopping", 0.6

        return "others", 0.5


class OllamaClassifier:
    """Ollama-based classifier with fallback"""

    def __init__(self, host="http://10.112.30.10:11434", model="qwen3:8b"):
        self.host = host
        self.model = model
        self.fallback = SimpleClassifier()
        self.is_available = self._check_ollama()

    def _check_ollama(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            LOG.warning("Ollama not available, using fallback classifier")
            return False

    def classify(self, description: str) -> tuple[str, float]:
        if not self.is_available:
            return self.fallback.classify(description)

        try:
            categories_str = ", ".join(DEFAULT_CATEGORIES)
            prompt = f"""Classify this transaction into ONE category: {categories_str}

Transaction: {description}

Respond with ONLY the category name."""

            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1
                },
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                category = result.get('response', '').strip().lower()

                # Validate category
                if category in [c.lower() for c in DEFAULT_CATEGORIES]:
                    return category, 0.95

            # Fallback if Ollama fails
            return self.fallback.classify(description)

        except Exception as e:
            LOG.debug(f"Ollama error: {e}")
            return self.fallback.classify(description)


def process_transactions(json_file: str, use_ollama: bool = True) -> Dict:
    """Process transactions and categorize spending"""

    # Load transactions
    with open(json_file, 'r') as f:
        transactions = json.load(f)

    # Initialize classifier
    if use_ollama:
        classifier = OllamaClassifier()
    else:
        classifier = SimpleClassifier()

    # Categorize and sum spending
    spending_by_category = {}
    total_spending = 0

    for transaction in transactions:
        # Only consider withdrawals (spending)
        amount = transaction.get('withdrawal', 0)
        if amount <= 0:
            continue

        description = transaction.get('description', '')
        category, confidence = classifier.classify(description)

        if category not in spending_by_category:
            spending_by_category[category] = {
                'amount': 0,
                'count': 0,
                'transactions': []
            }

        spending_by_category[category]['amount'] += amount
        spending_by_category[category]['count'] += 1
        spending_by_category[category]['transactions'].append({
            'date': transaction.get('date', ''),
            'description': description,
            'amount': amount,
            'confidence': confidence
        })

        total_spending += amount

    # Calculate percentages and prepare chart data
    chart_data = {
        'categories': [],
        'total': total_spending
    }

    for category, data in spending_by_category.items():
        percentage = (data['amount'] / total_spending) * 100 if total_spending > 0 else 0
        chart_data['categories'].append({
            'name': category.title(),
            'amount': round(data['amount'], 2),
            'percentage': round(percentage, 2),
            'count': data['count']
        })

    # Sort by amount
    chart_data['categories'].sort(key=lambda x: x['amount'], reverse=True)

    return chart_data


def generate_html_chart(chart_data: Dict) -> str:
    """Generate HTML with pie chart"""

    html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Expense Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }
        .total {
            text-align: center;
            font-size: 24px;
            color: #666;
            margin-bottom: 30px;
        }
        .currency {
            font-weight: bold;
            color: #667eea;
        }
        .chart-container {
            position: relative;
            height: 500px;
            margin-bottom: 40px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .stat-card h3 {
            margin: 0 0 5px 0;
            color: #333;
            font-size: 16px;
        }
        .stat-amount {
            font-size: 20px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-info {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Expense Analysis</h1>
        <div class="total">Total Spending: <span class="currency">BDT ''' + f"{chart_data['total']:,.2f}" + '''</span></div>

        <div class="chart-container">
            <canvas id="pieChart"></canvas>
        </div>

        <div class="stats">'''

    # Add category cards
    for cat in chart_data['categories'][:8]:  # Show top 8 categories
        html_template += f'''
            <div class="stat-card">
                <h3>{cat['name']}</h3>
                <div class="stat-amount">BDT {cat['amount']:,.0f}</div>
                <div class="stat-info">{cat['percentage']}% • {cat['count']} transactions</div>
            </div>'''

    html_template += '''
        </div>
    </div>

    <script>
        const data = ''' + json.dumps(chart_data) + ''';

        const ctx = document.getElementById('pieChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.categories.map(c => c.name),
                datasets: [{
                    data: data.categories.map(c => c.amount),
                    backgroundColor: [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF',
                        '#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            padding: 15,
                            font: { size: 12 },
                            generateLabels: function(chart) {
                                const data = chart.data;
                                return data.labels.map((label, i) => ({
                                    text: `${label} (${data.datasets[0].data[i].toLocaleString()})`,
                                    fillStyle: data.datasets[0].backgroundColor[i],
                                    hidden: false,
                                    index: i
                                }));
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const category = data.categories[context.dataIndex];
                                return [
                                    `${category.name}: BDT ${category.amount.toLocaleString()}`,
                                    `${category.percentage}% of total`,
                                    `${category.count} transactions`
                                ];
                            }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>'''

    return html_template


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate spending pie chart')
    parser.add_argument('--input', default='bkash_extracted_transactions_text.json',
                        help='Input JSON file')
    parser.add_argument('--output', default='spending_chart.html',
                        help='Output HTML file')
    parser.add_argument('--no-ollama', action='store_true',
                        help='Use only rule-based classification')

    args = parser.parse_args()

    # Process transactions
    LOG.info(f"Processing {args.input}...")
    chart_data = process_transactions(args.input, use_ollama=not args.no_ollama)

    # Generate HTML
    html = generate_html_chart(chart_data)

    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)

    LOG.info(f"Chart saved to {args.output}")

    # Print summary
    print("\n📊 Spending Summary:")
    print(f"Total: BDT {chart_data['total']:,.2f}\n")
    for cat in chart_data['categories'][:5]:
        print(f"{cat['name']:20} BDT {cat['amount']:10,.2f} ({cat['percentage']:5.1f}%) - {cat['count']} txns")