#!/usr/bin/env python3
# streamlit_app.py
"""
Streamlit app for bKash expense analysis
Upload PDF → Extract transactions → Categorize → Visualize
"""

import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
import tempfile
import re
from pathlib import Path

# Page config
st.set_page_config(
    page_title="bKash Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Categories and mappings
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


@st.cache_data
def parse_pdf_with_gemini(pdf_bytes: bytes) -> List[Dict]:
    """Parse bKash PDF using Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found! Please set it in .env file")
        return []

    genai.configure(api_key=api_key)
    # Use same model as working bkash_parser.py
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        # Upload file to Gemini
        uploaded_file = genai.upload_file(tmp_path, mime_type="application/pdf")

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
        - For withdrawals, use the TOTAL amount including fees
        - Mobile Recharge transactions should have withdrawal as 0.0
        - Pay Bill transactions should have withdrawal as 0.0
        - Return ONLY the JSON array, no other text

        Example output format:
        [
            {
                "date": "01-Feb-25",
                "description": "Make Payment Breadsmith Bakery-RM49603 / TRX ID: CB13G67PZ9",
                "withdrawal": 651.0,
                "deposit": 0.0,
                "balance": 0.0
            }
        ]
        """

        response = model.generate_content([uploaded_file, prompt])
        json_text = response.text.strip()

        # Clean response
        if json_text.startswith("```"):
            json_text = json_text.split("```")[1]
            if json_text.startswith("json"):
                json_text = json_text[4:]
        if json_text.endswith("```"):
            json_text = json_text.rsplit("```", 1)[0]

        transactions = json.loads(json_text)

        # Clean up transactions
        for trans in transactions:
            trans["description"] = " ".join(trans["description"].split())
            desc = trans["description"].lower()
            if ("mobile recharge" in desc or "pay bill" in desc) and "reversal" not in desc:
                trans["withdrawal"] = 0.0

        return transactions

    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return []
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def classify_transaction(description: str) -> Tuple[str, float]:
    """Simple rule-based classifier"""
    desc_lower = description.lower()

    # Check keyword mappings
    for category, keywords in KEYWORD_MAPPINGS.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category, 0.9

    # Special patterns
    if "pay bill" in desc_lower:
        return "utility bill", 0.85
    elif "send money" in desc_lower:
        return "cashout", 0.8
    elif "make payment" in desc_lower:
        if any(food in desc_lower for food in ["restaurant", "cafe", "grill", "bakery"]):
            return "food", 0.75
        return "shopping", 0.6

    return "others", 0.5


def process_transactions(transactions: List[Dict]) -> Dict:
    """Categorize and aggregate spending"""
    spending_by_category = {}
    total_spending = 0

    for trans in transactions:
        amount = trans.get('withdrawal', 0)
        if amount <= 0:
            continue

        description = trans.get('description', '')
        category, confidence = classify_transaction(description)

        if category not in spending_by_category:
            spending_by_category[category] = {
                'amount': 0,
                'count': 0,
                'transactions': []
            }

        spending_by_category[category]['amount'] += amount
        spending_by_category[category]['count'] += 1
        spending_by_category[category]['transactions'].append({
            'date': trans.get('date', ''),
            'description': description,
            'amount': amount,
            'confidence': confidence
        })

        total_spending += amount

    # Prepare chart data
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
            'count': data['count'],
            'transactions': data['transactions']
        })

    chart_data['categories'].sort(key=lambda x: x['amount'], reverse=True)
    return chart_data


def create_visualizations(chart_data: Dict):
    """Create pie and bar charts"""
    if not chart_data['categories']:
        st.warning("No spending data to visualize")
        return

    df = pd.DataFrame(chart_data['categories'])

    col1, col2 = st.columns([3, 2])

    with col1:
        # Pie chart
        fig_pie = px.pie(
            df,
            values='amount',
            names='name',
            title='Spending Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hover_data={'percentage': True, 'count': True}
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=16, family='Arial Black', color='black'),
            hovertemplate='<b style="font-size:16px">%{label}</b><br>' +
                          '<span style="font-size:14px">Amount: ৳%{value:,.0f}</span><br>',
            hoverlabel=dict(
                bgcolor="black",
                font_size=16,
                font_family="Arial"
            )
        )

        fig_pie.update_layout(
            height=700,  # Increased from 600
            title=dict(
                text='Spending Distribution',
                font=dict(size=24, family='Arial Black')  # Bigger title
            ),
            font=dict(size=14),  # Larger default font
            legend=dict(
                font=dict(size=14),  # Bigger legend text
                orientation='v',
                yanchor='middle',
                y=0.5
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar chart
        fig_bar = go.Figure(data=[
            go.Bar(
                x=df['name'][:10],  # Top 10
                y=df['amount'][:10],
                text=[f"৳{x:,.0f}" for x in df['amount'][:10]],
                textposition='auto',
                marker_color=px.colors.qualitative.Set3[:10]
            )
        ])
        fig_bar.update_layout(
            title='Top Spending Categories',
            xaxis_title='Category',
            yaxis_title='Amount (BDT)',
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Spending summary
    st.subheader("📊 Spending Summary")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Spending", f"৳{chart_data['total']:,.0f}")
    with col2:
        st.metric("Categories", len(chart_data['categories']))
    with col3:
        st.metric("Top Category", df.iloc[0]['name'] if not df.empty else "N/A")
    with col4:
        st.metric("Top Amount", f"৳{df.iloc[0]['amount']:,.0f}" if not df.empty else "N/A")

    # Category details table
    st.subheader("📋 Category Details")
    display_df = df[['name', 'amount', 'percentage', 'count']].copy()
    display_df.columns = ['Category', 'Amount (BDT)', 'Percentage (%)', 'Transactions']
    display_df['Amount (BDT)'] = display_df['Amount (BDT)'].apply(lambda x: f"৳{x:,.0f}")
    display_df['Percentage (%)'] = display_df['Percentage (%)'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Expandable transaction details
    with st.expander("🔍 View Transaction Details"):
        selected_cat = st.selectbox("Select Category", df['name'].tolist())
        if selected_cat:
            cat_data = next(c for c in chart_data['categories'] if c['name'] == selected_cat)
            trans_df = pd.DataFrame(cat_data['transactions'])
            if not trans_df.empty:
                trans_df['amount'] = trans_df['amount'].apply(lambda x: f"৳{x:,.0f}")
                trans_df['confidence'] = trans_df['confidence'].apply(lambda x: f"{x:.0%}")
                st.dataframe(trans_df[['date', 'description', 'amount', 'confidence']],
                             use_container_width=True, hide_index=True)


def main():
    st.title("💰 bKash Expense Analyzer")
    st.markdown("Upload your bKash statement PDF to analyze your spending patterns")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Check API key
        api_key_status = "✅ Configured" if os.getenv("GEMINI_API_KEY") else "❌ Not Found"
        # st.info(f"Gemini API Key: {api_key_status}")

        if not os.getenv("GEMINI_API_KEY"):
            st.warning("Please set GEMINI_API_KEY in .env file")
            api_key = st.text_input("Or enter API key here:", type="password")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
                st.rerun()

        st.divider()
        st.caption("Supported categories:")
        for cat in DEFAULT_CATEGORIES:
            st.caption(f"• {cat.title()}")

    # Main content
    uploaded_file = st.file_uploader("Choose a bKash PDF statement", type="pdf")

    if uploaded_file:
        # Process button
        if st.button("🚀 Analyze Expenses", type="primary"):
            with st.spinner("Processing your statement..."):
                # Step 1: Extract transactions
                with st.status("Extracting transactions from PDF...", expanded=True) as status:
                    pdf_bytes = uploaded_file.read()
                    transactions = parse_pdf_with_gemini(pdf_bytes)

                    if transactions:
                        st.write(f"✅ Extracted {len(transactions)} transactions")
                        status.update(label="Extraction complete!", state="complete")
                    else:
                        st.error("Failed to extract transactions")
                        status.update(label="Extraction failed", state="error")
                        return

                # Step 2: Process and categorize
                with st.status("Categorizing transactions...", expanded=True) as status:
                    chart_data = process_transactions(transactions)
                    st.write(f"✅ Processed {len(chart_data['categories'])} categories")
                    st.write(f"💵 Total spending: ৳{chart_data['total']:,.2f}")
                    status.update(label="Categorization complete!", state="complete")

                # Store in session state
                st.session_state['transactions'] = transactions
                st.session_state['chart_data'] = chart_data

    # Display results if available
    if 'chart_data' in st.session_state:
        st.divider()
        create_visualizations(st.session_state['chart_data'])

        # Download options
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download Transactions (JSON)"):
                json_str = json.dumps(st.session_state['transactions'], indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="bkash_transactions.json",
                    mime="application/json"
                )

        with col2:
            if st.button("📊 Download Analysis (CSV)"):
                df = pd.DataFrame(st.session_state['chart_data']['categories'])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="expense_analysis.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()