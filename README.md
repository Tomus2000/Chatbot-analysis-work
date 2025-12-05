# Chatbot Conversation Analyzer

A comprehensive Streamlit dashboard for analyzing internal chatbot conversations about photovoltaic systems and heat pumps.

## Features

- **Excel File Upload**: Upload and analyze chatbot conversation data
- **OpenAI Integration**: Automatic sentiment analysis, happiness scoring, and topic clustering
- **Interactive Visualizations**: Multiple charts and graphs using Plotly
- **Advanced Filtering**: Filter by date, product type, channel, sentiment, and happiness
- **Topic Analysis**: Automatic topic identification using embeddings and clustering
- **Time Series Analysis**: Track trends over time
- **Data Export**: Export enriched data to CSV or Excel

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Streamlit secrets:
Create a `.streamlit/secrets.toml` file (or configure in Streamlit Cloud) with:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Data Format

Your Excel file should contain at least these columns:

**Required:**
- `conversation_id`: Unique identifier for each conversation
- `timestamp`: Date and time of the conversation
- `user_question`: The question asked by the user
- `bot_answer`: The chatbot's response

**Optional:**
- `channel`: Communication channel (e.g., "web", "mobile", "sales-app")
- `product_type`: Type of product (e.g., "photovoltaic", "heat pump", "other")
- `sentiment_label`: Pre-existing sentiment label
- `sentiment_score`: Pre-existing sentiment score (-1.0 to 1.0)
- `user_happiness`: Pre-existing happiness label
- `happiness_score`: Pre-existing happiness score (0.0 to 1.0)
- `csat_score`: Customer satisfaction score

## App Structure

The app consists of 6 main tabs:

1. **Overview**: Key metrics and high-level statistics
2. **Sentiment & Happiness**: Detailed sentiment and happiness analysis
3. **Topics & Segments**: Topic clustering and visualization
4. **Time Series**: Temporal trends and patterns
5. **Tables / Raw Data**: Interactive data tables and export functionality
6. **Diagnostics / Data Quality**: Data quality metrics and diagnostics

## Configuration

Column names can be adjusted in the `COLUMN_CONFIG` dictionary at the top of `app.py` if your Excel file uses different column names.

