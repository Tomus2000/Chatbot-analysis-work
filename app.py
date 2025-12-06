"""
Streamlit Dashboard for Analyzing Internal Chatbot Conversations
About Photovoltaic Systems and Heat Pumps

This app allows users to upload Excel files containing chatbot conversations,
analyze them using OpenAI API, and visualize insights through multiple interactive tabs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import openai
from openai import OpenAI
import io
import re
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Email functionality (placeholder - will be configured later)
EMAIL_API_AVAILABLE = False
EMAIL_API_CONFIG = None

# Try to import sklearn, make it optional
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Column name mappings (adjustable)
COLUMN_CONFIG = {
    'conversation_id': 'conversation_id',
    'timestamp': 'timestamp',
    'channel': 'channel',
    'product_type': 'product_type',
    'user_question': 'user_question',
    'bot_answer': 'bot_answer',
    'sentiment_label': 'sentiment_label',
    'sentiment_score': 'sentiment_score',
    'user_happiness': 'user_happiness',
    'happiness_score': 'happiness_score',
    'csat_score': 'csat_score',
}

# Default values for missing data
DEFAULT_PRODUCT_TYPE = 'other'
DEFAULT_CHANNEL = 'unknown'

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Chatbot Conversation Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Load and parse Excel file into DataFrame."""
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def standardize_column_names(df):
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
    return df

def infer_product_type(question_text):
    """Infer product type from question text."""
    if pd.isna(question_text) or not question_text:
        return DEFAULT_PRODUCT_TYPE
    
    question_lower = str(question_text).lower()
    
    # Keywords for photovoltaic
    pv_keywords = ['pv', 'photovoltaic', 'solar panel', 'solar panels', 'solar system', 
                   'solar energy', 'inverter', 'battery', 'solar', 'panel', 'panels',
                   'kwh', 'kw system', 'energy production', 'solar output']
    
    # Keywords for heat pump
    hp_keywords = ['heat pump', 'heatpump', 'heating', 'cooling', 'refrigerant', 
                   'servicing', 'service', 'maintenance', 'heating system']
    
    pv_score = sum(1 for keyword in pv_keywords if keyword in question_lower)
    hp_score = sum(1 for keyword in hp_keywords if keyword in question_lower)
    
    if pv_score > hp_score and pv_score > 0:
        return 'photovoltaic'
    elif hp_score > pv_score and hp_score > 0:
        return 'heat pump'
    else:
        return DEFAULT_PRODUCT_TYPE

def preprocess_data(df):
    """Preprocess the loaded DataFrame."""
    if df is None or df.empty:
        return None
    
    df = df.copy()
    
    # Standardize column names
    df = standardize_column_names(df)
    
    # Explicit mappings for common column name variations
    explicit_mappings = {
        'question': 'user_question',
        'questions': 'user_question',
        'user_question': 'user_question',
        'answer': 'bot_answer',
        'answers': 'bot_answer',
        'bot_answer': 'bot_answer',
        'response': 'bot_answer',
    }
    
    # Apply explicit mappings first
    for old_name, new_name in explicit_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Map columns to expected names (for other columns)
    column_mapping = {}
    for expected, config_key in COLUMN_CONFIG.items():
        # Skip if already mapped or exists
        if expected in df.columns:
            continue
        # Try exact match first
        if config_key in df.columns:
            if expected != config_key:
                column_mapping[config_key] = expected
        else:
            # Try fuzzy matching
            for col in df.columns:
                if expected.lower() in col.lower() or col.lower() in expected.lower():
                    if col not in explicit_mappings.values():  # Don't remap already mapped columns
                        column_mapping[col] = expected
                    break
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])  # Drop rows with invalid timestamps
        df['date'] = df['timestamp'].dt.date
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.day_name()
        df['weekday_num'] = df['timestamp'].dt.dayofweek
    else:
        st.warning("No timestamp column found. Time-based analysis will be limited.")
        df['date'] = None
        df['hour_of_day'] = None
        df['weekday'] = None
        df['weekday_num'] = None
    
    # Handle missing optional columns
    if 'product_type' not in df.columns:
        # Try to infer product type from question text
        df['product_type'] = df['user_question'].apply(infer_product_type)
    else:
        df['product_type'] = df['product_type'].fillna(DEFAULT_PRODUCT_TYPE)
    
    if 'channel' not in df.columns:
        df['channel'] = DEFAULT_CHANNEL
    else:
        df['channel'] = df['channel'].fillna(DEFAULT_CHANNEL)
    
    # Ensure text columns exist
    if 'user_question' not in df.columns:
        st.error("Required column 'user_question' not found in the file.")
        return None
    
    if 'bot_answer' not in df.columns:
        df['bot_answer'] = ''
    
    # Clean text columns
    df['user_question'] = df['user_question'].astype(str).fillna('')
    df['bot_answer'] = df['bot_answer'].astype(str).fillna('')
    
    # Handle sentiment columns
    if 'sentiment_label' not in df.columns:
        df['sentiment_label'] = None
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = None
    
    # Handle happiness columns
    if 'user_happiness' not in df.columns:
        df['user_happiness'] = None
    if 'happiness_score' not in df.columns:
        df['happiness_score'] = None
    
    # Handle CSAT
    if 'csat_score' not in df.columns:
        df['csat_score'] = None
    
    # Ensure conversation_id exists
    if 'conversation_id' not in df.columns:
        df['conversation_id'] = range(len(df))
    
    return df

# ============================================================================
# OPENAI ANALYSIS FUNCTIONS
# ============================================================================

def get_openai_client():
    """Initialize OpenAI client from Streamlit secrets or session state."""
    api_key = None
    
    # Try to get from session state first (set via UI)
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        api_key = st.session_state.openai_api_key
    # Try to get from secrets
    elif hasattr(st, 'secrets'):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass
    
    if not api_key:
        return None
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return None

@st.cache_data
def analyze_sentiment_openai(client, text, batch_size=10):
    """Analyze sentiment using OpenAI API with improved prompts."""
    if not text or pd.isna(text) or text.strip() == '':
        return {"label": "neutral", "score": 0.0}
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert sentiment analysis system for customer service conversations in the renewable energy sector (photovoltaic systems and heat pumps). 

Your task is to analyze the sentiment of customer questions and interactions with precision. Consider:
- Positive sentiment: enthusiasm, satisfaction, interest, appreciation, helpful responses
- Neutral sentiment: factual questions, information requests, neutral inquiries
- Negative sentiment: frustration, complaints, problems, dissatisfaction, concerns

Respond with ONLY a JSON object containing:
- 'label': one of "positive", "neutral", or "negative" (lowercase)
- 'score': a float between -1.0 (very negative) and 1.0 (very positive), where 0.0 is neutral

Be nuanced: a question about a problem doesn't necessarily mean negative sentiment if asked politely."""
                },
                {
                    "role": "user",
                    "content": f"Analyze the sentiment of this customer service interaction:\n\n{text[:2000]}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        label = result.get("label", "neutral").lower()
        score = float(result.get("score", 0.0))
        
        # Ensure label is valid
        if label not in ["positive", "neutral", "negative"]:
            label = "neutral"
        
        # Clamp score to valid range
        score = max(-1.0, min(1.0, score))
        
        return {
            "label": label,
            "score": score
        }
    except Exception as e:
        st.warning(f"Error analyzing sentiment: {str(e)}")
        return {"label": "neutral", "score": 0.0}

@st.cache_data
def analyze_happiness_openai(client, question, answer):
    """Analyze user happiness/satisfaction using OpenAI API with improved prompt."""
    combined_text = f"Question: {question}\nAnswer: {answer}"
    
    if not combined_text.strip() or (not question.strip() and not answer.strip()):
        return {"label": "neutral", "score": 0.5}
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a customer satisfaction analyst specializing in technical support for photovoltaic systems and heat pumps.

Analyze the user's question and the chatbot's answer to determine satisfaction. Consider:
- Does the answer fully address the question?
- Is the answer clear and helpful?
- Does the user seem satisfied with the response?
- Are there signs of confusion, frustration, or appreciation?

Respond with ONLY a JSON object containing:
- 'label': one of "happy", "neutral", or "unhappy"
- 'score': a float between 0.0 (very unhappy) and 1.0 (very happy)

Be accurate: "happy" means the user is satisfied and the answer was helpful. "unhappy" means frustration, confusion, or dissatisfaction."""
                },
                {
                    "role": "user",
                    "content": f"User Question: {question}\n\nChatbot Answer: {answer}\n\nAnalyze the user's happiness/satisfaction with this interaction."
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        label = result.get("label", "neutral").lower()
        score = float(result.get("score", 0.5))
        
        # Ensure label is valid
        if label not in ["happy", "neutral", "unhappy"]:
            label = "neutral"
        
        # Clamp score to valid range
        score = max(0.0, min(1.0, score))
        
        return {
            "label": label,
            "score": score
        }
    except Exception as e:
        st.warning(f"Error analyzing happiness: {str(e)}")
        return {"label": "neutral", "score": 0.5}

@st.cache_data
def get_embeddings_openai(client, texts, batch_size=100):
    """Get embeddings for texts using OpenAI API."""
    if not texts or len(texts) == 0:
        return []
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            st.warning(f"Error getting embeddings for batch {i//batch_size + 1}: {str(e)}")
            # Fill with zeros if error
            embeddings.extend([[0.0] * 1536] * len(batch))
    
    return embeddings

def enrich_data_with_openai(df, recompute_sentiment=False, recompute_happiness=False):
    """Enrich DataFrame with OpenAI analysis."""
    client = get_openai_client()
    if client is None:
        st.warning("⚠️ OpenAI API key not set. Please enter your API key in the sidebar to enable analysis.")
        return df
    
    df = df.copy()
    total_rows = len(df)
    
    # Sentiment analysis
    needs_sentiment = df['sentiment_label'].isna() | recompute_sentiment
    sentiment_count = needs_sentiment.sum()
    
    if sentiment_count > 0:
        st.info(f"Analyzing sentiment for {sentiment_count} rows...")
        progress_bar = st.progress(0)
        
        sentiment_results = []
        for idx, (i, row) in enumerate(df[needs_sentiment].iterrows()):
            text = str(row.get('user_question', '')) + ' ' + str(row.get('bot_answer', ''))
            result = analyze_sentiment_openai(client, text)
            sentiment_results.append((i, result))
            progress_bar.progress((idx + 1) / sentiment_count)
        
        for i, result in sentiment_results:
            df.at[i, 'sentiment_label'] = result['label']
            df.at[i, 'sentiment_score'] = result['score']
        
        progress_bar.empty()
    
    # Happiness analysis
    needs_happiness = df['user_happiness'].isna() | recompute_happiness
    happiness_count = needs_happiness.sum()
    
    if happiness_count > 0:
        st.info(f"Analyzing user happiness for {happiness_count} rows...")
        progress_bar = st.progress(0)
        
        happiness_results = []
        for idx, (i, row) in enumerate(df[needs_happiness].iterrows()):
            question = str(row.get('user_question', ''))
            answer = str(row.get('bot_answer', ''))
            result = analyze_happiness_openai(client, question, answer)
            happiness_results.append((i, result))
            progress_bar.progress((idx + 1) / happiness_count)
        
        for i, result in happiness_results:
            df.at[i, 'user_happiness'] = result['label']
            df.at[i, 'happiness_score'] = result['score']
        
        progress_bar.empty()
    
    return df

# ============================================================================
# TOPIC ANALYSIS
# ============================================================================

@st.cache_data
def compute_topics(df, n_clusters=5):
    """Compute topics using embeddings and clustering."""
    if not SKLEARN_AVAILABLE:
        st.error("❌ scikit-learn is not installed. Please install it with: pip install scikit-learn")
        return df
    
    client = get_openai_client()
    if client is None:
        st.warning("⚠️ OpenAI API key not set. Please enter your API key in the sidebar to enable topic clustering.")
        return df
    
    df = df.copy()
    
    # Get embeddings for user questions
    questions = df['user_question'].astype(str).tolist()
    st.info(f"Computing embeddings for {len(questions)} questions...")
    
    embeddings = get_embeddings_openai(client, questions)
    
    if not embeddings or len(embeddings) == 0:
        st.error("Failed to compute embeddings.")
        return df
    
    # Perform clustering
    st.info(f"Clustering into {n_clusters} topics...")
    embeddings_array = np.array(embeddings)
    
    # Standardize embeddings
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)
    
    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    topic_ids = kmeans.fit_predict(embeddings_scaled)
    
    df['topic_id'] = topic_ids
    
    # Generate topic labels using OpenAI
    st.info("Generating topic labels...")
    topic_labels = {}
    
    for topic_id in range(n_clusters):
        # Get sample questions for this topic
        topic_questions = df[df['topic_id'] == topic_id]['user_question'].head(10).tolist()
        sample_text = "\n".join([f"- {q[:200]}" for q in topic_questions[:5]])
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a topic labeling expert. Based on the sample questions provided, generate a short, descriptive label (2-4 words) for this topic. Focus on photovoltaic systems and heat pumps if relevant. Respond with ONLY the label text, no additional explanation."
                    },
                    {
                        "role": "user",
                        "content": f"Sample questions:\n{sample_text}"
                    }
                ],
                temperature=0.5
            )
            label = response.choices[0].message.content.strip()
            topic_labels[topic_id] = label
        except Exception as e:
            topic_labels[topic_id] = f"Topic {topic_id + 1}"
    
    df['topic_label'] = df['topic_id'].map(topic_labels)
    
    # Store embeddings for visualization
    df['embedding'] = embeddings
    
    return df

# ============================================================================
# FILTERING
# ============================================================================

def filter_data(df, filters):
    """Apply filters to DataFrame."""
    if df is None or df.empty:
        return df
    
    df_filtered = df.copy()
    
    # Date range filter
    if filters.get('date_range') and len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        if start_date and end_date:
            df_filtered = df_filtered[
                (df_filtered['date'] >= start_date) & 
                (df_filtered['date'] <= end_date)
            ]
    
    # Product type filter
    if filters.get('product_types') and len(filters['product_types']) > 0:
        df_filtered = df_filtered[df_filtered['product_type'].isin(filters['product_types'])]
    
    # Channel filter
    if filters.get('channels') and len(filters['channels']) > 0:
        df_filtered = df_filtered[df_filtered['channel'].isin(filters['channels'])]
    
    # Sentiment filter
    if filters.get('sentiments') and len(filters['sentiments']) > 0:
        df_filtered = df_filtered[df_filtered['sentiment_label'].isin(filters['sentiments'])]
    
    # Happiness filter
    if filters.get('happiness') and len(filters['happiness']) > 0:
        df_filtered = df_filtered[df_filtered['user_happiness'].isin(filters['happiness'])]
    
    return df_filtered

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_sentiment_distribution(df):
    """Plot sentiment distribution."""
    if df is None or df.empty or 'sentiment_label' not in df.columns:
        return None
    
    sentiment_counts = df['sentiment_label'].value_counts()
    
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title="Sentiment Distribution",
        labels={'x': 'Sentiment', 'y': 'Count'},
        color=sentiment_counts.index,
        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_happiness_distribution(df):
    """Plot happiness distribution."""
    if df is None or df.empty or 'user_happiness' not in df.columns:
        return None
    
    happiness_counts = df['user_happiness'].value_counts()
    
    fig = px.bar(
        x=happiness_counts.index,
        y=happiness_counts.values,
        title="User Happiness Distribution",
        labels={'x': 'Happiness', 'y': 'Count'},
        color=happiness_counts.index,
        color_discrete_map={'happy': 'green', 'neutral': 'gray', 'unhappy': 'red'}
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_sentiment_by_product(df):
    """Plot sentiment by product type."""
    if df is None or df.empty or 'sentiment_label' not in df.columns:
        return None
    
    crosstab = pd.crosstab(df['product_type'], df['sentiment_label'])
    
    fig = px.bar(
        crosstab,
        title="Sentiment by Product Type",
        labels={'value': 'Count', 'index': 'Product Type'},
        barmode='stack'
    )
    return fig

def plot_happiness_by_product(df):
    """Plot happiness by product type."""
    if df is None or df.empty or 'user_happiness' not in df.columns:
        return None
    
    crosstab = pd.crosstab(df['product_type'], df['user_happiness'])
    
    fig = px.bar(
        crosstab,
        title="Happiness by Product Type",
        labels={'value': 'Count', 'index': 'Product Type'},
        barmode='stack'
    )
    return fig

def plot_sentiment_over_time(df):
    """Plot sentiment score over time."""
    if df is None or df.empty or 'date' not in df.columns or 'sentiment_score' not in df.columns:
        return None
    
    daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()
    
    fig = px.line(
        daily_sentiment,
        x='date',
        y='sentiment_score',
        title="Average Sentiment Score Over Time",
        labels={'sentiment_score': 'Average Sentiment Score', 'date': 'Date'}
    )
    return fig

def plot_volume_over_time(df):
    """Plot question volume over time."""
    if df is None or df.empty or 'date' not in df.columns:
        return None
    
    daily_volume = df.groupby('date').size().reset_index(name='count')
    
    fig = px.line(
        daily_volume,
        x='date',
        y='count',
        title="Question Volume Over Time",
        labels={'count': 'Number of Questions', 'date': 'Date'}
    )
    return fig

def plot_volume_by_product_over_time(df):
    """Plot question volume by product type over time."""
    if df is None or df.empty or 'date' not in df.columns:
        return None
    
    daily_volume = df.groupby(['date', 'product_type']).size().reset_index(name='count')
    
    fig = px.line(
        daily_volume,
        x='date',
        y='count',
        color='product_type',
        title="Question Volume by Product Type Over Time",
        labels={'count': 'Number of Questions', 'date': 'Date'}
    )
    return fig

def plot_heatmap_weekday_hour(df, metric='count'):
    """Plot heatmap of weekday vs hour."""
    if df is None or df.empty or 'weekday' not in df.columns or 'hour_of_day' not in df.columns:
        return None
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
    
    if metric == 'count':
        heatmap_data = df.groupby(['weekday', 'hour_of_day']).size().reset_index(name='count')
        z_values = 'count'
        title = "Question Count by Weekday and Hour"
    else:
        heatmap_data = df.groupby(['weekday', 'hour_of_day'])['sentiment_score'].mean().reset_index()
        heatmap_data.columns = ['weekday', 'hour_of_day', 'avg_sentiment']
        z_values = 'avg_sentiment'
        title = "Average Sentiment by Weekday and Hour"
    
    pivot = heatmap_data.pivot(index='weekday', columns='hour_of_day', values=z_values)
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Weekday", color=z_values),
        title=title,
        aspect="auto"
    )
    return fig

def plot_topic_distribution(df):
    """Plot topic distribution."""
    if df is None or df.empty or 'topic_label' not in df.columns:
        return None
    
    topic_counts = df['topic_label'].value_counts()
    
    fig = px.bar(
        x=topic_counts.values,
        y=topic_counts.index,
        orientation='h',
        title="Questions per Topic",
        labels={'x': 'Count', 'y': 'Topic'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def plot_embeddings_2d(df, color_by='topic_id'):
    """Plot 2D projection of embeddings."""
    if df is None or df.empty or 'embedding' not in df.columns:
        return None
    
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn required for embedding visualization")
        return None
    
    try:
        embeddings = np.array(df['embedding'].tolist())
        
        # PCA to 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plot_df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'color': df[color_by] if color_by in df.columns else 'all'
        })
        
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='color',
            title=f"2D Embedding Projection (colored by {color_by})",
            labels={'x': 'PC1', 'y': 'PC2'}
        )
        return fig
    except Exception as e:
        st.warning(f"Error creating embedding plot: {str(e)}")
        return None

def plot_sentiment_vs_csat(df):
    """Plot sentiment score vs CSAT score."""
    if df is None or df.empty:
        return None
    
    if 'sentiment_score' not in df.columns or 'csat_score' not in df.columns:
        return None
    
    valid_data = df.dropna(subset=['sentiment_score', 'csat_score'])
    if valid_data.empty:
        return None
    
    fig = px.scatter(
        valid_data,
        x='sentiment_score',
        y='csat_score',
        title="Sentiment Score vs CSAT Score",
        labels={'sentiment_score': 'Sentiment Score', 'csat_score': 'CSAT Score'},
        trendline="ols"
    )
    return fig

# ============================================================================
# TAB BUILDING FUNCTIONS
# ============================================================================

def build_overview_tab(df):
    """Build the Overview tab."""
    st.header("📊 Overview")
    
    if df is None or df.empty:
        st.warning("No data available. Please upload a file.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Conversations", len(df))
    
    with col2:
        st.metric("Unique Conversation IDs", df['conversation_id'].nunique())
    
    with col3:
        if 'sentiment_score' in df.columns and df['sentiment_score'].notna().any():
            avg_sentiment = df['sentiment_score'].mean()
            st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
        else:
            st.metric("Average Sentiment Score", "N/A")
    
    with col4:
        if 'user_happiness' in df.columns:
            happy_count = (df['user_happiness'] == 'happy').sum()
            happy_pct = (happy_count / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Happy Users", f"{happy_pct:.1f}%")
        else:
            st.metric("Happy Users", "N/A")
    
    st.divider()
    
    # Product type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Questions by Product Type")
        product_counts = df['product_type'].value_counts()
        fig = px.bar(
            x=product_counts.index,
            y=product_counts.values,
            labels={'x': 'Product Type', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_product_type")
    
    with col2:
        st.subheader("Questions by Channel")
        channel_counts = df['channel'].value_counts()
        fig = px.bar(
            x=channel_counts.index,
            y=channel_counts.values,
            labels={'x': 'Channel', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_channel")
    
    # Sentiment and happiness overview
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sentiment_label' in df.columns and df['sentiment_label'].notna().any():
            fig = plot_sentiment_distribution(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="overview_sentiment_dist")
    
    with col2:
        if 'user_happiness' in df.columns and df['user_happiness'].notna().any():
            fig = plot_happiness_distribution(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="overview_happiness_dist")

def build_sentiment_tab(df):
    """Build the Sentiment & Happiness tab."""
    st.header("😊 Sentiment & Happiness Analysis")
    
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sentiment_label' in df.columns and df['sentiment_label'].notna().any():
            fig = plot_sentiment_distribution(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="sentiment_tab_sentiment_dist")
    
    with col2:
        if 'user_happiness' in df.columns and df['user_happiness'].notna().any():
            fig = plot_happiness_distribution(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="sentiment_tab_happiness_dist")
    
    # Cross-tabulations
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sentiment_label' in df.columns and df['sentiment_label'].notna().any():
            fig = plot_sentiment_by_product(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="sentiment_tab_sentiment_by_product")
    
    with col2:
        if 'user_happiness' in df.columns and df['user_happiness'].notna().any():
            fig = plot_happiness_by_product(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="sentiment_tab_happiness_by_product")
    
    # Sentiment vs CSAT
    if 'sentiment_score' in df.columns and 'csat_score' in df.columns:
        fig = plot_sentiment_vs_csat(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="sentiment_tab_sentiment_vs_csat")
    
    # Statistics table
    st.subheader("Sentiment Statistics")
    if 'sentiment_label' in df.columns and df['sentiment_label'].notna().any():
        sentiment_stats = df.groupby('sentiment_label').agg({
            'sentiment_score': ['mean', 'std', 'count'] if 'sentiment_score' in df.columns else 'count'
        }).round(2)
        st.dataframe(sentiment_stats, use_container_width=True, key="sentiment_tab_stats_table")

def build_topics_tab(df):
    """Build the Topics & Segments tab."""
    st.header("🏷️ Topics & Segments")
    
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    if 'topic_label' not in df.columns:
        st.info("Topics have not been computed yet. Use the sidebar to compute topics.")
        return
    
    # Topic distribution
    fig = plot_topic_distribution(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="topic_distribution")
    
    # Topic statistics
    st.subheader("Topic Statistics")
    topic_stats = df.groupby('topic_label').agg({
        'sentiment_score': 'mean' if 'sentiment_score' in df.columns else lambda x: 0,
        'happiness_score': 'mean' if 'happiness_score' in df.columns else lambda x: 0,
        'user_question': 'count'
    }).round(2)
    topic_stats.columns = ['Avg Sentiment', 'Avg Happiness', 'Count']
    topic_stats = topic_stats.sort_values('Count', ascending=False)
    st.dataframe(topic_stats, use_container_width=True, key="topic_stats_table")
    
    # Example questions per topic
    st.subheader("Example Questions by Topic")
    selected_topic = st.selectbox("Select a topic to view examples:", df['topic_label'].unique(), key="topic_selector")
    
    topic_questions = df[df['topic_label'] == selected_topic]['user_question'].head(10)
    for i, q in enumerate(topic_questions, 1):
        st.write(f"{i}. {q[:200]}...")
    
    # Embedding visualizations
    st.subheader("Topic Visualization (2D Embedding Projection)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        color_by = st.radio("Color by:", ['topic_id', 'product_type', 'sentiment_label'], key='embedding_color')
    
    if 'embedding' in df.columns:
        fig = plot_embeddings_2d(df, color_by=color_by)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"embeddings_2d_{color_by}")

def build_time_tab(df):
    """Build the Time Series tab."""
    st.header("📈 Time Series Analysis")
    
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    if 'date' not in df.columns or df['date'].isna().all():
        st.warning("No timestamp data available for time series analysis.")
        return
    
    # Sentiment over time
    if 'sentiment_score' in df.columns and df['sentiment_score'].notna().any():
        fig = plot_sentiment_over_time(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="sentiment_over_time")
    
    # Volume over time
    fig = plot_volume_over_time(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="volume_over_time")
    
    # Volume by product over time
    fig = plot_volume_by_product_over_time(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="volume_by_product_time")
    
    # Heatmaps
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_heatmap_weekday_hour(df, metric='count')
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="heatmap_count")
    
    with col2:
        if 'sentiment_score' in df.columns and df['sentiment_score'].notna().any():
            fig = plot_heatmap_weekday_hour(df, metric='sentiment')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="heatmap_sentiment")

def build_tables_tab(df):
    """Build the Tables / Raw Data tab."""
    st.header("📋 Tables & Raw Data")
    
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    # Filter options for tables
    table_type = st.selectbox(
        "Select table view:",
        ["All Data", "Worst Conversations", "Best Conversations", "Top Topics"],
        key="table_type_selector"
    )
    
    if table_type == "All Data":
        st.subheader("All Conversations")
        # Remove embedding column for display (too large)
        display_df = df.drop(columns=['embedding'], errors='ignore')
        st.dataframe(display_df, use_container_width=True, height=400, key="table_all_data")
    
    elif table_type == "Worst Conversations":
        st.subheader("Conversations with Lowest Sentiment/Happiness")
        worst_df = df.nsmallest(20, 'sentiment_score', keep='all') if 'sentiment_score' in df.columns else df.head(20)
        display_df = worst_df[['conversation_id', 'timestamp', 'product_type', 'user_question', 'sentiment_score', 'user_happiness']].drop(columns=['embedding'], errors='ignore')
        st.dataframe(display_df, use_container_width=True, key="table_worst")
    
    elif table_type == "Best Conversations":
        st.subheader("Conversations with Highest Sentiment/Happiness")
        best_df = df.nlargest(20, 'sentiment_score', keep='all') if 'sentiment_score' in df.columns else df.head(20)
        display_df = best_df[['conversation_id', 'timestamp', 'product_type', 'user_question', 'sentiment_score', 'user_happiness']].drop(columns=['embedding'], errors='ignore')
        st.dataframe(display_df, use_container_width=True, key="table_best")
    
    elif table_type == "Top Topics":
        if 'topic_label' in df.columns:
            st.subheader("Topic Summary")
            topic_summary = df.groupby('topic_label').agg({
                'user_question': 'count',
                'sentiment_score': 'mean',
                'happiness_score': 'mean'
            }).round(2)
            topic_summary.columns = ['Count', 'Avg Sentiment', 'Avg Happiness']
            topic_summary = topic_summary.sort_values('Count', ascending=False)
            st.dataframe(topic_summary, use_container_width=True, key="table_topics")
        else:
            st.info("Topics not computed yet.")
    
    # Export functionality
    st.divider()
    st.subheader("Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Remove embedding column for export (too large for CSV)
        export_df = df.drop(columns=['embedding'], errors='ignore')
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="chatbot_analysis.csv",
            mime="text/csv",
            key="download_csv"
        )
    
    with col2:
        # Excel export
        try:
            output = io.BytesIO()
            export_df = df.drop(columns=['embedding'], errors='ignore')
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Data')
            excel_data = output.getvalue()
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name="chatbot_analysis.xlsx",
                mime="application/vnd.openpyxl-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )
        except Exception as e:
            st.error(f"Error creating Excel file: {str(e)}")
    
    with col3:
        # Email report button
        st.markdown("**Email Report**")
        recipient_email = st.text_input(
            "Recipient Email",
            key="email_recipient_input",
            placeholder="email@example.com",
            help="Enter email address to send the report to"
        )
        email_sent = st.button(
            label="📧 Send Email Report",
            key="email_report_button",
            help="Send analysis findings via email (API configuration required)",
            use_container_width=True,
            disabled=not recipient_email or not EMAIL_API_AVAILABLE
        )
        if email_sent and recipient_email:
            if EMAIL_API_AVAILABLE:
                success, message = send_email_report(df, recipient_email, EMAIL_API_CONFIG)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("📧 Email API not configured yet. Please provide email API details to enable this feature.")
                # Show preview of report
                with st.expander("Preview Report"):
                    report_text = generate_email_report(df)
                    st.text(report_text)

def build_diagnostics_tab(df):
    """Build the Diagnostics / Data Quality tab."""
    st.header("🔍 Diagnostics & Data Quality")
    
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    # Data quality metrics
    st.subheader("Data Quality Metrics")
    
    quality_metrics = {
        "Total Rows": len(df),
        "Rows with Timestamp": df['timestamp'].notna().sum() if 'timestamp' in df.columns else 0,
        "Rows with User Question": df['user_question'].notna().sum() if 'user_question' in df.columns else 0,
        "Rows with Bot Answer": df['bot_answer'].notna().sum() if 'bot_answer' in df.columns else 0,
        "Rows with Sentiment Label": df['sentiment_label'].notna().sum() if 'sentiment_label' in df.columns else 0,
        "Rows with Sentiment Score": df['sentiment_score'].notna().sum() if 'sentiment_score' in df.columns else 0,
        "Rows with Happiness Label": df['user_happiness'].notna().sum() if 'user_happiness' in df.columns else 0,
        "Rows with Happiness Score": df['happiness_score'].notna().sum() if 'happiness_score' in df.columns else 0,
        "Rows with CSAT Score": df['csat_score'].notna().sum() if 'csat_score' in df.columns else 0,
        "Unique Product Types": df['product_type'].nunique() if 'product_type' in df.columns else 0,
        "Unique Channels": df['channel'].nunique() if 'channel' in df.columns else 0,
    }
    
    quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Metric', 'Value'])
    st.dataframe(quality_df, use_container_width=True, key="quality_metrics_table")
    
    # Missing data visualization
    st.subheader("Missing Data Pattern")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column",
            labels={'x': 'Column', 'y': 'Missing Count'}
        )
        st.plotly_chart(fig, use_container_width=True, key="missing_data_chart")
    else:
        st.success("No missing data found!")
    
    # Column information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.notna().sum(),
        'Null Count': df.isna().sum()
    })
    st.dataframe(col_info, use_container_width=True, key="column_info_table")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app."""
    # Enpal branding header with black background and high contrast
    # Try to load Enpal logo if available
    import os
    import base64
    logo_html = ''
    logo_path = None
    for ext in ['.png', '.svg', '.jpg', '.webp']:
        for filename in ['enpal_logo', 'enpal-logo', 'logo']:
            path = filename + ext
            if os.path.exists(path):
                logo_path = path
                break
        if logo_path:
            break
    
    if logo_path:
        # Read and encode logo as base64
        with open(logo_path, 'rb') as f:
            logo_data = f.read()
            logo_base64 = base64.b64encode(logo_data).decode()
            file_ext = os.path.splitext(logo_path)[1].lower()
            mime_type = 'image/png' if file_ext == '.png' else ('image/svg+xml' if file_ext == '.svg' else 'image/jpeg')
            logo_html = f'<img src="data:{mime_type};base64,{logo_base64}" style="max-height: 80px; width: auto;" alt="Enpal Logo">'
    else:
        # Enpal logo styled text - matches brand: bold sans-serif, white text with orange dot
        logo_html = '<div style="display: flex; align-items: center; gap: 8px;"><span style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', \'Roboto\', \'Helvetica Neue\', Arial, sans-serif; font-size: 48px; font-weight: 700; color: #ffffff; letter-spacing: 0px;">Enpal</span><span style="display: inline-block; width: 12px; height: 12px; background-color: #FF6B35; border-radius: 50%; margin-left: 2px;"></span></div>'
    
    header_html = f"""
    <div style='background-color: #000000; padding: 35px 30px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 10px 20px rgba(0,0,0,0.25); border: 2px solid #1a1a1a;'>
        <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 30px;'>
            <div style='display: flex; align-items: center; gap: 30px;'>
                {logo_html}
            </div>
            <div style='flex: 1; text-align: center; min-width: 400px;'>
                <h1 style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif; color: #ffffff; margin: 0; font-size: 36px; font-weight: 700; letter-spacing: 0.5px;'>💬 Chatbot Conversation Analyzer</h1>
                <p style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif; color: #ffffff; margin: 12px 0 0 0; font-size: 18px; font-weight: 400; opacity: 0.9;'>Analyze internal chatbot conversations about photovoltaic systems and heat pumps</p>
            </div>
        </div>
    </div>
    """
    
    css_style = """
    <style>
        /* Main app background - premium black */
        .stApp {
            background-color: #0a0a0a !important;
            color: #ffffff;
        }
        
        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            background-color: #0a0a0a;
            color: #ffffff;
        }
        
        /* All text elements - white for dark background */
        .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, span, div {
            color: #ffffff !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-weight: 600;
        }
        
        /* Tabs - highly visible with dark theme */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1a1a1a;
            border-radius: 8px 8px 0 0;
            padding: 8px 8px 0 8px;
            gap: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #2a2a2a;
            color: #ffffff !important;
            border-radius: 8px 8px 0 0;
            padding: 12px 24px;
            font-weight: 500;
            border: 1px solid #3a3a3a;
            margin-right: 4px;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #3a3a3a;
            color: #ffffff !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0a0a0a !important;
            color: #ffffff !important;
            border-bottom: 3px solid #FF6B35;
            font-weight: 600;
        }
        
        /* Tab content area */
        .stTabs [data-baseweb="tab-panel"] {
            background-color: #0a0a0a;
            padding: 20px;
        }
        
        /* Sidebar - dark theme */
        .css-1d391kg {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] {
            background-color: #1a1a1a !important;
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Input fields */
        .stTextInput > div > div > input, 
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            background-color: #2a2a2a !important;
            color: #ffffff !important;
            border: 1px solid #3a3a3a;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #FF6B35 !important;
            color: #ffffff !important;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            padding: 0.5rem 1.5rem;
        }
        
        .stButton > button:hover {
            background-color: #ff7b4a !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #cccccc !important;
        }
        
        /* Dataframes */
        .stDataFrame, .dataframe {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
        }
        
        /* Dividers */
        hr {
            border-color: #3a3a3a !important;
        }
        
        /* Info/Warning/Success boxes */
        .stInfo, .stSuccess, .stWarning, .stError {
            background-color: #1a1a1a !important;
            border-left: 4px solid #FF6B35;
            color: #ffffff !important;
        }
        
        /* Checkboxes */
        .stCheckbox label {
            color: #ffffff !important;
        }
        
        /* Select boxes */
        .stSelectbox label, .stMultiSelect label {
            color: #ffffff !important;
        }
        
        /* Slider */
        .stSlider label {
            color: #ffffff !important;
        }
        
        /* File uploader */
        .stFileUploader {
            background-color: #1a1a1a !important;
            border: 2px dashed #3a3a3a !important;
        }
        
        .stFileUploader label {
            color: #ffffff !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #2a2a2a !important;
            color: #ffffff !important;
        }
        
        /* Radio buttons */
        .stRadio label {
            color: #ffffff !important;
        }
        
        /* Use Enpal brand font throughout */
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif !important;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #3a3a3a;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #4a4a4a;
        }
    </style>
    """
    
    st.markdown(header_html + css_style, unsafe_allow_html=True)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔑 OpenAI API Key")
        api_key_input = st.text_input(
            "Enter OpenAI API Key",
            type="password",
            value=st.session_state.get('openai_api_key', ''),
            help="Enter your OpenAI API key. You can also set it in .streamlit/secrets.toml as OPENAI_API_KEY",
            key="api_key_input"
        )
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
            st.success("✓ API key set")
        elif 'openai_api_key' in st.session_state:
            # Clear if user removes the key
            del st.session_state.openai_api_key
        
        st.divider()
        
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file containing chatbot conversation data",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Load data
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.success(f"Loaded {len(df)} rows")
        
        st.divider()
        
        st.header("⚙️ Analysis Options")
        
        use_openai = st.checkbox(
            "Use OpenAI to compute sentiment & happiness if missing",
            value=True,
            help="Enable OpenAI API analysis for missing sentiment and happiness labels",
            key="use_openai_checkbox"
        )
        
        recompute_sentiment = st.checkbox(
            "Recompute sentiment even if columns exist",
            value=False,
            help="Force recomputation of sentiment analysis",
            key="recompute_sentiment_checkbox"
        )
        
        recompute_happiness = st.checkbox(
            "Recompute happiness even if columns exist",
            value=False,
            help="Force recomputation of happiness analysis",
            key="recompute_happiness_checkbox"
        )
        
        compute_topics_flag = st.checkbox(
            "Compute topics (clustering)",
            value=False,
            help="Perform topic analysis using embeddings and clustering",
            key="compute_topics_checkbox"
        )
        
        n_clusters = st.slider(
            "Number of topics (clusters)",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of topics to identify",
            key="n_clusters_slider"
        )
        
        st.divider()
        
        st.header("🔍 Filters")
        
        # Date range filter
        if st.session_state.df is not None and 'date' in st.session_state.df.columns:
            dates = st.session_state.df['date'].dropna().unique()
            if len(dates) > 0:
                min_date = min(dates)
                max_date = max(dates)
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range_filter"
                )
            else:
                date_range = None
        else:
            date_range = None
        
        # Product type filter
        if st.session_state.df is not None and 'product_type' in st.session_state.df.columns:
            product_types = st.session_state.df['product_type'].unique().tolist()
            selected_products = st.multiselect(
                "Product Type",
                options=product_types,
                default=product_types,
                help="Filter by product type",
                key="product_type_filter"
            )
        else:
            selected_products = []
        
        # Channel filter
        if st.session_state.df is not None and 'channel' in st.session_state.df.columns:
            channels = st.session_state.df['channel'].unique().tolist()
            selected_channels = st.multiselect(
                "Channel",
                options=channels,
                default=channels,
                help="Filter by channel",
                key="channel_filter"
            )
        else:
            selected_channels = []
        
        # Sentiment filter
        if st.session_state.df is not None and 'sentiment_label' in st.session_state.df.columns:
            sentiments = st.session_state.df['sentiment_label'].dropna().unique().tolist()
            selected_sentiments = st.multiselect(
                "Sentiment",
                options=sentiments,
                default=sentiments,
                help="Filter by sentiment",
                key="sentiment_filter"
            )
        else:
            selected_sentiments = []
        
        # Happiness filter
        if st.session_state.df is not None and 'user_happiness' in st.session_state.df.columns:
            happiness_options = st.session_state.df['user_happiness'].dropna().unique().tolist()
            selected_happiness = st.multiselect(
                "Happiness",
                options=happiness_options,
                default=happiness_options,
                help="Filter by happiness",
                key="happiness_filter"
            )
        else:
            selected_happiness = []
        
        st.divider()
        
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload an Excel file with chatbot conversation data
        2. Configure analysis options in the sidebar
        3. Apply filters as needed
        4. Explore results in the tabs above
        """)
    
    # Main content area
    if st.session_state.df is not None:
        # Preprocess data
        df_processed = preprocess_data(st.session_state.df)
        
        if df_processed is not None:
            # Apply OpenAI analysis if requested
            if use_openai:
                df_processed = enrich_data_with_openai(
                    df_processed,
                    recompute_sentiment=recompute_sentiment,
                    recompute_happiness=recompute_happiness
                )
            
            # Compute topics if requested
            if compute_topics_flag:
                df_processed = compute_topics(df_processed, n_clusters=n_clusters)
            
            # Apply filters
            filters = {
                'date_range': date_range if date_range and len(date_range) == 2 else None,
                'product_types': selected_products,
                'channels': selected_channels,
                'sentiments': selected_sentiments,
                'happiness': selected_happiness
            }
            
            df_filtered = filter_data(df_processed, filters)
            
            # Store in session state
            st.session_state.df_processed = df_filtered
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Overview",
                "Sentiment & Happiness",
                "Topics & Segments",
                "Time Series",
                "Tables / Raw Data",
                "Diagnostics / Data Quality"
            ])
            
            with tab1:
                build_overview_tab(df_filtered)
            
            with tab2:
                build_sentiment_tab(df_filtered)
            
            with tab3:
                build_topics_tab(df_filtered)
            
            with tab4:
                build_time_tab(df_filtered)
            
            with tab5:
                build_tables_tab(df_filtered)
            
            with tab6:
                build_diagnostics_tab(df_filtered)
        else:
            st.error("Error preprocessing data. Please check your file format.")
    else:
        st.info("👈 Please upload an Excel file to get started.")
        
        # Show example data structure
        with st.expander("Expected Data Structure"):
            st.markdown("""
            Your Excel file should contain at least these columns:
            
            - **conversation_id**: Unique identifier for each conversation
            - **timestamp**: Date and time of the conversation
            - **user_question**: The question asked by the user
            - **bot_answer**: The chatbot's response
            
            Optional columns:
            - **channel**: Communication channel (e.g., "web", "mobile")
            - **product_type**: Type of product (e.g., "photovoltaic", "heat pump")
            - **sentiment_label**: Pre-existing sentiment label
            - **sentiment_score**: Pre-existing sentiment score
            - **user_happiness**: Pre-existing happiness label
            - **happiness_score**: Pre-existing happiness score
            - **csat_score**: Customer satisfaction score
            """)

if __name__ == "__main__":
    main()

