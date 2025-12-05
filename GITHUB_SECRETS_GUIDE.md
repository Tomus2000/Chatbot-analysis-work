# GitHub Secrets Setup Guide

## How the OpenAI API is Called

The app uses the OpenAI Python client to make API calls. Here's how it works:

### 1. Client Initialization
```python
# In app.py, line 225-246
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
    
    return OpenAI(api_key=api_key)
```

### 2. Chat Completions API Call
```python
# Example from analyze_sentiment_openai (line 255-269)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a sentiment analysis expert..."
        },
        {
            "role": "user",
            "content": f"Text to analyze: {text[:2000]}"
        }
    ],
    response_format={"type": "json_object"},
    temperature=0.3
)
```

### 3. Embeddings API Call
```python
# Example from get_embeddings_openai (line 324-327)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=batch
)
```

## Setting Up GitHub Secrets

### Option 1: For Streamlit Cloud Deployment

1. **Go to your GitHub repository**
2. **Navigate to Settings → Secrets and variables → Actions** (or if deploying to Streamlit Cloud, use Streamlit Cloud's secrets)
3. **Click "New repository secret"**
4. **Add the secret:**
   - **Name:** `OPENAI_API_KEY`
   - **Value:** Your OpenAI API key (starts with `sk-...`)
5. **Click "Add secret"**

### Option 2: For GitHub Actions CI/CD

1. **Go to your repository on GitHub**
2. **Click Settings → Secrets and variables → Actions**
3. **Click "New repository secret"**
4. **Add:**
   - **Name:** `OPENAI_API_KEY`
   - **Value:** Your API key
5. **Save**

Then in your GitHub Actions workflow (`.github/workflows/your-workflow.yml`):
```yaml
- name: Run Streamlit App
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    # Your deployment commands here
```

### Option 3: For Streamlit Cloud (Recommended)

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Select your app**
3. **Click "Settings" → "Secrets"**
4. **Add:**
   ```toml
   OPENAI_API_KEY = "sk-proj-..."
   ```
5. **Save**

The app will automatically read from `st.secrets["OPENAI_API_KEY"]` when deployed.

## Local Development

For local development, create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-proj-your-key-here"
```

**⚠️ IMPORTANT:** Make sure `.streamlit/secrets.toml` is in your `.gitignore` file!

## Security Best Practices

1. ✅ **Never commit API keys to Git**
   - Already added to `.gitignore`
   
2. ✅ **Use environment variables or secrets**
   - GitHub Secrets for CI/CD
   - Streamlit Secrets for Streamlit Cloud
   - `.streamlit/secrets.toml` for local (gitignored)

3. ✅ **Rotate keys if exposed**
   - If you accidentally commit a key, rotate it immediately in OpenAI dashboard

4. ✅ **Use least privilege**
   - Consider creating a separate API key with limited permissions if possible

## Current Implementation

The app checks for the API key in this order:
1. **Session state** (from UI input) - highest priority
2. **Streamlit secrets** (`st.secrets["OPENAI_API_KEY"]`)
3. **Returns None** if not found (shows warning)

This allows flexibility for both local development and cloud deployment.

