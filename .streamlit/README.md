# Streamlit Configuration

This directory contains configuration files for Streamlit deployment.

## Files

### `secrets.toml.template`
Template for secrets configuration. **DO NOT commit `secrets.toml` with real keys!**

For local development:
```bash
cp secrets.toml.template secrets.toml
# Edit secrets.toml with your real API keys
```

### For Streamlit Cloud Deployment

1. **Login to Streamlit Cloud**: https://share.streamlit.io
2. **Deploy your app** (if not already deployed)
3. **Add Secrets**:
   - Go to your app dashboard
   - Click the three dots (⋮) menu → **Settings** → **Secrets**
   - Paste this configuration:

```toml
OPENAI_API_KEY = "sk-proj-YOUR-REAL-KEY-HERE"
OPENAI_MODEL = "gpt-4o-mini"
HICXAI_GENAI = "on"
HICXAI_OPENAI_MODEL = "gpt-4o-mini"
HICXAI_TEMPERATURE = "0.7"
HICXAI_MAX_TOKENS = "100"
GITHUB_TOKEN = "ghp_YOUR-GITHUB-TOKEN"
GITHUB_REPO = "https://github.com/yourusername/hicxai-data-private.git"
HICXAI_VERSION = "v0"
HICXAI_DEBUG_MODE = "false"
```

4. **Click "Save"** - Your app will automatically restart

## Important Notes

- **`secrets.toml`** is in `.gitignore` - it will never be committed
- Streamlit Cloud reads secrets from the dashboard UI, not from the repo
- The app code accesses secrets via `st.secrets["KEY_NAME"]`
- Our `env_loader.py` already handles `st.secrets` fallback automatically

## Testing Locally

To test with secrets locally:
```bash
# Create secrets file
cp .streamlit/secrets.toml.template .streamlit/secrets.toml

# Edit with real keys
nano .streamlit/secrets.toml

# Run app
streamlit run app_v1.py
```

## Verification

Once deployed, test the LLM validation:
1. Fill out the loan application
2. Enter invalid data (e.g., "gg" for age)
3. You should see a warm, friendly LLM-generated message like:
   - "Oh no, it looks like that input didn't quite fit! 😊"
   - "I see that might have been a little mix-up!"

If you see hardcoded messages instead, check:
- Secrets are correctly added in Streamlit Cloud dashboard
- `HICXAI_GENAI = "on"` is set
- `OPENAI_API_KEY` is valid and has billing enabled
