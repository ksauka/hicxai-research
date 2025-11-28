"""
Entry point for Condition 6: E_shap_A_high
Explanation: feature_importance | Anthropomorphism: high
Luna with SHAP visualizations (original v1)
"""
import os
import sys
import time
import streamlit as st

# Force Condition 6: Feature importance + High anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'feature_importance'
os.environ['HICXAI_ANTHRO'] = 'high'
# Legacy compatibility
os.environ['HICXAI_VERSION'] = 'v1'

# ===== QUALTRICS/PROLIFIC INTEGRATION =====
# Read query parameters from Qualtrics
qs = st.query_params
pid = qs.get("pid", "")
cond = qs.get("cond", "")
return_url = qs.get("return", "")  # URL-encoded Qualtrics return URL with ?stage=post

def back_to_survey():
    """Return participant to Qualtrics survey"""
    if not return_url:
        st.warning("Return link missing. Please use your browser Back button to return to the survey.")
        return
    final = f"{return_url}&pid={pid}&cond={cond}&done=1"
    st.components.v1.html(f'<script>window.location.replace("{final}");</script>', height=0)

# Store in session state for access from main app
if "back_to_survey" not in st.session_state:
    st.session_state.back_to_survey = back_to_survey
    st.session_state.has_return_url = bool(return_url)

# Initialize timer: 3-minute cap (180 seconds)
if "deadline" not in st.session_state:
    st.session_state.deadline = time.time() + 180

# Auto-return when time limit reached
if time.time() >= st.session_state.deadline:
    st.info("⏰ Time limit reached. Returning you to the survey...")
    back_to_survey()
    st.stop()
# ===== END QUALTRICS INTEGRATION =====

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())