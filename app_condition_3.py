"""
Entry point for Condition 3: E_cf_A_low
Explanation: counterfactual | Anthropomorphism: low
Counterfactual explanations (DiCE) with minimal/technical interface
"""
import os
import sys
import time
import streamlit as st

# Force Condition 3: counterfactual explanation + low anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'counterfactual'
os.environ['HICXAI_ANTHRO'] = 'low'

# ===== QUALTRICS/PROLIFIC INTEGRATION =====
qs = st.query_params
pid = qs.get("pid", "")
cond = qs.get("cond", "")
return_url = qs.get("return", "")

def back_to_survey():
    if not return_url:
        st.warning("Return link missing. Please use your browser Back button to return to the survey.")
        return
    final = f"{return_url}&pid={pid}&cond={cond}&done=1"
    st.components.v1.html(f'<script>window.location.replace("{final}");</script>', height=0)

if "back_to_survey" not in st.session_state:
    st.session_state.back_to_survey = back_to_survey
    st.session_state.has_return_url = bool(return_url)

if "deadline" not in st.session_state:
    st.session_state.deadline = time.time() + 180

if time.time() >= st.session_state.deadline:
    st.info("⏰ Time limit reached. Returning you to the survey...")
    back_to_survey()
    st.stop()
# ===== END QUALTRICS INTEGRATION =====

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())
