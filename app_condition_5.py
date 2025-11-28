"""
Entry point for Condition 5: E_shap_A_low
Explanation: feature_importance | Anthropomorphism: low
Feature importance (SHAP) with minimal/technical interface
"""
import os
import sys
import time
import streamlit as st

# Force Condition 5: feature_importance explanation + low anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'feature_importance'
os.environ['HICXAI_ANTHRO'] = 'low'

# ===== QUALTRICS/PROLIFIC INTEGRATION (safe redirect) =====
from urllib.parse import unquote, urlparse, parse_qsl, urlencode, urlunparse

# Get query params
try:
    qs = dict(st.query_params)
except Exception:
    try:
        qs = st.experimental_get_query_params()
        # Convert lists to strings
        qs = {k: v[0] if isinstance(v, list) and v else v for k, v in qs.items()}
    except Exception:
        qs = {}

pid = qs.get("pid", "")
cond = qs.get("cond", "")
return_raw = qs.get("return", "")  # ENCODED Qualtrics URL

def back_to_survey():
    """Safely decode and append params to Qualtrics URL to prevent loops."""
    if not return_raw:
        st.warning("Return link missing. Please use your browser Back to return to the survey.")
        return
    
    # 1) Decode the encoded Qualtrics URL
    decoded = unquote(return_raw)
    
    # 2) Parse and append pid, cond, done=1 safely (no double-encoding)
    p = urlparse(decoded)
    q = dict(parse_qsl(p.query))
    q.update({"pid": pid, "cond": cond, "done": "1"})
    final = urlunparse(p._replace(query=urlencode(q)))
    
    # 3) Redirect exactly once
    st.components.v1.html(
        f'<script>window.location.replace("{final}");</script>',
        height=0
    )

# Store in session state for use elsewhere
if "has_return_url" not in st.session_state:
    st.session_state.has_return_url = bool(return_raw)
if "pid" not in st.session_state:
    st.session_state.pid = pid
if "cond" not in st.session_state:
    st.session_state.cond = cond
st.session_state.back_to_survey = back_to_survey

# 3-minute timer: set once
if "deadline" not in st.session_state:
    st.session_state.deadline = time.time() + 180
# ===== END QUALTRICS INTEGRATION =====

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())

# Timer check and countdown AFTER app loads
if st.session_state.get("has_return_url", False):
    if time.time() >= st.session_state.deadline:
        st.info("⏰ Time limit reached. Returning to the survey…")
        back_to_survey()
        st.stop()
    else:
        # Show countdown and auto-rerun every 5 seconds
        remaining = int(max(0, st.session_state.deadline - time.time()))
        mins, secs = divmod(remaining, 60)
        st.sidebar.caption(f"⏱️ Auto-return in ~{mins}:{secs:02d}")
        time.sleep(5)
        st.rerun()
