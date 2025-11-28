"""
Entry point for Condition 2: E_none_A_high
Explanation: none | Anthropomorphism: high
High anthropomorphism (Luna) but NO explanations
"""
import os
import sys
import time
import streamlit as st

# Force Condition 2: none explanation + high anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'none'
os.environ['HICXAI_ANTHRO'] = 'high'

# ===== QUALTRICS/PROLIFIC INTEGRATION =====
import urllib.parse

def _get_query_params():
    """Compat helper: Streamlit 1.31+ st.query_params; fallback to experimental API."""
    try:
        return dict(st.query_params)
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

def _as_str(v):
    if isinstance(v, list):
        return v[0] if v else ""
    return v if isinstance(v, str) else ""

# Read query params
_qs = _get_query_params()
pid = _as_str(_qs.get("pid", ""))
cond = _as_str(_qs.get("cond", ""))
return_url = _as_str(_qs.get("return", ""))
has_return = bool(return_url)

# Session wiring (one-time)
if "has_return_url" not in st.session_state:
    st.session_state.has_return_url = has_return
if "pid" not in st.session_state and pid:
    st.session_state.pid = pid
if "cond" not in st.session_state and cond:
    st.session_state.cond = cond
if "return_url" not in st.session_state and return_url:
    st.session_state.return_url = return_url
if "_returned" not in st.session_state:
    st.session_state._returned = False

def back_to_survey(done_flag=True):
    """Redirect participant back to Qualtrics with &pid=...&cond=...&done=1|0"""
    if st.session_state._returned:
        return
    ru = st.session_state.get("return_url") or return_url
    if not ru:
        st.warning("Return link missing. Please use your browser Back button to return to the survey.")
        return
    
    d = "1" if done_flag else "0"
    _pid = st.session_state.get("pid", pid)
    _cond = st.session_state.get("cond", cond)
    
    final = f"{ru}&pid={urllib.parse.quote_plus(_pid or '')}&cond={urllib.parse.quote_plus(_cond or '')}&done={d}"
    st.session_state._returned = True
    st.components.v1.html(f'<script>window.location.replace("{final}");</script>', height=0)

st.session_state.back_to_survey = back_to_survey

# 3-minute auto-return with countdown
AUTO_LIMIT_SECONDS = 180
if "deadline_ts" not in st.session_state:
    st.session_state.deadline_ts = time.time() + AUTO_LIMIT_SECONDS

remaining = int(max(0, st.session_state.deadline_ts - time.time()))
if st.session_state.has_return_url and remaining > 0:
    mins, secs = divmod(remaining, 60)
    st.caption(f"⏱️ You will be returned to the survey in ~{mins}:{secs:02d} unless you continue manually.")
    # Auto-rerun every second to update countdown and check expiration
    time.sleep(1)
    st.rerun()

if time.time() >= st.session_state.deadline_ts and st.session_state.has_return_url:
    st.info("⏰ Time limit reached. Returning you to the survey…")
    back_to_survey(done_flag=True)
    st.stop()
# ===== END QUALTRICS INTEGRATION =====

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())
