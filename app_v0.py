"""
Entry point for Condition 1: E_none_A_low
Explanation: none | Anthropomorphism: low
Control baseline - no explanations, technical interface
"""
import os
import sys
import time
import streamlit as st

# Force Condition 1: No explanations + Low anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'none'
os.environ['HICXAI_ANTHRO'] = 'low'
# Legacy compatibility
os.environ['HICXAI_VERSION'] = 'v0'

# ===== QUALTRICS/PROLIFIC INTEGRATION (final, loop-proof) =====
from urllib.parse import unquote, urlparse, parse_qsl, urlencode, urlunparse

def _get_query_params():
    try:
        return dict(st.query_params)
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

def _as_str(v):
    return v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")

qs = _get_query_params()
pid = _as_str(qs.get("pid", ""))
cond = _as_str(qs.get("cond", ""))
return_raw = _as_str(qs.get("return", ""))

# Persist once per session
if "pid" not in st.session_state and pid:
    st.session_state.pid = pid
if "cond" not in st.session_state and cond:
    st.session_state.cond = cond
if "return_raw" not in st.session_state and return_raw:
    st.session_state.return_raw = return_raw
if "has_return_url" not in st.session_state:
    st.session_state.has_return_url = bool(return_raw)
if "_returned" not in st.session_state:
    st.session_state._returned = False

def build_return_url(done_flag=True):
    """Decode Qualtrics URL, append pid/cond/done safely, and rebuild."""
    if not st.session_state.get("return_raw"):
        return None
    decoded = unquote(st.session_state["return_raw"])
    p = urlparse(decoded)
    q = dict(parse_qsl(p.query))
    q.update({
        "pid": st.session_state.get("pid", ""),
        "cond": st.session_state.get("cond", ""),
        "done": "1" if done_flag else "0",
    })
    return urlunparse(p._replace(query=urlencode(q)))

def back_to_survey(done_flag=True):
    """Single redirect path. No auto-call on load."""
    if st.session_state._returned:
        return
    final = build_return_url(done_flag=done_flag)
    if not final:
        st.warning("Return link missing. Please use your browser Back button to return to the survey.")
        return
    st.session_state._returned = True
    st.components.v1.html(f'<script>window.location.replace("{final}");</script>', height=0)

st.session_state.back_to_survey = back_to_survey

# 3-minute cap: set once and only trigger after expiry
LIMIT_SECS = 180
if "deadline_ts" not in st.session_state:
    st.session_state.deadline_ts = time.time() + LIMIT_SECS

if time.time() >= st.session_state.deadline_ts and st.session_state.has_return_url:
    st.info("⏰ Time limit reached. Returning you to the survey…")
    back_to_survey(done_flag=True)
    st.stop()

# Optional status line (no auto JS redirect)
remaining = max(0, int(st.session_state.deadline_ts - time.time()))
mins, secs = divmod(remaining, 60)
if st.session_state.has_return_url:
    st.caption(f"⏱️ Up to {mins}:{secs:02d} remaining. You can continue manually when finished.")
# ===== END INTEGRATION =====

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())