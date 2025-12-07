"""Entry point for Condition 2: E_none_A_high
Explanation: none | Anthropomorphism: high"""
import os, sys, streamlit as st

# Hide Streamlit branding for anonymous review
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stHeader"]  {display: none !important;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stFooter"]  {display: none !important;}
div[role="contentinfo"]   {display: none !important;}
[data-testid="manage-app-button"] {display: none !important;}
.stAppDeployButton, .stDeployButton {display: none !important;}
.viewerBadge_link__Ua7HT,
.viewerBadge_container__2QSob,
a.viewer-badge,
a[href*="streamlit.io/cloud"] {display: none !important;}
section.main > div {padding-bottom: 0 !important;}
</style>
<meta name="robots" content="noindex, nofollow">
""", unsafe_allow_html=True)

os.environ['HICXAI_EXPLANATION'] = 'none'
os.environ['HICXAI_ANTHRO'] = 'high'
sys.path.append('src')
exec(open('src/app.py').read())
