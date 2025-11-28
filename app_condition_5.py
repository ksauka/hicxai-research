"""Entry point for Condition 5: E_shap_A_high
Explanation: shap (feature importance) | Anthropomorphism: high"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'shap'
os.environ['HICXAI_ANTHRO'] = 'high'
sys.path.append('src')
exec(open('src/app.py').read())
