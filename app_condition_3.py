"""Entry point for Condition 3: E_shap_A_low
Explanation: shap (feature importance) | Anthropomorphism: low"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'shap'
os.environ['HICXAI_ANTHRO'] = 'low'
sys.path.append('src')
exec(open('src/app.py').read())
