"""Entry point for Condition 5: E_shap_A_low
Explanation: feature_importance | Anthropomorphism: low"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'feature_importance'
os.environ['HICXAI_ANTHRO'] = 'low'
sys.path.append('src')
exec(open('src/app.py').read())
