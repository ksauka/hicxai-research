"""Entry point for Condition 6: E_shap_A_high
Explanation: feature_importance | Anthropomorphism: high"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'feature_importance'
os.environ['HICXAI_ANTHRO'] = 'high'
sys.path.append('src')
exec(open('src/app.py').read())
