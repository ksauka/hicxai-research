"""Entry point for Condition 4: E_cf_A_high
Explanation: counterfactual | Anthropomorphism: high"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'counterfactual'
os.environ['HICXAI_ANTHRO'] = 'high'
sys.path.append('src')
exec(open('src/app.py').read())
