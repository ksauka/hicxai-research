"""Entry point for Condition 3: E_cf_A_low
Explanation: counterfactual | Anthropomorphism: low"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'counterfactual'
os.environ['HICXAI_ANTHRO'] = 'low'
sys.path.append('src')
exec(open('src/app.py').read())
