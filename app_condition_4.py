"""Entry point for Condition 4: E_cf_A_low
Explanation: cf (counterfactual) | Anthropomorphism: low"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'cf'
os.environ['HICXAI_ANTHRO'] = 'low'
sys.path.append('src')
exec(open('src/app.py').read())
