"""Entry point for Condition 1: E_cf_A_high
Explanation: cf (counterfactual) | Anthropomorphism: high"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'cf'
os.environ['HICXAI_ANTHRO'] = 'high'
sys.path.append('src')
exec(open('src/app.py').read())
