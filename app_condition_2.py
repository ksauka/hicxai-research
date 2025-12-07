"""Entry point for Condition 2: E_none_A_high
Explanation: none | Anthropomorphism: high"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'none'
os.environ['HICXAI_ANTHRO'] = 'high'
sys.path.append('src')
exec(open('src/app.py').read())
