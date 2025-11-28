"""Entry point for Condition 1: E_none_A_low
Explanation: none | Anthropomorphism: low"""
import os, sys, streamlit as st
os.environ['HICXAI_EXPLANATION'] = 'none'
os.environ['HICXAI_ANTHRO'] = 'low'
sys.path.append('src')
exec(open('src/app.py').read())
