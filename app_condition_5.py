"""
Entry point for Condition 5: E_shap_A_low
Explanation: feature_importance | Anthropomorphism: low
Feature importance (SHAP) with minimal/technical interface
"""
import os
import sys

# Force Condition 5: feature_importance explanation + low anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'feature_importance'
os.environ['HICXAI_ANTHRO'] = 'low'

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())
