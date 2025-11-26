"""
Entry point for Condition 6: E_shap_A_high
Explanation: feature_importance | Anthropomorphism: high
Luna with SHAP visualizations (original v1)
"""
import os
import sys

# Force Condition 6: Feature importance + High anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'feature_importance'
os.environ['HICXAI_ANTHRO'] = 'high'
# Legacy compatibility
os.environ['HICXAI_VERSION'] = 'v1'

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())