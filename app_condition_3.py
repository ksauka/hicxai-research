"""
Entry point for Condition 3: E_cf_A_low
Explanation: counterfactual | Anthropomorphism: low
Counterfactual explanations (DiCE) with minimal/technical interface
"""
import os
import sys

# Force Condition 3: counterfactual explanation + low anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'counterfactual'
os.environ['HICXAI_ANTHRO'] = 'low'

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())
