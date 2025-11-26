"""
Entry point for Condition 4: E_cf_A_high
Explanation: counterfactual | Anthropomorphism: high
Counterfactual explanations (DiCE) with Luna's friendly interface
"""
import os
import sys

# Force Condition 4: counterfactual explanation + high anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'counterfactual'
os.environ['HICXAI_ANTHRO'] = 'high'

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())
