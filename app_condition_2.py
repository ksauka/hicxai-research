"""
Entry point for Condition 2: E_none_A_high
Explanation: none | Anthropomorphism: high
High anthropomorphism (Luna) but NO explanations
"""
import os
import sys

# Force Condition 2: none explanation + high anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'none'
os.environ['HICXAI_ANTHRO'] = 'high'

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())
