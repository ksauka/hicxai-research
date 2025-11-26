"""
Entry point for Condition 1: E_none_A_low
Explanation: none | Anthropomorphism: low
Control baseline - no explanations, technical interface
"""
import os
import sys

# Force Condition 1: No explanations + Low anthropomorphism
os.environ['HICXAI_EXPLANATION'] = 'none'
os.environ['HICXAI_ANTHRO'] = 'low'
# Legacy compatibility
os.environ['HICXAI_VERSION'] = 'v0'

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())