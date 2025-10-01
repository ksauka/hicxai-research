"""
Entry point for V1 (Treatment Group) deployment
Forces the app to run in V1 mode regardless of environment variables
"""
import os
import sys

# Force V1 mode
os.environ['HICXAI_VERSION'] = 'v1'

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())