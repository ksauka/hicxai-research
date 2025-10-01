"""
Entry point for V0 (Control Group) deployment
Forces the app to run in V0 mode regardless of environment variables
"""
import os
import sys

# Force V0 mode
os.environ['HICXAI_VERSION'] = 'v0'

# Add src to path and run main app
sys.path.append('src')
exec(open('src/app.py').read())