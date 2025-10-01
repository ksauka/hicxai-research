"""
A/B Testing Configuration for HicXAI Agent
Environment variables to control different app versions
"""

import os
import streamlit as st

class AppConfig:
    """Configuration class for A/B testing versions"""
    
    def __init__(self):
        # Get version from command line args or environment variable
        self.version = self._get_version()
        self.session_id = self._generate_session_id()  # Unique session tracking
        
        # Version-specific configurations
        if self.version == "v1":
            self.use_full_features = True
            self.show_anthropomorphic = True
            self.show_profile_pic = True
            self.show_shap_visualizations = True
            self.assistant_name = "Luna"
            self.assistant_intro = "Your AI Loan Assistant - I'll guide you step-by-step through your loan application"
            self.collect_feedback = True
            self.show_debug_info = False  # Hidden from users
        else:  # v0 (minimal version)
            self.use_full_features = False
            self.show_anthropomorphic = False
            self.show_profile_pic = False
            self.show_shap_visualizations = False
            self.assistant_name = "AI Assistant"
            self.assistant_intro = "AI system for loan application processing"
            self.collect_feedback = True
            self.show_debug_info = False  # Hidden from users
    
    def _get_version(self):
        """Get version from command line or environment"""
        # Check command line arguments
        import sys
        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                if arg.startswith('--v'):
                    return arg[2:]  # Remove '--' prefix
        
        # Check environment variable
        version = os.getenv('HICXAI_VERSION', 'v0')
        return version
    
    def _generate_session_id(self):
        """Generate unique session ID for concurrent user tracking"""
        import uuid
        import time
        return f"{self.version}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    def get_assistant_avatar(self):
        """Get avatar file path based on version"""
        if self.show_profile_pic:
            # Try to find Luna's avatar file
            possible_paths = [
                "data_questions/Luna_is_a_Dutch_customer_service_assistant_working_at_a_restaurant_she_is_27_years_old_Please_genera.png",
                "assets/luna_avatar.png",
                "images/assistant_avatar.png"
            ]
            for path in possible_paths:
                import os
                if os.path.exists(path):
                    return path
            return None  # Will use initials fallback
        else:
            return None  # Will use initials fallback
    
    def get_welcome_message(self):
        """Get version-specific welcome message"""
        if self.show_anthropomorphic:
            return f"Hi! I'm {self.assistant_name}, your AI loan assistant. I'm here to help you understand your loan application decision and explore what factors matter most. Let's get started!"
        else:
            return f"Welcome to the {self.assistant_name}. I can help explain loan decisions and analyze important factors. Please provide your information to begin."
    
    def should_show_visual_explanations(self):
        """Whether to show SHAP visualizations"""
        return self.show_shap_visualizations
    
    def get_explanation_style(self):
        """Get explanation style based on version"""
        if self.show_anthropomorphic:
            return "conversational"  # Friendly, personal explanations
        else:
            return "technical"       # Direct, factual explanations

# Global config instance
config = AppConfig()

def show_debug_sidebar():
    """Display A/B testing debug info in sidebar"""
    if config.version == "v1":
        st.sidebar.success(f"🧪 A/B Test Version: **V1** (Full Features)")
    else:
        st.sidebar.info(f"🧪 A/B Test Version: **V0** (Minimal)")
    
    st.sidebar.write(f"**Assistant:** {config.assistant_name}")
    st.sidebar.write(f"**Visual SHAP:** {'✅' if config.show_shap_visualizations else '❌'}")
    st.sidebar.write(f"**Anthropomorphic:** {'✅' if config.show_anthropomorphic else '❌'}")