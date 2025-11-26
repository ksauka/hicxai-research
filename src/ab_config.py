"""
A/B Testing Configuration for HicXAI Agent
This module configures experimental conditions for the live study.

Experiment factors (3 × 2):
- Explanation type: none | counterfactual | feature_importance
- Anthropomorphism: low | high

Backwards compatibility:
- HICXAI_VERSION = v0 | v1 still works
  v0 -> explanation=none, anthropomorphism=low
  v1 -> explanation=feature_importance, anthropomorphism=high

Environment variables (preferred) or CLI flags:
- HICXAI_EXPLANATION = none | counterfactual | feature_importance
- HICXAI_ANTHRO      = low | high
- HICXAI_VERSION     = v0 | v1  (legacy)
CLI flags:
  --explanation=none|counterfactual|feature_importance
  --anthro=low|high
  --HICXAI_VERSION=v0|v1 or --v0 / --v1 or --ab=v0|v1
"""

import os
import sys
import uuid
import time
import streamlit as st

_VALID_EXPLANATIONS = {"none", "counterfactual", "feature_importance"}
_VALID_ANTHRO       = {"low", "high"}


class AppConfig:
    """Configuration class for A/B testing versions and factor levels."""

    def __init__(self):
        # read factor levels (env and CLI), then derive UI toggles
        self.explanation = self._get_explanation_level()           # none | counterfactual | feature_importance
        self.anthro      = self._get_anthropomorphism_level()      # low | high
        self.version     = self._legacy_version_label()            # v0 | v1 (for sidebar display only)
        self.session_id  = self._generate_session_id()             # unique session tracking

        # derived feature flags for UI rendering, explanations, and logging
        self.show_anthropomorphic   = (self.anthro == "high")
        self.show_profile_pic       = self.show_anthropomorphic
        self.show_shap_visualizations = (self.explanation == "feature_importance")
        self.show_counterfactual    = (self.explanation == "counterfactual")
        self.show_any_explanation   = (self.explanation != "none")

        # assistant identity and copy are derived from anthropomorphism
        self.assistant_name = "Luna" if self.show_anthropomorphic else "AI Assistant"
        if self.show_anthropomorphic:
            self.assistant_intro = "Your AI loan assistant, I will guide you step by step and explain what matters for your decision."
        else:
            self.assistant_intro = "AI system for loan decision support, explanations are provided according to your selection."

        # data collection options
        self.collect_feedback = True
        self.show_debug_info  = False  # keep False in production
        
        # Legacy compatibility
        self.use_full_features = self.show_any_explanation

    # ------------- parsing helpers -------------

    def _get_explanation_level(self):
        """Resolve explanation factor from env or CLI, with legacy fallback."""
        # env first
        env_val = os.getenv("HICXAI_EXPLANATION", "").strip().lower()
        if env_val in _VALID_EXPLANATIONS:
            return env_val

        # CLI flags
        for arg in sys.argv[1:]:
            if arg.startswith("--explanation="):
                cand = arg.split("=", 1)[1].strip().lower()
                if cand in _VALID_EXPLANATIONS:
                    return cand

        # legacy version mapping
        legacy = os.getenv("HICXAI_VERSION", "").strip().lower()
        cli_ver = self._cli_version_flag()
        legacy = cli_ver or legacy
        if legacy == "v1":
            return "feature_importance"
        if legacy == "v0":
            return "none"

        # default
        return "none"

    def _get_anthropomorphism_level(self):
        """Resolve anthropomorphism factor from env or CLI, with legacy fallback."""
        # env first
        env_val = os.getenv("HICXAI_ANTHRO", "").strip().lower()
        if env_val in _VALID_ANTHRO:
            return env_val

        # CLI flags
        for arg in sys.argv[1:]:
            if arg.startswith("--anthro="):
                cand = arg.split("=", 1)[1].strip().lower()
                if cand in _VALID_ANTHRO:
                    return cand

        # legacy version mapping
        legacy = os.getenv("HICXAI_VERSION", "").strip().lower()
        cli_ver = self._cli_version_flag()
        legacy = cli_ver or legacy
        if legacy == "v1":
            return "high"
        if legacy == "v0":
            return "low"

        # default
        return "low"

    def _cli_version_flag(self):
        """Read legacy version flags from CLI to support existing scripts."""
        for arg in sys.argv[1:]:
            if arg in ("--v0", "--v1"):
                return arg[2:]
            if arg.startswith("--HICXAI_VERSION="):
                cand = arg.split("=", 1)[1].strip().lower()
                if cand in {"v0", "v1"}:
                    return cand
            if arg.startswith("--ab="):
                cand = arg.split("=", 1)[1].strip().lower()
                if cand in {"v0", "v1"}:
                    return cand
        return ""

    def _legacy_version_label(self):
        """Provide a simple label for the sidebar, does not affect factor levels."""
        # map current factors to a human friendly tag
        if self.explanation == "feature_importance" and self.anthro == "high":
            return "v1"
        if self.explanation == "none" and self.anthro == "low":
            return "v0"
        return "custom"

    def _generate_session_id(self):
        """Generate unique session ID for concurrent user tracking."""
        return f"{self.condition_code()}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

    # ------------- public helpers for UI and logging -------------

    def condition_code(self):
        """
        Compact code for logging and analysis.
        Examples: E_none_A_low, E_cf_A_high, E_shap_A_high
        """
        e = {"none": "none", "counterfactual": "cf", "feature_importance": "shap"}[self.explanation]
        a = {"low": "low", "high": "high"}[self.anthro]
        return f"E_{e}_A_{a}"

    def get_assistant_avatar(self):
        """Return avatar path for high anthropomorphism, else None."""
        if not self.show_profile_pic:
            return None
        possible_paths = [
            "assets/luna_avatar.png",
            "images/assistant_avatar.png",
            "data_questions/Luna_is_a_Dutch_customer_service_assistant_working_at_a_restaurant_she_is_27_years_old_Please_genera.png",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None  # UI can fall back to initials

    def get_welcome_message(self):
        """Version specific welcome message for the chat header."""
        if self.show_anthropomorphic:
            return f"Hi, I am {self.assistant_name}. I will review your information and explain what factors influenced this loan decision."
        return "Welcome, this AI credit assistant can review your information and show which factors influenced the decision."

    def should_show_visual_explanations(self):
        """Whether to render SHAP bars or equivalent visuals."""
        return self.show_shap_visualizations

    def should_show_counterfactuals(self):
        """Whether to render counterfactual suggestions."""
        return self.show_counterfactual

    def explanation_style(self):
        """Control tone for natural language explanations."""
        return "conversational" if self.show_anthropomorphic else "technical"

    def explanation_label(self):
        """Human readable label for the assigned explanation type."""
        if self.explanation == "none":
            return "No explanation"
        if self.explanation == "counterfactual":
            return "Counterfactual explanation"
        return "Feature importance explanation"
    
    # Legacy compatibility methods
    def get_explanation_style(self):
        """Get explanation style based on version (alias for explanation_style)"""
        return self.explanation_style()


# ------------- sidebar debug -------------

def show_debug_sidebar():
    """Display condition and toggles for quick inspection."""
    st.sidebar.write("### Experiment condition")
    st.sidebar.write(f"Version tag: **{config.version}**")
    st.sidebar.write(f"Condition: **{config.condition_code()}**")
    st.sidebar.write(f"Assistant: **{config.assistant_name}**")
    st.sidebar.write(f"Anthropomorphism: **{config.anthro}**")
    st.sidebar.write(f"Explanation: **{config.explanation}**")
    st.sidebar.write(f"Visual SHAP: {'✅' if config.show_shap_visualizations else '❌'}")
    st.sidebar.write(f"Counterfactual: {'✅' if config.show_counterfactual else '❌'}")
    st.sidebar.caption(f"Session ID: {config.session_id}")


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