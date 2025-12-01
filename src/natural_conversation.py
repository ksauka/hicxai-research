"""
Natural conversation helpers: OpenAI GPT enhancement for explanations (gpt-4o-mini by default).

Behavior:
- If OPENAI_API_KEY is set (via env or Streamlit Secrets), use OpenAI to enhance explanations
- Style is determined by anthropomorphism level:
  - HIGH: Warm, conversational, actionable (Luna style)
  - LOW: Professional, technical, direct (AI Assistant style)
- Otherwise, return the original text unchanged

Notes:
- Keep outputs faithful: do not invent numbers or facts; preserve lists and key points
- This module is optional. LoanAssistant guards imports accordingly
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional
from pathlib import Path

# Try to import streamlit to fetch secrets when running on Streamlit Cloud
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    st = None  # type: ignore

# Ensure .env file is loaded (in case env_loader hasn't run yet)
def _ensure_env_loaded():
    """Load .env file if not already loaded"""
    # Try to load .env files (prefer .env.local over .env, like env_loader.py)
    try:
        root = Path(__file__).parent.parent
        env_files = [root / '.env.local', root / '.env']  # Check .env.local first
        
        for env_file in env_files:
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        
                        key, value = line.split('=', 1)
                        k = key.strip()
                        v = value.strip()
                        
                        # ALWAYS override OPENAI_API_KEY to ensure we have the latest from .env files
                        if k == "OPENAI_API_KEY" and v:
                            os.environ[k] = v
                        elif k not in os.environ:
                            os.environ[k] = v
    except Exception:
        pass


def _should_use_genai() -> bool:
    """LLM is REQUIRED for natural conversation - always returns True if API key available."""
    _ensure_env_loaded()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Allow pulling key from Streamlit Secrets when not present in env
    if not api_key and st is not None:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)  # type: ignore[attr-defined]
            if key:
                os.environ["OPENAI_API_KEY"] = str(key)
                api_key = str(key)
        except Exception:
            pass
    
    if not api_key:
        # Warn if missing - this is now required for quality conversation
        import warnings
        warnings.warn("OPENAI_API_KEY not found - conversation quality will be degraded")
    
    return bool(api_key)


def _get_openai_client():
    """Return an OpenAI client configured from environment/Streamlit secrets.

    Honors optional base URL (HICXAI_OPENAI_BASE_URL or OPENAI_BASE_URL) for proxies.
    """
    _ = _should_use_genai()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    base_url = (
        os.environ.get("HICXAI_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or None
    )
    
    try:
        from openai import OpenAI  # type: ignore
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _build_system_prompt(high_anthropomorphism: bool = True) -> str:
    """Build system prompt respecting anthropomorphism condition."""
    if high_anthropomorphism:
        # Luna: Warm, friendly, conversational, actionable
        return (
            "You are Luna, a friendly loan assistant having a real conversation with someone. "
            "Rewrite this explanation as if you're speaking naturally to a friend - warm, supportive, and genuinely human. "
            "Write like you're a real person explaining this, not a robot reading a script. Use natural, flowing language. "
            "Preserve ALL factual content, numbers, and data points exactly. "
            "Use 1-2 emojis naturally where they fit. Sound like a real human having a conversation. "
            "Structure as actionable insights when appropriate. "
            "Use clear formatting with bullets or short paragraphs. "
            "Never add meta-commentary - just speak naturally and directly. "
            "Do not fabricate data. Do not change any numeric values."
        )
    else:
        # AI Assistant: Professional, technical, direct
        return (
            "You are a professional AI loan advisor explaining this to a client. "
            "Rewrite this explanation in clear, professional language - direct and informative. "
            "Write like a knowledgeable professional communicating important information. "
            "Preserve ALL factual content, numbers, and data points exactly. "
            "Be direct, clear, and authoritative. No emojis. No casual language. "
            "DO NOT add letter formatting (no 'Dear', no 'Subject:', no salutations, no signatures). "
            "Just provide the core explanation directly without any letter structure. "
            "Use technical precision and structured formatting. "
            "Never add meta-commentary - just provide the professional explanation directly. "
            "Do not fabricate data. Do not change any numeric values."
        )


def _compose_messages(response: str, context: Optional[Dict[str, Any]], high_anthropomorphism: bool = True):
    sys_prompt = _build_system_prompt(high_anthropomorphism)
    ctx_lines = []
    if context:
        for k, v in context.items():
            if v is None:
                continue
            ctx_lines.append(f"- {k}: {v}")
    ctx_blob = "\n".join(ctx_lines) if ctx_lines else "(no extra context)"

    user_prompt = (
        "Rewrite the following explanation for the end user. Preserve all factual content and numbers.\n\n"
        f"Context:\n{ctx_blob}\n\n"
        f"Original Explanation:\n{response}\n\n"
        "Return only the rewritten explanation text."
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def enhance_validation_message(field: str, user_input: str, expected_format: str, attempt: int = 1, high_anthropomorphism: bool = True) -> Optional[str]:
    """Generate a validation message using LLM (REQUIRED for natural conversation).
    
    Args:
        field: The field name being validated
        user_input: The invalid input provided by user
        expected_format: Description of the expected format
        attempt: Which attempt this is (1, 2, 3+)
        high_anthropomorphism: If True, use warm/friendly Luna tone. If False, use professional AI Assistant tone.
    
    Returns None only if LLM fails - caller should have hardcoded fallback.
    """
    if not _should_use_genai():
        return None  # Will use fallback, but this should not happen in production
    
    try:
        client = _get_openai_client()
        if client is None:
            return None
        
        if high_anthropomorphism:
            system_prompt = (
                "You are Luna, a friendly and warm AI loan assistant. Generate a brief, empathetic validation message "
                "when a user enters invalid input. Be encouraging, use appropriate emojis (1-2), and guide them gently. "
                "Keep it to 1-2 sentences. Show understanding and warmth."
            )
        else:
            system_prompt = (
                "You are Luna, a professional AI loan assistant. Generate a clear, concise validation message "
                "when a user enters invalid input. Be direct and helpful. No emojis. "
                "Keep it to 1-2 sentences. Focus on what the user needs to provide."
            )
        
        user_prompt = (
            f"The user entered '{user_input}' for the field '{field.replace('_', ' ')}', but this is invalid. "
            f"Expected format: {expected_format}. "
            f"This is attempt #{attempt}. "
            f"Generate a friendly validation message that helps them correct their input."
        )
        
        model_name = os.getenv("HICXAI_OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("HICXAI_TEMPERATURE", "0.7"))  # Higher for more variety
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=400,
        )
        
        result = completion.choices[0].message.content if completion and completion.choices else None
        return result
    except Exception:
        return None


def enhance_response(response: str, context: Optional[Dict[str, Any]] = None, response_type: str = "explanation", high_anthropomorphism: bool = True) -> str:
    """Enhance response using OpenAI to respect anthropomorphism condition (REQUIRED for quality).

    Args:
        response: The original response text
        context: Optional context dictionary 
        response_type: Type of response (explanation, loan, etc)
        high_anthropomorphism: If True, use warm Luna style with actionable insights. 
                               If False, use professional AI Assistant style.

    If OpenAI is not configured, returns the original response (degraded quality).
    """
    if not response or not isinstance(response, str):
        return response

    if not _should_use_genai():
        return response

    try:
        # Preferred path: OpenAI SDK v1.x
        client = _get_openai_client()
        messages = _compose_messages(response, context, high_anthropomorphism)
        model_name = os.getenv("HICXAI_OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("HICXAI_TEMPERATURE", "0.2"))
        
        # For SHAP explanations, we need more tokens (especially for denials)
        # Response type determines token budget
        if response_type == "explanation" and context and context.get('explanation_type') == 'feature_importance':
            # SHAP explanations need more space (denial cases are typically 400-500 tokens)
            default_tokens = 600
        else:
            # Other responses can be shorter (validation, greetings, etc.)
            default_tokens = 400
        
        max_tokens = int(os.getenv("HICXAI_MAX_TOKENS", str(default_tokens)))

        if client is not None:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = completion.choices[0].message.content if completion and completion.choices else None
                return content or response
            except Exception:
                pass

        # Fallback: legacy openai SDK
        try:
            import openai  # type: ignore
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            # Support optional base URL on legacy sdk too
            base_url = (
                os.environ.get("HICXAI_OPENAI_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or None
            )
            if base_url:
                try:
                    openai.base_url = base_url  # type: ignore[attr-defined]
                except Exception:
                    pass
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion["choices"][0]["message"]["content"] if completion else None
            return content or response
        except Exception:
            return response
    except Exception:
        # Never break the app if the API call fails
        return response
