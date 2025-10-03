"""
Natural conversation helpers for v1: optional GPT paraphrasing (gpt-4o-mini by default).

Behavior:
- If OPENAI_API_KEY is set (via env or Streamlit Secrets) and HICXAI_GENAI is not 'off',
    use OpenAI to rewrite explanations in the selected style (HICXAI_STYLE: short|detailed|actionable).
- Otherwise, return the original text unchanged.

Notes:
- Keep outputs faithful: do not invent numbers or facts; preserve lists and key points.
- This module is optional. LoanAssistant guards imports accordingly.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

# Try to import streamlit to fetch secrets when running on Streamlit Cloud
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    st = None  # type: ignore


def _should_use_genai() -> bool:
    if os.getenv("HICXAI_GENAI", "on").lower() in {"0", "false", "off"}:
        return False
    # Allow pulling key from Streamlit Secrets when not present in env
    if not os.getenv("OPENAI_API_KEY") and st is not None:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)  # type: ignore[attr-defined]
            if key:
                os.environ["OPENAI_API_KEY"] = str(key)
        except Exception:
            pass
    return bool(os.getenv("OPENAI_API_KEY"))


def _get_openai_client():
    """Return an OpenAI client configured from environment/Streamlit secrets.

    Honors optional base URL (HICXAI_OPENAI_BASE_URL or OPENAI_BASE_URL) for proxies.
    """
    # Ensure key loaded from st.secrets if available
    _ = _should_use_genai()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = (
        os.environ.get("HICXAI_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or None
    )
    # Prefer OpenAI SDK v1.x
    try:
        from openai import OpenAI  # type: ignore
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)
    except Exception:
        # Legacy SDK fallback will be handled in enhance_response
        return None


def _build_system_prompt(style: str) -> str:
    base = (
        "You are Luna, a friendly AI loan assistant. Rephrase the provided explanation "
        "to be clear, warm, and helpful while preserving all factual content, numbers, and lists. "
        "Do not fabricate data. Do not change numeric values. Avoid hedging."
    )
    if style == "short":
        base += " Be concise (2-4 sentences)."
    elif style == "actionable":
        base += " Highlight 2-3 concrete next steps or changes a user can make."
    else:  # detailed
        base += " Provide a clear, well-structured explanation (short paragraphs or bullets)."
    return base


def _compose_messages(response: str, context: Optional[Dict[str, Any]], style: str):
    sys_prompt = _build_system_prompt(style)
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


def enhance_response(response: str, context: Optional[Dict[str, Any]] = None, response_type: str = "explanation") -> str:
    """Optionally paraphrase an explanation using OpenAI (gpt-4o-mini by default).

    If OpenAI is not configured, returns the original response unchanged.
    Style is controlled via HICXAI_STYLE: short | detailed | actionable.
    """
    if not response or not isinstance(response, str):
        return response

    if not _should_use_genai():
        return response

    style = os.getenv("HICXAI_STYLE", "detailed").strip().lower()
    try:
        # Preferred path: OpenAI SDK v1.x
        client = _get_openai_client()
        messages = _compose_messages(response, context, style)
        model_name = os.getenv("HICXAI_OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("HICXAI_TEMPERATURE", "0.2"))
        max_tokens = int(os.getenv("HICXAI_MAX_TOKENS", "300"))

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
"""
Natural Language Generation Module using Ollama
Makes conversations more natural while staying strictly within system constraints
"""

import requests
import json
import re
from typing import Dict, Any, Optional

class NaturalConversationGenerator:
    def __init__(self, ollama_url="http://localhost:11434", model="llama3.2:latest"):
        self.ollama_url = ollama_url
        self.model = model
        self.is_available = self._check_ollama_availability()
        
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # First check if Ollama service is responding
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                # Check for exact model or any llama3.2 variant
                model_name = self.model.split(':')[0]  # e.g., 'llama3.2'
                for model in models:
                    if model['name'].startswith(model_name):
                        return True
            return False
        except requests.exceptions.ConnectionError:
            # Ollama service not running
            return False
        except requests.exceptions.Timeout:
            # Ollama service taking too long to respond
            return False
        except Exception:
            # Any other error
            return False
    
    def enhance_loan_response(self, system_response: str, conversation_context: Dict[str, Any]) -> str:
        """
        Make loan application responses more natural using Ollama
        STRICTLY guided by system constraints - no additional information
        """
        if not self.is_available:
            return system_response  # Fallback to original response
        
        # Extract conversation state
        state = conversation_context.get('state', 'unknown')
        completion = conversation_context.get('completion_percentage', 0)
        current_field = conversation_context.get('current_field', '')
        
        # Create constrained prompt for natural conversation
        prompt = self._create_loan_prompt(system_response, state, completion, current_field)
        
        try:
            enhanced_response = self._call_ollama(prompt)
            return self._validate_and_filter_response(enhanced_response, system_response)
        except:
            return system_response  # Fallback on any error
    
    def enhance_explanation_response(self, system_explanation: str, explanation_type: str, user_question: str) -> str:
        """
        Make XAI explanations more conversational while keeping technical accuracy
        STRICTLY preserves all technical information from system
        """
        if not self.is_available:
            return system_explanation
        
        prompt = self._create_explanation_prompt(system_explanation, explanation_type, user_question)
        
        try:
            enhanced_explanation = self._call_ollama(prompt)
            return self._validate_explanation_response(enhanced_explanation, system_explanation)
        except:
            return system_explanation
    
    def _create_loan_prompt(self, system_response: str, state: str, completion: float, current_field: str) -> str:
        """Create constrained prompt for loan application conversations optimized for LLaMA 3.2"""
        return f"""Rewrite this message to sound more friendly and natural while keeping ALL information identical:

"{system_response}"

Rules:
- Keep ALL numbers, percentages, field names exactly the same
- Keep the same meaning and requirements  
- Make it sound more conversational and warm
- Don't add extra information or commentary
- Just return the improved version, nothing else

Improved:"""

    def _create_explanation_prompt(self, system_explanation: str, explanation_type: str, user_question: str) -> str:
        """Create constrained prompt for XAI explanations optimized for LLaMA 3.2"""
        return f"""Make this technical explanation more conversational while keeping ALL details exactly the same:

"{system_explanation}"

Rules:
- Keep ALL numbers, percentages, feature names exactly as written
- Don't add new explanations or information
- Make it easier to understand but technically identical
- Just return the improved explanation, nothing else

Improved explanation:"""

    def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama"""
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,  # Very low for strict adherence
                    'top_p': 0.8,        
                    'max_tokens': 100,   # Keep responses concise
                    'stop': ['\n\n', '\n', 'Original:', 'Rules:', 'Requirements:', 'Note:', 'Remember:']
                }
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def _validate_and_filter_response(self, enhanced_response: str, original_response: str) -> str:
        """Validate enhanced response doesn't add unauthorized information"""
        
        # Clean up the response
        enhanced = enhanced_response.strip()
        
        # Remove any meta-commentary that LLaMA might add
        lines = enhanced.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that are clearly meta-commentary
            if (line and 
                not line.startswith('Here') and
                not line.startswith('This') and  
                not line.startswith('I ') and
                not line.startswith('The ') and
                not line.lower().startswith('enhanced') and
                not line.lower().startswith('improved') and
                not line.lower().startswith('natural') and
                not line.lower().startswith('conversational') and
                not line.startswith('Rules') and
                not line.startswith('Requirements') and
                ':' not in line[:20]):  # Skip lines that look like labels
                filtered_lines.append(line)
        
        # Take the first meaningful line if multiple lines
        if filtered_lines:
            enhanced = filtered_lines[0].strip()
        else:
            enhanced = enhanced_response.strip()
        
        # Remove quotes if LLaMA added them
        if enhanced.startswith('"') and enhanced.endswith('"'):
            enhanced = enhanced[1:-1]
        
        # Safety checks
        if (not enhanced or 
            len(enhanced) > len(original_response) * 2 or 
            len(enhanced) < len(original_response) * 0.3):
            return original_response
        
        # Check if essential information is preserved (percentages, numbers)
        import re
        original_numbers = re.findall(r'\d+%|\d+\.\d+|\d+', original_response)
        enhanced_numbers = re.findall(r'\d+%|\d+\.\d+|\d+', enhanced)
        
        # If numbers don't match, use original
        if original_numbers and set(original_numbers) != set(enhanced_numbers):
            return original_response
        
        return enhanced
    
    def _validate_explanation_response(self, enhanced_explanation: str, original_explanation: str) -> str:
        """Validate explanation enhancement preserves technical accuracy"""
        
        # Extract numbers from both responses to ensure they match
        original_numbers = re.findall(r'\d+\.?\d*', original_explanation)
        enhanced_numbers = re.findall(r'\d+\.?\d*', enhanced_explanation)
        
        # If numbers don't match, use original
        if set(original_numbers) != set(enhanced_numbers):
            return original_explanation
        
        enhanced = self._validate_and_filter_response(enhanced_explanation, original_explanation)
        
        # Additional safety for explanations
        if ('feature' in original_explanation.lower() and 
            'feature' not in enhanced.lower()):
            return original_explanation
            
        return enhanced

class ConversationEnhancer:
    """Main class to enhance conversations using Ollama while maintaining system constraints"""
    
    def __init__(self):
        self.generator = NaturalConversationGenerator()
        self.enhancement_enabled = self.generator.is_available
        
    def enhance_loan_assistant_response(self, response: str, conversation_state: Dict[str, Any]) -> str:
        """Enhance loan assistant responses to be more natural"""
        if not self.enhancement_enabled:
            return response
            
        return self.generator.enhance_loan_response(response, conversation_state)
    
    def enhance_xai_explanation(self, explanation: str, explanation_type: str, user_question: str) -> str:
        """Enhance XAI explanations to be more conversational"""
        if not self.enhancement_enabled:
            return explanation
            
        return self.generator.enhance_explanation_response(explanation, explanation_type, user_question)
    
    def is_available(self) -> bool:
        """Check if enhancement is available"""
        return self.enhancement_enabled

# Global instance for easy importing
conversation_enhancer = ConversationEnhancer()

def enhance_response(response: str, context: Dict[str, Any] = None, response_type: str = "loan") -> str:
    """Convenience function to enhance any response"""
    if response_type == "loan":
        return conversation_enhancer.enhance_loan_assistant_response(response, context or {})
    elif response_type == "explanation":
        explanation_type = context.get('explanation_type', 'general') if context else 'general'
        user_question = context.get('user_question', '') if context else ''
        return conversation_enhancer.enhance_xai_explanation(response, explanation_type, user_question)
    else:
        return response