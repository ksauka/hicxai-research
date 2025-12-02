"""
Loan Application Assistant - Multi-turn Conversational Agent
This module handles the conversational flow for collecting loan application information.
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from ab_config import config
import pandas as pd
import numpy as np
import difflib
from xai_methods import get_friendly_feature_name, FEATURE_DISPLAY_NAMES

# Import natural conversation enhancer
try:
    from natural_conversation import enhance_response, enhance_validation_message
    NATURAL_CONVERSATION_AVAILABLE = True
except ImportError:
    NATURAL_CONVERSATION_AVAILABLE = False
    def enhance_response(response, context=None, response_type="loan"):
        return response
    def enhance_validation_message(field, user_input, expected_format, attempt=1):
        return None  # Fallback returns None to use hardcoded messages

class ConversationState(Enum):
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    REVIEWING = "reviewing"
    PROCESSING = "processing"
    COMPLETE = "complete"
    EXPLAINING = "explaining"

@dataclass
class LoanApplication:
    # Personal Information
    age: Optional[int] = None
    marital_status: Optional[str] = None
    
    # Employment Information
    workclass: Optional[str] = None
    occupation: Optional[str] = None
    hours_per_week: Optional[int] = None
    
    # Education
    education: Optional[str] = None
    education_num: Optional[int] = None
    
    # Financial Information
    capital_gain: Optional[int] = None
    capital_loss: Optional[int] = None
    
    # Demographics
    race: Optional[str] = None
    sex: Optional[str] = None
    native_country: Optional[str] = None
    relationship: Optional[str] = None
    
    # Application metadata
    is_complete: bool = False
    completion_percentage: float = 0.0
    loan_approved: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model prediction"""
        return {
            'age': self.age,
            'workclass': self.workclass,
            'education': self.education,
            'education_num': self.education_num,
            'marital_status': self.marital_status,
            'occupation': self.occupation,
            'relationship': self.relationship,
            'race': self.race,
            'sex': self.sex,
            'capital_gain': self.capital_gain or 0,
            'capital_loss': self.capital_loss or 0,
            'hours_per_week': self.hours_per_week,
            'native_country': self.native_country
        }
    
    def calculate_completion(self) -> float:
        """Calculate completion percentage"""
        total_fields = 13
        completed_fields = sum(1 for field_name, field_def in self.__dataclass_fields__.items() 
                              if field_name not in ['is_complete', 'completion_percentage'] 
                              and getattr(self, field_name) is not None)
        self.completion_percentage = (completed_fields / total_fields) * 100
        self.is_complete = self.completion_percentage >= 80  # 80% completion threshold
        return self.completion_percentage

class LoanAssistant:
    def __init__(self, agent):
        self.agent = agent
        self.conversation_state = ConversationState.GREETING
        self.application = LoanApplication()
        self.conversation_history = []
        self.current_field = None
        self.field_attempts = {}
        self.last_shap_result = None  # Store SHAP results for visualization
        self.show_what_if_lab = False  # Show What‑if Lab in UI when user asks what-if
        # Allowed categorical values (canonical forms)
        self.allowed_values = {
            'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
            'education': ['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate'],
            'marital_status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
            'occupation': ['Tech-support','Craft-repair','Other-service','Sales','Exec-managerial','Prof-specialty','Handlers-cleaners','Machine-op-inspct','Adm-clerical','Farming-fishing','Transport-moving','Priv-house-serv','Protective-serv','Armed-Forces','?'],
            'sex': ['Male', 'Female'],
            'race': ['Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'White', 'Other'],
            'native_country': ['United-States', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'Vietnam', 'Yugoslavia', '?'],
            'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
        }
        
        # Field collection order and prompts
        self.field_order = [
            'age', 'workclass', 'education', 'marital_status', 
            'occupation', 'hours_per_week', 'sex', 'race',
            'native_country', 'relationship', 'capital_gain', 'capital_loss'
        ]
        
        self.field_prompts = {
            'age': "What's your age?",
            'workclass': "What's your employment type? (e.g., Private sector, Self-employed, Federal/Local/State government, etc.)",
            'education': "What's your highest education level? (e.g., Bachelor's, High school graduate, Master's, Some college, Associate's degree, Doctorate, etc.)",
            'marital_status': "What's your marital status? (e.g., Married, Divorced, Never married, Separated, Widowed, etc.)",
            'occupation': "What's your occupation? (e.g., Tech support, Sales, Professional, Management, Administrative, Farming, etc.)",
            'hours_per_week': "How many hours per week do you work?",
            'sex': "What's your gender? (Male/Female)",
            'race': "What's your race? (e.g., White, Asian-Pacific Islander, Indigenous American, Black, Other)",
            'native_country': "What's your native country? (e.g., United States, Canada, England, Germany, etc.)",
            'relationship': "What's your relationship status? (e.g., Wife, Own-child, Husband, Not in family, Other relative, Unmarried)",
            'capital_gain': "Enter your capital gains for this year. Range: -5000 (losses) to 9000. Enter 0 if none.",
            'capital_loss': "Do you have any capital losses this year? (Enter amount or 0 if none)"
        }
        
        self.validation_rules = {
            'age': {'type': 'int', 'min': 17, 'max': 90},
            'hours_per_week': {'type': 'int', 'min': 1, 'max': 99},
            'capital_gain': {'type': 'int', 'min': -5000, 'max': 9000},
            'capital_loss': {'type': 'int', 'min': 0, 'max': 4356},
            'education_num': {'type': 'int', 'min': 1, 'max': 16}
        }
        
        # Step mapping for progress tracking
        self.field_to_step = {
            'age': 1, 'sex': 2,  # Personal Info
            'workclass': 3, 'occupation': 4, 'hours_per_week': 5,  # Employment
            'education': 6,  # Education
            'capital_gain': 7, 'capital_loss': 8,  # Financial
            'native_country': 9, 'marital_status': 10, 'relationship': 10, 'race': 10  # Background
        }
        
        self.step_descriptions = {
            1: "Personal Info - Age", 2: "Personal Info - Gender",
            3: "Employment - Work Class", 4: "Employment - Occupation", 5: "Employment - Hours",
            6: "Education Level",
            7: "Financial - Capital Gains", 8: "Financial - Capital Losses",
            9: "Background - Country", 10: "Background - Demographics"
        }

    # ----- Helper methods for version-aware messaging -----
    def _is_v1(self) -> bool:
        try:
            return getattr(config, 'version', 'v0') == 'v1'
        except Exception:
            return False

    def _pretty_field(self, field: str) -> str:
        return field.replace('_', ' ').title()

    def _format_error(self, field: str, base_msg: str, allowed: Optional[List[str]] = None) -> str:
        if not self._is_v1():
            # v0: concise, technical
            if allowed:
                return f"{base_msg} Valid options are: {', '.join(allowed)}."
            return base_msg
        # v1: anthropomorphic tone
        msg = f"I might be misreading this. For {self._pretty_field(field)}, {base_msg}"
        if allowed:
            sample = allowed[:8]
            more = " (+ more)" if len(allowed) > 8 else ""
            msg += f"\n\nHere are some examples I can accept: {', '.join(sample)}{more}."
        msg += "\nIf you'd like, I can list all choices or you can pick from the buttons."
        return msg

    def _format_warning(self, field: str, warn_msg: str) -> str:
        if not self._is_v1():
            return warn_msg
        return f"Quick heads‑up about {self._pretty_field(field)}: {warn_msg} If that was intentional, we’ll proceed; otherwise feel free to adjust."

    def _suggest_categorical(self, field: str, value: str, n: int = 3) -> List[str]:
        """Suggest close categorical options using fuzzy match."""
        allowed = self.allowed_values.get(field, [])
        if not allowed:
            return []
        return difflib.get_close_matches(str(value), allowed, n=n, cutoff=0.6)

    def handle_message(self, user_input: str) -> str:
        """Main message handler with enhanced natural conversation and XAI routing"""
        if not user_input.strip():
            return "I didn't catch that. Could you please say something?"
        
        # Check for XAI questions regardless of current state once a decision has been presented
        user_lower = user_input.lower()
        xai_keywords = ['what if', 'why', 'explain', 'how', 'which factors', 'feature importance',
                        'counterfactual', 'simple rules', 'rule-based', 'rule based', 'rules', 'anchor',
                        'what changes', 'what would happen']

        # If we've already provided a decision (loan_approved is True/False), allow immediate XAI routing
        if self.application.loan_approved is not None and any(keyword in user_lower for keyword in xai_keywords):
            return self._handle_explanation(user_input)
        
        # Route to appropriate handler based on conversation state
        if self.conversation_state == ConversationState.GREETING:
            return self._handle_greeting(user_input)
        elif self.conversation_state == ConversationState.COLLECTING_INFO:
            return self._handle_info_collection(user_input)
        elif self.conversation_state == ConversationState.REVIEWING:
            return self._handle_review(user_input)
        elif self.conversation_state == ConversationState.PROCESSING:
            return self._handle_processing(user_input)
        elif self.conversation_state == ConversationState.COMPLETE:
            return self._handle_complete(user_input)
        elif self.conversation_state == ConversationState.EXPLAINING:
            return self._handle_explanation(user_input)
        else:
            return "I'm not sure how to help with that. Please try again."

    def _handle_greeting(self, user_input: str) -> str:
        """Handle initial greeting and start application process"""
        greeting_keywords = ['hi', 'hello', 'hey', 'start', 'apply', 'loan', 'application']
        
        if config.show_anthropomorphic:
            # High anthropomorphism: Warm, personal, conversational
            if any(keyword in user_input.lower() for keyword in greeting_keywords) or user_input.lower() in ['yes', 'y']:
                self.conversation_state = ConversationState.COLLECTING_INFO
                base_greeting = ("Hello! I'm Luna, your personal loan application assistant. 😊 I will process your information and provide you with your loan qualification results. If you have any questions about the results, feel free to ask!\n\n"
                       "**I will collect information step by step** (not all at once) and you can **track your progress on the Progress Tracker** in the sidebar:\n"
                       "• **Step 1-2:** Personal Information (Age, Gender, etc.)\n"
                       "• **Step 3-5:** Employment Details (Work Class, Occupation, Hours)\n"
                       "• **Step 6:** Education Level\n"
                       "• **Step 7-8:** Financial Information (Capital Gains/Losses)\n"
                       "• **Step 9-10:** Background & Relationship Status\n\n"
                       "**Check the blue progress bar on the left to see your completion status!**\n\n"
                       "Let's start with **Step 1:**")
                
                # Enhance greeting with LLM
                if NATURAL_CONVERSATION_AVAILABLE:
                    try:
                        enhanced = enhance_response(base_greeting, {}, "greeting", high_anthropomorphism=True)
                        if enhanced and len(enhanced.strip()) > 20:
                            return f"{enhanced}\n\n{self._get_next_question()}"
                    except Exception:
                        pass
                
                return f"{base_greeting}\n\n{self._get_next_question()}"
            else:
                base_prompt = ("Hi there! I'm Luna, your personal loan application assistant. 😊 I will process your information and provide you with your loan qualification results. If you have any questions about the results, feel free to ask!\n\n"
                       "**I will collect your information step by step** (not all at once) and you can **track your progress on the Progress Tracker** in the sidebar:\n"
                       "• **Step 1-2:** Personal Information (Age, Gender, etc.)\n"
                       "• **Step 3-5:** Employment Details (Work Class, Occupation, Hours)\n"
                       "• **Step 6:** Education Level\n"
                       "• **Step 7-8:** Financial Information (Capital Gains/Losses)\n"
                       "• **Step 9-10:** Background & Relationship Status\n\n"
                       "**Watch the blue progress bar fill up as we complete each step!**\n\n"
                       "Would you like to start your loan application? Just say 'yes' or 'start' to begin!")
                
                # Enhance with LLM
                if NATURAL_CONVERSATION_AVAILABLE:
                    try:
                        enhanced = enhance_response(base_prompt, {}, "greeting_prompt", high_anthropomorphism=True)
                        if enhanced and len(enhanced.strip()) > 20:
                            return enhanced
                    except Exception:
                        pass
                
                return base_prompt
        else:
            # Low anthropomorphism: Technical, concise, machine-like
            if any(keyword in user_input.lower() for keyword in greeting_keywords) or user_input.lower() in ['yes', 'y']:
                self.conversation_state = ConversationState.COLLECTING_INFO
                base_greeting = ("AI Loan Assistant initialized. I will collect your information and process your loan qualification.\n\n"
                       "**Information collection process** (sequential):\n"
                       "• **Step 1-2:** Personal data (Age, Gender)\n"
                       "• **Step 3-5:** Employment data (Work Class, Occupation, Hours)\n"
                       "• **Step 6:** Education level\n"
                       "• **Step 7-8:** Financial data (Capital Gains/Losses)\n"
                       "• **Step 9-10:** Demographics & Relationship\n\n"
                       "Progress tracking available in sidebar.\n\n"
                       "**Step 1:**")
                
                # Enhance with LLM for professional tone
                if NATURAL_CONVERSATION_AVAILABLE:
                    try:
                        enhanced = enhance_response(base_greeting, {}, "greeting", high_anthropomorphism=False)
                        if enhanced and len(enhanced.strip()) > 20:
                            return f"{enhanced}\n\n{self._get_next_question()}"
                    except Exception:
                        pass
                
                return f"{base_greeting}\n\n{self._get_next_question()}"
            else:
                base_prompt = ("AI Loan Assistant system ready. I will collect your data and evaluate your loan qualification.\n\n"
                       "**Data collection process** (sequential):\n"
                       "• **Step 1-2:** Personal data (Age, Gender)\n"
                       "• **Step 3-5:** Employment data (Work Class, Occupation, Hours)\n"
                       "• **Step 6:** Education level\n"
                       "• **Step 7-8:** Financial data (Capital Gains/Losses)\n"
                       "• **Step 9-10:** Demographics & Relationship\n\n"
                       "Progress tracking available in sidebar.\n\n"
                       "Enter 'yes' or 'start' to begin data collection.")
                
                # Enhance with LLM for professional tone
                if NATURAL_CONVERSATION_AVAILABLE:
                    try:
                        enhanced = enhance_response(base_prompt, {}, "greeting_prompt", high_anthropomorphism=False)
                        if enhanced and len(enhanced.strip()) > 20:
                            return enhanced
                    except Exception:
                        pass
                
                return base_prompt

    def _handle_info_collection(self, user_input: str) -> str:
        """Handle information collection phase"""
        if user_input.lower() in ['quit', 'exit', 'stop', 'cancel']:
            if config.show_anthropomorphic:
                return "Application cancelled. Feel free to start again anytime by saying 'hi'! 😊"
            else:
                return "Application process terminated. Restart available via 'hi' command."
        
        if user_input.lower() in ['review', 'check', 'status']:
            return self._show_progress()
        
        # Handle '?' - provide copyable list of valid options
        if user_input.strip() == '?' and self.current_field:
            return self._get_copyable_options(self.current_field)
        
        if user_input.lower() in ['help', 'help me', 'what do i do', 'stuck', 'confused']:
            if self.current_field:
                if config.show_anthropomorphic:
                    return f"No problem! Let me help you with {self.current_field.replace('_', ' ')}:\n\n{self._get_field_help(self.current_field)}\n\n✨ You can also use the quick-select buttons if available!"
                else:
                    return f"Help for {self.current_field.replace('_', ' ')}:\n\n{self._get_field_help(self.current_field)}\n\nQuick-select buttons available when applicable."
            else:
                if config.show_anthropomorphic:
                    return "I'm here to help! I'm collecting information for your loan application step by step. You can say 'review' to see your progress, or just answer the current question. 😊"
                else:
                    return "Data collection in progress. Enter 'review' for status or respond to current query."
        
        # Process the current field
        if self.current_field:
            # Store the field we're processing BEFORE it gets changed
            completed_field = self.current_field
            result = self._process_field_input(self.current_field, user_input)
            if result['success']:
                completion = self.application.calculate_completion()
                
                # Check if we have all required information
                if self.application.is_complete:
                    self.conversation_state = ConversationState.REVIEWING
                    if config.show_anthropomorphic:
                        return (f"🎉 Fantastic! I've collected all the necessary information ({completion:.0f}% complete).\n\n"
                               + self._show_application_summary() + 
                               "\n\n💼 Would you like me to process your loan application now? (yes/no)")
                    else:
                        return (f"Data collection complete ({completion:.0f}%).\n\n"
                               + self._show_application_summary() + 
                               "\n\nProcess application? (yes/no)")
                else:
                    next_question = self._get_next_question()
                    if next_question:
                        # Get step information for the NEXT field (self.current_field is now updated)
                        next_step = self.field_to_step.get(self.current_field, 0) if self.current_field else 0
                        # Get the value that was just set for the COMPLETED field
                        completed_value = str(getattr(self.application, completed_field, ''))
                        # Use LLM-enhanced success message with the correct completed field
                        success_msg = self._get_success_message(
                            completed_field,
                            completed_value,
                            completion,
                            next_step
                        )
                        return f"{success_msg}\n{next_question}"
                    else:
                        # Fallback if no next question
                        self.conversation_state = ConversationState.REVIEWING
                        return (f"Great! I have most of the information ({completion:.0f}% complete).\n\n"
                               + self._show_application_summary() + 
                               "\n\nWould you like me to process your loan application? (yes/no)")
            else:
                return result['message']
        else:
            return self._get_next_question()

    def _handle_review(self, user_input: str) -> str:
        """Handle application review phase"""
        if user_input.lower() in ['yes', 'y', 'proceed', 'process', 'submit']:
            self.conversation_state = ConversationState.PROCESSING
            return self._process_application()
        elif user_input.lower() in ['no', 'n', 'edit', 'change', 'modify']:
            self.conversation_state = ConversationState.COLLECTING_INFO
            return ("No problem! What would you like to change? You can say things like:\n"
                   "- 'Change my age to 30'\n"
                   "- 'Update my occupation'\n"
                   "- 'My education is Masters'\n\n"
                   "Or just tell me which field you'd like to update.")
        else:
            return ("Please let me know if you'd like to proceed with the application (yes) or make changes (no).\n\n"
                   + self._show_application_summary())

    def _handle_processing(self, user_input: str) -> str:
        """Handle post-processing phase"""
        if self._is_xai_query(user_input):
            self.conversation_state = ConversationState.EXPLAINING
            return self._handle_explanation(user_input)
        elif user_input.lower() in ['new', 'another', 'restart', 'again']:
            return self._restart_application()
        else:
            if config.explanation == "none":
                return ("Your application has been processed! You can:\n"
                       "- Say 'new' to start a new application\n"
                       "- For further details, please visit your local branch")
            else:
                return ("Your application has been processed! Here are your options:\n"
                       "- Say 'explain' to understand how the decision was made\n"
                       "- Say 'new' to start a new application\n"
                       "- Ask me any questions about the result")

    def _handle_complete(self, user_input: str) -> str:
        """Handle interactions when application is complete"""
        user_lower = user_input.lower()
        
        if user_lower in ['new', 'another', 'restart', 'again', 'start over']:
            return self._restart_application()
        elif self._is_xai_query(user_input):
            return self._handle_explanation(user_input)
        else:
            # Check if the application has been processed
            if self.application.loan_approved is None:
                return ("It looks like your application hasn't been processed yet. Please complete the application first, "
                       "or if you believe this is an error, try saying 'start over' to begin a new application.")
            
            # Provide helpful guidance
            result = self.application.loan_approved
            status = "approved" if result else "denied"
            
            if config.explanation == "none":
                # Simple message without explanation options
                return (f"Your loan application has been {status}. "
                       f"For further details, please visit your local branch. "
                       f"Would you like to start a new application? Just say 'new' or 'restart'.")
            else:
                # Full options when explanations are available
                return (f"Your loan application has been {status}! Here's what you can do:\n\n"
                       "🔍 **Ask for explanations:** 'Why was I approved/denied?' or 'Explain the decision'\n"
                       "🔧 **What-if analysis:** 'What if my income was higher?' or 'What changes would help?'\n"
                       "📊 **Feature importance:** 'Which factors were most important?'\n"
                       "🆕 **New application:** 'Start a new application'\n\n"
                       "Just ask me anything about your loan decision!")

    def _handle_explanation(self, user_input: str) -> str:
        """Handle explanation requests using the XAgent-compatible approach"""
        # If explanations are disabled (none condition), provide generic response
        if config.explanation == "none":
            return ("I'm sorry, but detailed explanations are not available at this time. "
                   "For further information about your loan decision, please visit your local branch. "
                   "Would you like to start a new application?")
        
        try:
            # Set up the agent instance properly first
            self._setup_agent_instance()
            
            # Use XAgent-style matching approach
            features = self.agent.data.get('features', [])
            prediction = self.agent.predicted_class
            current_instance = self.agent.current_instance
            labels = self.agent.data.get('classes', [])
            
            # Use the match function like XAgent does
            matched_question = self.agent.nlu_model.match(user_input, features, prediction, current_instance, labels)
            
            # Debug logging
            print(f"🔍 DEBUG: Matched question: {matched_question}")
            print(f"🔍 DEBUG: Input: {user_input}")
            
            if matched_question != "unknown":
                # Good match found - get the label and map to XAI method
                try:
                    import pandas as pd
                    # Get the label for the matched question from the NLU dataframe
                    df_matches = self.agent.nlu_model.df.query('Question == @matched_question')
                    if len(df_matches) > 0:
                        label = df_matches['Label'].iloc[0]
                        xai_method = self.agent.nlu_model.map_label_to_xai_method(label)
                    else:
                        # Fallback: infer method from the matched question text
                        mq = (matched_question or '').lower()
                        if any(k in mq for k in ['rule', 'rule-based', 'rule based', 'anchor', 'simple requirement', 'minimum requirement']):
                            xai_method = 'anchor'
                            label = None
                        elif any(k in mq for k in ['what if', 'change', 'counterfactual']):
                            xai_method = 'dice'
                            label = None
                        else:
                            xai_method = 'shap'
                            label = None
                    
                    print(f"🔍 DEBUG: Label: {label}, XAI method: {xai_method}")
                    
                    # Heuristic override: if mapping returned 'general', infer from input
                    inferred_method = xai_method
                    if xai_method == 'general':
                        ui = (user_input or '').lower()
                        if any(k in ui for k in ['rule', 'rule-based', 'rule based', 'anchor']):
                            inferred_method = 'anchor'
                        elif any(k in ui for k in ['what if', 'change', 'different', 'counterfactual']):
                            inferred_method = 'dice'
                        else:
                            inferred_method = 'shap'
                    
                    # CRITICAL: Remap to available explanation method for this experimental condition
                    # Force all explanation requests to use the method enabled for this condition
                    inferred_method = inferred_method.lower().strip()  # Normalize
                    
                    if config.explanation == 'counterfactual':
                        # Conditions 3-4: Only counterfactual (DiCE) available
                        # Map ANYTHING except anchor to DiCE
                        if inferred_method != 'anchor':
                            print(f"🔍 DEBUG MAIN: Remapping '{inferred_method}' → 'dice' for counterfactual condition")
                            inferred_method = 'dice'
                    elif config.explanation == 'feature_importance':
                        # Conditions 5-6: Only feature importance (SHAP) available
                        # Map ANYTHING except anchor to SHAP
                        if inferred_method != 'anchor':
                            print(f"🔍 DEBUG MAIN: Remapping '{inferred_method}' → 'shap' for feature_importance condition")
                            inferred_method = 'shap'
                    # Note: 'anchor' is always available as a baseline in all conditions

                    # If user is asking what-if (counterfactual), enable What‑if Lab in UI
                    if inferred_method == 'dice':
                        self.show_what_if_lab = True
                    # Create intent result in the format expected by route_to_xai_method
                    intent_result = {
                        'intent': inferred_method,
                        'label': label,
                        'matched_question': matched_question
                    }
                    
                    # Route to the correct XAI method
                    from xai_methods import route_to_xai_method
                    explanation_result = route_to_xai_method(self.agent, intent_result)
                    
                    explanation = explanation_result.get('explanation', 'Sorry, I could not generate an explanation.')
                    
                    # Store SHAP results for visualization (if available)
                    if (inferred_method == 'shap' and 
                        isinstance(explanation_result, dict) and 
                        ('feature_impacts' in explanation_result or 'shap_values' in explanation_result)):
                        self.last_shap_result = explanation_result
                    else:
                        self.last_shap_result = None
                    
                except Exception as e:
                    explanation = f"Sorry, I couldn't generate that explanation right now. Error: {str(e)}"
                
            else:
                # No good match found - fallback to general agent
                # But first apply remapping for experimental conditions
                try:
                    print(f"🔍 DEBUG FALLBACK PATH: user_input='{user_input}'")
                    print(f"🔍 DEBUG CONFIG: explanation={config.explanation}, show_shap={config.show_shap_visualizations}, show_counterfactual={config.show_counterfactual}")
                    
                    intent_result_fallback, confidence, suggestions = self.agent.nlu_model.classify_intent(user_input)
                    
                    print(f"🔍 DEBUG FALLBACK: classify_intent returned: {intent_result_fallback}")
                    
                    if isinstance(intent_result_fallback, dict) and 'intent' in intent_result_fallback:
                        inferred_method = intent_result_fallback['intent'].lower().strip()
                        
                        print(f"🔍 DEBUG FALLBACK: Original intent: '{inferred_method}'")
                        print(f"🔍 DEBUG FALLBACK: Config explanation: '{config.explanation}'")
                        
                        # CRITICAL: Remap to available explanation method for this experimental condition
                        # Force all explanation requests to use the method enabled for this condition
                        if config.explanation == 'counterfactual':
                            # Conditions 3-4: Only counterfactual (DiCE) available
                            # Map ANYTHING except anchor to DiCE
                            if inferred_method != 'anchor':
                                print(f"🔍 DEBUG FALLBACK: Remapping '{inferred_method}' → 'dice' for counterfactual condition")
                                inferred_method = 'dice'
                        elif config.explanation == 'feature_importance':
                            # Conditions 5-6: Only feature importance (SHAP) available
                            # Map ANYTHING except anchor to SHAP
                            if inferred_method != 'anchor':
                                print(f"🔍 DEBUG FALLBACK: Remapping '{inferred_method}' → 'shap' for feature_importance condition")
                                inferred_method = 'shap'
                        
                        print(f"🔍 DEBUG FALLBACK: Final method after remapping: '{inferred_method}'")
                        
                        # Update the intent in the result
                        intent_result_fallback['intent'] = inferred_method
                        
                        # Route directly to XAI method with remapped intent
                        from xai_methods import route_to_xai_method
                        explanation_result = route_to_xai_method(self.agent, intent_result_fallback)
                        explanation = explanation_result.get('explanation', 'Sorry, I could not generate an explanation.')
                        
                        # Store SHAP results if applicable
                        if (inferred_method == 'shap' and 
                            isinstance(explanation_result, dict) and 
                            ('feature_impacts' in explanation_result or 'shap_values' in explanation_result)):
                            self.last_shap_result = explanation_result
                        else:
                            self.last_shap_result = None
                    else:
                        # Truly couldn't classify - use original fallback
                        explanation = self.agent.handle_user_input(user_input)
                except Exception as e:
                    print(f"🔍 DEBUG FALLBACK ERROR: {str(e)}")
                    explanation = self.agent.handle_user_input(user_input)
            
            # Format the explanation nicely
            if isinstance(explanation, dict) and 'explanation' in explanation:
                formatted_explanation = explanation['explanation']
            elif isinstance(explanation, str):
                formatted_explanation = explanation
            else:
                formatted_explanation = str(explanation)

            # XAI methods already enhance with LLM - no need to enhance again
            # Double enhancement causes wrapper text and duplicate content
            # For LOW anthropomorphism, return explanation without header (already has its own structure)
            if not config.show_anthropomorphic:
                return formatted_explanation
            else:
                return f"**Explanation:**\n\n{formatted_explanation}"
            
        except Exception as e:
            return ("I'm sorry, I couldn't generate that explanation right now. "
                   "Would you like to try asking differently or start a new application? "
                   f"Error: {str(e)}")

    def _is_xai_query(self, text: str) -> bool:
        """Decide if user input is an XAI question using NLU intent mapping with a small fallback.
        Uses sentence-transformers-based classifier; falls back to lightweight keywords if needed."""
        try:
            intent_result, _, _ = self.agent.nlu_model.classify_intent(text)
            if isinstance(intent_result, dict):
                return intent_result.get('intent') in {'shap', 'dice', 'anchor'}
        except Exception:
            pass
        # Fallback minimal heuristic
        t = (text or '').lower()
        return any(k in t for k in ['why', 'explain', 'factor', 'feature', 'importance', 'what if', 'change', 'counterfactual', 'rule', 'anchor'])

    def _process_field_input(self, field: str, user_input: str) -> Dict[str, Any]:
        """Process input for a specific field"""
        try:
            # Extract value from user input
            value = self._extract_field_value(field, user_input)
            
            if value is None:
                self.field_attempts[field] = self.field_attempts.get(field, 0) + 1
                if self.field_attempts[field] >= 3:
                    return {
                        'success': False,
                        'message': f"I'm having trouble understanding your {field.replace('_', ' ')}. Let me provide some specific examples to help:\n\n{self._get_field_help(field)}\n\nOr you can say 'help' for more guidance."
                    }
                
                # Provide context-specific help on second attempt
                help_msg = self._get_smart_validation_message(field, user_input, self.field_attempts[field])
                return {
                    'success': False,
                    'message': help_msg
                }
            
            # Validate the value
            validation_result = self._validate_field_value(field, value)
            if not validation_result['valid']:
                self.field_attempts[field] = self.field_attempts.get(field, 0) + 1
                
                # Use smart validation message instead of generic one
                if self.field_attempts[field] >= 3:
                    err = f"I'm having trouble understanding your {field.replace('_', ' ')}. Let me provide some specific examples to help:\n\n{self._get_field_help(field)}\n\nOr type **'?'** to see all valid options you can copy-paste!"
                else:
                    err = self._get_smart_validation_message(field, user_input, self.field_attempts[field])
                
                return {
                    'success': False,
                    'message': err
                }
            
            # Set the normalized value if provided
            normalized = validation_result.get('normalized', value)
            setattr(self.application, field, normalized)
            self.field_attempts[field] = 0  # Reset attempts on success
            
            warn = validation_result.get('warning')
            
            # Generate natural confirmation message using LLM
            from ab_config import config
            if config.show_anthropomorphic:
                base_msg = f"Got it! {field.replace('_', ' ').title()}: {normalized}"
            else:
                base_msg = f"Confirmed. {field.replace('_', ' ').title()}: {normalized}"
            
            # Enhance with LLM for natural conversation
            if NATURAL_CONVERSATION_AVAILABLE:
                try:
                    context = {'field': field, 'value': normalized}
                    enhanced = enhance_response(
                        base_msg,
                        context,
                        "field_confirmation",
                        high_anthropomorphism=config.show_anthropomorphic
                    )
                    if enhanced and len(enhanced.strip()) > 5:
                        base_msg = enhanced
                except Exception:
                    pass  # Keep base message
            
            if warn:
                base_msg += f"\n\n⚠️ {self._format_warning(field, warn)}"
            
            return {
                'success': True,
                'message': base_msg
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': "Sorry, there was an error processing that. Please try again."
            }

    def _extract_field_value(self, field: str, user_input: str):
        """Extract field value from user input using pattern matching"""
        user_input = user_input.strip()
        
        # Numeric fields
        if field in ['age', 'hours_per_week', 'capital_gain', 'capital_loss', 'education_num']:
            # Look for numbers in the input
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                return int(numbers[0])
            return None
        
        # For categorical fields, try to match against known values
        elif field == 'sex':
            if re.search(r'\b(male|man|m)\b', user_input, re.IGNORECASE):
                return 'Male'
            elif re.search(r'\b(female|woman|f)\b', user_input, re.IGNORECASE):
                return 'Female'
        
        elif field == 'marital_status':
            input_lower = user_input.lower()
            if 'married' in input_lower and 'spouse' in input_lower:
                return 'Married-civ-spouse'
            elif 'married' in input_lower:
                return 'Married-civ-spouse'
            elif 'divorced' in input_lower:
                return 'Divorced'
            elif 'never' in input_lower or 'single' in input_lower:
                return 'Never-married'
            elif 'separated' in input_lower:
                return 'Separated'
            elif 'widow' in input_lower:
                return 'Widowed'
        
        # Handle "Other" responses for any categorical field
        if user_input.lower() in ['other', 'others', 'something else', 'none of the above']:
            return 'Other'
        
        # For other fields, return the input as-is for now
        # This can be enhanced with more sophisticated matching
        return user_input if user_input else None

    def _validate_field_value(self, field: str, value) -> Dict[str, Any]:
        """Validate field value against rules"""
        if field in self.validation_rules:
            rules = self.validation_rules[field]
            
            if rules['type'] == 'int':
                try:
                    int_val = int(value)
                    if int_val < rules.get('min', float('-inf')) or int_val > rules.get('max', float('inf')):
                        return {
                            'valid': False,
                            'message': f"Please enter a number between {rules.get('min', 'any')} and {rules.get('max', 'any')} for {self._pretty_field(field)}."
                        }
                    # Soft plausibility warnings for extreme-but-allowed values
                    warning = None
                    if field == 'hours_per_week' and int_val >= 80:
                        warning = "That is an unusually high number of weekly hours. Please confirm this is intentional."
                    if field == 'age' and int_val >= 85:
                        warning = (warning or "") + (" " if warning else "") + "Age is at the extreme upper bound of the dataset."
                    if field == 'capital_gain' and int_val > 5000:
                        warning = (warning or "") + (" " if warning else "") + "Note: Capital gains above $5,000 are relatively uncommon in this assessment."
                    if field == 'capital_loss' and int_val > 2000:
                        warning = (warning or "") + (" " if warning else "") + "Capital loss values above $2,000 are less common."
                    result = {'valid': True, 'message': '', 'normalized': int_val}
                    if warning:
                        result['warning'] = warning
                    return result
                except ValueError:
                    return {
                        'valid': False,
                        'message': f"Please enter a valid number for {self._pretty_field(field)}."
                    }

        # Enforce categorical whitelists (case-insensitive) and normalize to canonical values
        if field in getattr(self, 'allowed_values', {}):
            allowed = self.allowed_values[field]
            # Exact match
            if value in allowed:
                return {'valid': True, 'message': '', 'normalized': value}
            # Case-insensitive match
            val_norm = str(value).strip()
            for opt in allowed:
                if val_norm.lower() == opt.lower():
                    return {'valid': True, 'message': '', 'normalized': opt}
            
            # === COMPREHENSIVE FUZZY MATCHING FOR ALL CATEGORICAL FIELDS ===
            print(f"🔍 DEBUG: Fuzzy matching for field='{field}', val_norm='{val_norm}'")
            
            # Workclass: Employment type mappings
            if field == 'workclass':
                workclass_mapping = {
                    # Private sector variations
                    'private': 'Private', 'private sector': 'Private', 'pvt': 'Private',
                    'private company': 'Private', 'private employer': 'Private', 'pvt sector': 'Private',
                    'corporate': 'Private', 'company': 'Private', 'business': 'Private',
                    
                    # Self-employed not incorporated
                    'self employed': 'Self-emp-not-inc', 'self-employed': 'Self-emp-not-inc',
                    'self emp': 'Self-emp-not-inc', 'self employed not incorporated': 'Self-emp-not-inc',
                    'freelance': 'Self-emp-not-inc', 'freelancer': 'Self-emp-not-inc',
                    'independent': 'Self-emp-not-inc', 'contractor': 'Self-emp-not-inc',
                    'self': 'Self-emp-not-inc', 'own business': 'Self-emp-not-inc',
                    
                    # Self-employed incorporated
                    'self employed incorporated': 'Self-emp-inc', 'self-employed inc': 'Self-emp-inc',
                    'self emp inc': 'Self-emp-inc', 'incorporated': 'Self-emp-inc',
                    'own company': 'Self-emp-inc', 'business owner': 'Self-emp-inc',
                    
                    # Federal government
                    'federal': 'Federal-gov', 'federal government': 'Federal-gov',
                    'fed gov': 'Federal-gov', 'federal govt': 'Federal-gov',
                    'fed': 'Federal-gov', 'us government': 'Federal-gov',
                    'national government': 'Federal-gov',
                    
                    # Local government
                    'local': 'Local-gov', 'local government': 'Local-gov',
                    'local gov': 'Local-gov', 'local govt': 'Local-gov',
                    'city': 'Local-gov', 'municipal': 'Local-gov',
                    'county': 'Local-gov', 'city government': 'Local-gov',
                    
                    # State government
                    'state': 'State-gov', 'state government': 'State-gov',
                    'state gov': 'State-gov', 'state govt': 'State-gov',
                    
                    # Government (general - default to Federal)
                    'government': 'Federal-gov', 'govt': 'Federal-gov', 'gov': 'Federal-gov',
                    'public sector': 'Federal-gov', 'public': 'Federal-gov',
                    
                    # Other
                    'without pay': 'Without-pay', 'unpaid': 'Without-pay', 'volunteer': 'Without-pay',
                    'never worked': 'Never-worked', 'never': 'Never-worked', 'unemployed': 'Never-worked',
                    'unknown': '?', 'not sure': '?', 'prefer not to say': '?',
                }
                
                val_lower = val_norm.lower()
                if val_lower in workclass_mapping:
                    return {'valid': True, 'message': '', 'normalized': workclass_mapping[val_lower]}
            
            # Education: Education level mappings
            if field == 'education':
                education_mapping = {
                    # Doctorate variations
                    'phd': 'Doctorate', 'doctorate': 'Doctorate', 'doctoral': 'Doctorate',
                    'ph.d': 'Doctorate', 'ph.d.': 'Doctorate', 'doctor': 'Doctorate',
                    'doctoral degree': 'Doctorate',
                    
                    # Professional school
                    'professional': 'Prof-school', 'prof school': 'Prof-school',
                    'professional school': 'Prof-school', 'professional degree': 'Prof-school',
                    'law school': 'Prof-school', 'medical school': 'Prof-school',
                    'mba': 'Prof-school', 'jd': 'Prof-school', 'md': 'Prof-school',
                    
                    # Masters variations
                    'masters': 'Masters', 'master': 'Masters', "master's": 'Masters',
                    'masters degree': 'Masters', 'graduate degree': 'Masters',
                    'ms': 'Masters', 'm.s.': 'Masters', 'ma': 'Masters', 'm.a.': 'Masters',
                    'msc': 'Masters', 'grad school': 'Masters',
                    
                    # Bachelors variations
                    'bachelors': 'Bachelors', 'bachelor': 'Bachelors', "bachelor's": 'Bachelors',
                    'bachelors degree': 'Bachelors', 'undergraduate': 'Bachelors',
                    'college degree': 'Bachelors', 'university degree': 'Bachelors',
                    'bs': 'Bachelors', 'b.s.': 'Bachelors', 'ba': 'Bachelors', 'b.a.': 'Bachelors',
                    '4 year degree': 'Bachelors', 'four year degree': 'Bachelors',
                    
                    # Associates vocational
                    'associate vocational': 'Assoc-voc', 'assoc voc': 'Assoc-voc',
                    'associates voc': 'Assoc-voc', 'vocational': 'Assoc-voc',
                    'trade school': 'Assoc-voc', 'technical school': 'Assoc-voc',
                    
                    # Associates academic
                    'associate': 'Assoc-acdm', 'associates': 'Assoc-acdm',
                    'associate degree': 'Assoc-acdm', 'associates degree': 'Assoc-acdm',
                    'assoc': 'Assoc-acdm', 'aa': 'Assoc-acdm', 'as': 'Assoc-acdm',
                    '2 year degree': 'Assoc-acdm', 'two year degree': 'Assoc-acdm',
                    'community college': 'Assoc-acdm',
                    
                    # Some college
                    'some college': 'Some-college', 'college': 'Some-college',
                    'some university': 'Some-college', 'incomplete college': 'Some-college',
                    'partial college': 'Some-college', 'attended college': 'Some-college',
                    
                    # High school graduate
                    'high school': 'HS-grad', 'hs grad': 'HS-grad', 'hs-grad': 'HS-grad',
                    'high school graduate': 'HS-grad', 'high school diploma': 'HS-grad',
                    'hs diploma': 'HS-grad', 'diploma': 'HS-grad', 'graduated high school': 'HS-grad',
                    'secondary school': 'HS-grad', 'secondary': 'HS-grad',
                    
                    # 12th grade
                    '12th': '12th', '12th grade': '12th', 'grade 12': '12th',
                    'twelfth grade': '12th', 'senior year': '12th',
                    
                    # 11th grade
                    '11th': '11th', '11th grade': '11th', 'grade 11': '11th',
                    'eleventh grade': '11th', 'junior year': '11th',
                    
                    # 10th grade
                    '10th': '10th', '10th grade': '10th', 'grade 10': '10th',
                    'tenth grade': '10th', 'sophomore year': '10th',
                    
                    # 9th grade
                    '9th': '9th', '9th grade': '9th', 'grade 9': '9th',
                    'ninth grade': '9th', 'freshman year': '9th',
                    
                    # Elementary grades
                    '7th-8th': '7th-8th', '7th 8th': '7th-8th', 'middle school': '7th-8th',
                    '5th-6th': '5th-6th', '5th 6th': '5th-6th', 'elementary': '5th-6th',
                    '1st-4th': '1st-4th', '1st 4th': '1st-4th', 'primary school': '1st-4th',
                    'preschool': 'Preschool', 'pre school': 'Preschool', 'kindergarten': 'Preschool',
                }
                
                val_lower = val_norm.lower()
                print(f"🎓 DEBUG: Education check - val_lower='{val_lower}', in mapping: {val_lower in education_mapping}")
                if val_lower in education_mapping:
                    print(f"✅ DEBUG: Education match found: '{val_lower}' → '{education_mapping[val_lower]}'")
                    return {'valid': True, 'message': '', 'normalized': education_mapping[val_lower]}
            
            # Marital Status: Relationship status mappings
            if field == 'marital_status':
                marital_mapping = {
                    # Married civilian spouse
                    'married': 'Married-civ-spouse', 'married civilian': 'Married-civ-spouse',
                    'married civ spouse': 'Married-civ-spouse', 'spouse': 'Married-civ-spouse',
                    'wed': 'Married-civ-spouse', 'wedded': 'Married-civ-spouse',
                    
                    # Divorced
                    'divorced': 'Divorced', 'divorce': 'Divorced', 'div': 'Divorced',
                    'ex spouse': 'Divorced', 'former spouse': 'Divorced',
                    
                    # Never married
                    'never married': 'Never-married', 'single': 'Never-married',
                    'unmarried': 'Never-married', 'never wed': 'Never-married',
                    'not married': 'Never-married', 'bachelor': 'Never-married',
                    'bachelorette': 'Never-married',
                    
                    # Separated
                    'separated': 'Separated', 'sep': 'Separated', 'legal separation': 'Separated',
                    'legally separated': 'Separated',
                    
                    # Widowed
                    'widowed': 'Widowed', 'widow': 'Widowed', 'widower': 'Widowed',
                    
                    # Married spouse absent
                    'married spouse absent': 'Married-spouse-absent',
                    'spouse absent': 'Married-spouse-absent', 'separated married': 'Married-spouse-absent',
                    
                    # Married armed forces spouse
                    'married af spouse': 'Married-AF-spouse', 'military spouse': 'Married-AF-spouse',
                    'armed forces spouse': 'Married-AF-spouse', 'military': 'Married-AF-spouse',
                }
                
                val_lower = val_norm.lower()
                if val_lower in marital_mapping:
                    return {'valid': True, 'message': '', 'normalized': marital_mapping[val_lower]}
            
            # Occupation: Job type mappings
            if field == 'occupation':
                occupation_mapping = {
                    # Tech support
                    'tech support': 'Tech-support', 'tech': 'Tech-support', 'it': 'Tech-support',
                    'technical support': 'Tech-support', 'help desk': 'Tech-support',
                    'it support': 'Tech-support', 'computer support': 'Tech-support',
                    'technology': 'Tech-support',
                    
                    # Craft repair
                    'craft': 'Craft-repair', 'repair': 'Craft-repair', 'craft repair': 'Craft-repair',
                    'mechanic': 'Craft-repair', 'electrician': 'Craft-repair',
                    'plumber': 'Craft-repair', 'carpenter': 'Craft-repair',
                    'technician': 'Craft-repair', 'maintenance': 'Craft-repair',
                    
                    # Other service
                    'service': 'Other-service', 'other service': 'Other-service',
                    'customer service': 'Other-service', 'hospitality': 'Other-service',
                    'waiter': 'Other-service', 'waitress': 'Other-service',
                    'server': 'Other-service', 'food service': 'Other-service',
                    
                    # Sales
                    'sales': 'Sales', 'sale': 'Sales', 'salesperson': 'Sales',
                    'retail': 'Sales', 'cashier': 'Sales', 'sales rep': 'Sales',
                    'sales representative': 'Sales', 'seller': 'Sales',
                    
                    # Executive managerial
                    'manager': 'Exec-managerial', 'management': 'Exec-managerial',
                    'executive': 'Exec-managerial', 'exec': 'Exec-managerial',
                    'managerial': 'Exec-managerial', 'supervisor': 'Exec-managerial',
                    'director': 'Exec-managerial', 'ceo': 'Exec-managerial',
                    'president': 'Exec-managerial', 'vp': 'Exec-managerial',
                    'vice president': 'Exec-managerial', 'administrator': 'Exec-managerial',
                    'boss': 'Exec-managerial', 'leader': 'Exec-managerial',
                    
                    # Professional specialty
                    'professional': 'Prof-specialty', 'prof specialty': 'Prof-specialty',
                    'specialist': 'Prof-specialty', 'professional specialty': 'Prof-specialty',
                    'engineer': 'Prof-specialty', 'doctor': 'Prof-specialty',
                    'lawyer': 'Prof-specialty', 'accountant': 'Prof-specialty',
                    'scientist': 'Prof-specialty', 'researcher': 'Prof-specialty',
                    'teacher': 'Prof-specialty', 'professor': 'Prof-specialty',
                    'architect': 'Prof-specialty', 'analyst': 'Prof-specialty',
                    
                    # Handlers cleaners
                    'handler': 'Handlers-cleaners', 'cleaner': 'Handlers-cleaners',
                    'handlers cleaners': 'Handlers-cleaners', 'janitor': 'Handlers-cleaners',
                    'custodian': 'Handlers-cleaners', 'cleaning': 'Handlers-cleaners',
                    
                    # Machine operator inspector
                    'machine operator': 'Machine-op-inspct', 'operator': 'Machine-op-inspct',
                    'inspector': 'Machine-op-inspct', 'machine op': 'Machine-op-inspct',
                    'factory worker': 'Machine-op-inspct', 'assembly': 'Machine-op-inspct',
                    
                    # Administrative clerical
                    'admin': 'Adm-clerical', 'administrative': 'Adm-clerical',
                    'clerical': 'Adm-clerical', 'adm clerical': 'Adm-clerical',
                    'office': 'Adm-clerical', 'secretary': 'Adm-clerical',
                    'receptionist': 'Adm-clerical', 'clerk': 'Adm-clerical',
                    'office worker': 'Adm-clerical', 'assistant': 'Adm-clerical',
                    
                    # Farming fishing
                    'farming': 'Farming-fishing', 'fishing': 'Farming-fishing',
                    'farmer': 'Farming-fishing', 'agriculture': 'Farming-fishing',
                    'agricultural': 'Farming-fishing', 'farm': 'Farming-fishing',
                    'fisherman': 'Farming-fishing',
                    
                    # Transport moving
                    'transport': 'Transport-moving', 'transportation': 'Transport-moving',
                    'driver': 'Transport-moving', 'mover': 'Transport-moving',
                    'truck driver': 'Transport-moving', 'delivery': 'Transport-moving',
                    'courier': 'Transport-moving', 'logistics': 'Transport-moving',
                    
                    # Private house service
                    'private house': 'Priv-house-serv', 'house service': 'Priv-house-serv',
                    'domestic': 'Priv-house-serv', 'housekeeper': 'Priv-house-serv',
                    'maid': 'Priv-house-serv', 'nanny': 'Priv-house-serv',
                    
                    # Protective service
                    'protective': 'Protective-serv', 'protective service': 'Protective-serv',
                    'police': 'Protective-serv', 'security': 'Protective-serv',
                    'guard': 'Protective-serv', 'firefighter': 'Protective-serv',
                    'officer': 'Protective-serv', 'cop': 'Protective-serv',
                    
                    # Armed forces
                    'armed forces': 'Armed-Forces', 'military': 'Armed-Forces',
                    'army': 'Armed-Forces', 'navy': 'Armed-Forces',
                    'air force': 'Armed-Forces', 'marines': 'Armed-Forces',
                    'soldier': 'Armed-Forces', 'service member': 'Armed-Forces',
                    
                    # Unknown
                    'unknown': '?', 'not sure': '?', 'prefer not to say': '?',
                }
                
                val_lower = val_norm.lower()
                if val_lower in occupation_mapping:
                    return {'valid': True, 'message': '', 'normalized': occupation_mapping[val_lower]}
            
            # Sex: Gender mappings
            if field == 'sex':
                sex_mapping = {
                    # Male
                    'male': 'Male', 'm': 'Male', 'man': 'Male', 'boy': 'Male',
                    'guy': 'Male', 'gentleman': 'Male', 'he': 'Male',
                    
                    # Female
                    'female': 'Female', 'f': 'Female', 'woman': 'Female',
                    'girl': 'Female', 'lady': 'Female', 'she': 'Female',
                }
                
                val_lower = val_norm.lower()
                if val_lower in sex_mapping:
                    return {'valid': True, 'message': '', 'normalized': sex_mapping[val_lower]}
            
            # Race: Ethnicity mappings
            if field == 'race':
                race_mapping = {
                    # White
                    'white': 'White', 'caucasian': 'White', 'european': 'White',
                    
                    # Black
                    'black': 'Black', 'african american': 'Black', 'african': 'Black',
                    'afro american': 'Black',
                    
                    # Asian Pacific Islander
                    'asian': 'Asian-Pac-Islander', 'asian pacific islander': 'Asian-Pac-Islander',
                    'asian pac islander': 'Asian-Pac-Islander', 'pacific islander': 'Asian-Pac-Islander',
                    'filipino': 'Asian-Pac-Islander', 'chinese': 'Asian-Pac-Islander',
                    'japanese': 'Asian-Pac-Islander', 'korean': 'Asian-Pac-Islander',
                    'vietnamese': 'Asian-Pac-Islander', 'hawaiian': 'Asian-Pac-Islander',
                    
                    # American Indian Eskimo
                    'american indian': 'Amer-Indian-Eskimo', 'native american': 'Amer-Indian-Eskimo',
                    'indigenous': 'Amer-Indian-Eskimo', 'eskimo': 'Amer-Indian-Eskimo',
                    'native': 'Amer-Indian-Eskimo', 'indian': 'Amer-Indian-Eskimo',
                    
                    # Other
                    'other': 'Other', 'mixed': 'Other', 'multiracial': 'Other',
                    'hispanic': 'Other', 'latino': 'Other', 'latina': 'Other',
                }
                
                val_lower = val_norm.lower()
                if val_lower in race_mapping:
                    return {'valid': True, 'message': '', 'normalized': race_mapping[val_lower]}
            
            # Relationship: Family relationship mappings
            if field == 'relationship':
                relationship_mapping = {
                    # Wife
                    'wife': 'Wife', 'married woman': 'Wife', 'spouse female': 'Wife',
                    
                    # Husband
                    'husband': 'Husband', 'married man': 'Husband', 'spouse male': 'Husband',
                    
                    # Own child
                    'own child': 'Own-child', 'child': 'Own-child', 'son': 'Own-child',
                    'daughter': 'Own-child', 'kid': 'Own-child', 'offspring': 'Own-child',
                    
                    # Not in family
                    'not in family': 'Not-in-family', 'alone': 'Not-in-family',
                    'independent': 'Not-in-family', 'single person': 'Not-in-family',
                    'living alone': 'Not-in-family',
                    
                    # Other relative
                    'other relative': 'Other-relative', 'relative': 'Other-relative',
                    'cousin': 'Other-relative', 'aunt': 'Other-relative',
                    'uncle': 'Other-relative', 'niece': 'Other-relative',
                    'nephew': 'Other-relative', 'grandparent': 'Other-relative',
                    'grandchild': 'Other-relative',
                    
                    # Unmarried
                    'unmarried': 'Unmarried', 'partner': 'Unmarried',
                    'boyfriend': 'Unmarried', 'girlfriend': 'Unmarried',
                    'domestic partner': 'Unmarried', 'roommate': 'Unmarried',
                }
                
                val_lower = val_norm.lower()
                if val_lower in relationship_mapping:
                    return {'valid': True, 'message': '', 'normalized': relationship_mapping[val_lower]}
            
            # Native Country: Country name mappings
            if field == 'native_country':
                country_mapping = {
                    # United States variations
                    'united states': 'United-States', 'usa': 'United-States', 'us': 'United-States',
                    'u.s.': 'United-States', 'u.s.a.': 'United-States', 'america': 'United-States',
                    'united states of america': 'United-States', 'the us': 'United-States',
                    
                    # Country spelling variations
                    'colombia': 'Columbia', 'columbia': 'Columbia',
                    'netherlands': 'Holand-Netherlands', 'holland': 'Holand-Netherlands',
                    'hong kong': 'Hong', 'hongkong': 'Hong',
                    'south korea': 'South', 'korea': 'South', 's korea': 'South',
                    'trinidad': 'Trinadad&Tobago', 'trinidad and tobago': 'Trinadad&Tobago',
                    'el salvador': 'El-Salvador', 'salvador': 'El-Salvador',
                    'dominican republic': 'Dominican-Republic', 'dominican': 'Dominican-Republic',
                    'puerto rico': 'Puerto-Rico', 'puertorico': 'Puerto-Rico',
                    'us territory': 'Outlying-US(Guam-USVI-etc)', 'guam': 'Outlying-US(Guam-USVI-etc)',
                    'virgin islands': 'Outlying-US(Guam-USVI-etc)', 'usvi': 'Outlying-US(Guam-USVI-etc)',
                    'former yugoslavia': 'Yugoslavia', 'yugoslavia': 'Yugoslavia',
                    
                    # Full country names
                    'cambodia': 'Cambodia', 'canada': 'Canada', 'china': 'China',
                    'cuba': 'Cuba', 'ecuador': 'Ecuador', 'england': 'England',
                    'france': 'France', 'germany': 'Germany', 'greece': 'Greece',
                    'guatemala': 'Guatemala', 'haiti': 'Haiti', 'honduras': 'Honduras',
                    'hungary': 'Hungary', 'india': 'India', 'iran': 'Iran',
                    'ireland': 'Ireland', 'italy': 'Italy', 'jamaica': 'Jamaica',
                    'japan': 'Japan', 'laos': 'Laos', 'mexico': 'Mexico',
                    'nicaragua': 'Nicaragua', 'peru': 'Peru', 'philippines': 'Philippines',
                    'poland': 'Poland', 'portugal': 'Portugal', 'scotland': 'Scotland',
                    'taiwan': 'Taiwan', 'thailand': 'Thailand', 'vietnam': 'Vietnam',
                    
                    # Unknown
                    'unknown': '?', 'not sure': '?', 'prefer not to say': '?',
                }
                
                val_lower = val_norm.lower()
                if val_lower in country_mapping:
                    return {'valid': True, 'message': '', 'normalized': country_mapping[val_lower]}
                
                # Try partial match for countries
                for friendly, technical in country_mapping.items():
                    if friendly in val_lower or val_lower in friendly:
                        return {'valid': True, 'message': '', 'normalized': technical}
            # === FALLBACK: FUZZY MATCHING WITH DIFFLIB ===
            # If no exact mapping found, try fuzzy matching against allowed values
            suggestions = difflib.get_close_matches(val_norm, allowed, n=1, cutoff=0.7)
            if suggestions:
                # Found a close match - auto-normalize to it
                return {'valid': True, 'message': '', 'normalized': suggestions[0]}
            
            # Case-insensitive fuzzy match (try lowercase comparison)
            suggestions_lower = difflib.get_close_matches(
                val_norm.lower(), 
                [opt.lower() for opt in allowed], 
                n=1, 
                cutoff=0.7
            )
            if suggestions_lower:
                # Find the original allowed value
                for opt in allowed:
                    if opt.lower() == suggestions_lower[0]:
                        return {'valid': True, 'message': '', 'normalized': opt}
            
            # Special handling for generic 'other'
            if val_norm.lower() in ['other', 'others', 'something else', 'none of the above']:
                if field == 'race' and 'Other' in allowed:
                    return {'valid': True, 'message': '', 'normalized': 'Other'}
                if field == 'occupation':
                    return {
                        'valid': False,
                        'message': "For occupation, the dataset has 'Other-service' but not plain 'Other'. Please choose 'Other-service' or a specific occupation."
                    }
                if field == 'relationship':
                    return {
                        'valid': False,
                        'message': "For relationship, the dataset has 'Other-relative' but not plain 'Other'. Please choose 'Other-relative' or a specific relationship."
                    }
            
            # Unknown '?' permitted only if present in allowed list
            if val_norm == '?' and '?' in allowed:
                return {'valid': True, 'message': '', 'normalized': '?'}
            
            # No match found - provide helpful suggestions
            close_matches = difflib.get_close_matches(val_norm, allowed, n=3, cutoff=0.5)
            if close_matches:
                return {
                    'valid': False,
                    'message': f"'{value}' isn't a valid {self._pretty_field(field)}. Did you mean: {', '.join(close_matches)}?",
                    'allowed': allowed
                }
            
            return {
                'valid': False,
                'message': f"'{value}' isn't a valid {self._pretty_field(field)}.",
                'allowed': allowed
            }
        
        # Handle "Other" category and "?" (unknown/missing) values with accurate dataset information
        if value == 'Other' or value == '?' or value == 'Unknown/Prefer not to say':
            # Fields that actually DO have "Other" categories in the Adult dataset
            fields_with_other = ['race']  # race has "Other", occupation has "Other-service", relationship has "Other-relative"
            
            # Fields that support "?" (unknown/missing values)
            fields_with_unknown = ['workclass', 'occupation', 'native_country']
            
            # Fields that do NOT have "Other" or "?" categories
            fields_without_other_or_unknown = ['education', 'marital_status', 'sex', 'relationship']
            
            # Handle "?" values for fields that support them
            if value in ['?', 'Unknown/Prefer not to say'] and field in fields_with_unknown:
                return {'valid': True, 'message': ''}  # Accept "?" for these fields
            
            # Handle "Other" requests for fields that don't support either
            if field in fields_without_other_or_unknown:
                if field == 'sex':
                    return {
                        'valid': False,
                        'message': f"I understand that gender identity is more diverse than the binary options available. Unfortunately, the Adult (Census Income) dataset that trained this model only contains 'Male' and 'Female' categories. We are actively working on accommodating more inclusive gender categories in future versions. For now, could you please select the option that best represents you, or choose the one you're most comfortable with for this application?"
                    }
                else:
                    if value in ['?', 'Unknown/Prefer not to say']:
                        return {
                            'valid': False,
                            'message': f"I understand you may prefer not to specify, but this field doesn't support 'unknown' values in the dataset. Could you select the closest option for {self._pretty_field(field)}?"
                        }
                    else:
                        return {
                            'valid': False,
                            'message': f"The dataset doesn't include 'Other' as a category for {self._pretty_field(field)}. Please choose from the available options."
                        }
            elif field == 'occupation':
                # Occupation has "Other-service" but not plain "Other"
                return {
                    'valid': False,
                    'message': "For occupation, please choose 'Other-service' or a specific occupation from the list."
                }
            elif field == 'relationship':
                # Relationship has "Other-relative" but not plain "Other"  
                return {
                    'valid': False,
                    'message': "For relationship, please choose 'Other-relative' or one of the listed relationships."
                }
        
        return {'valid': True, 'message': '', 'normalized': value}

    def _setup_agent_instance(self):
        """Set up the agent's current instance using the user's application data"""
        try:
            # Convert application to the format expected by the agent
            app_data = self.application.to_dict()
            
            # Add missing required fields with defaults
            if 'fnlwgt' not in app_data or app_data['fnlwgt'] is None:
                app_data['fnlwgt'] = 100000
            
            # Create a DataFrame row representing this user's data
            app_df = pd.DataFrame([app_data])
            
            # Preprocess it the same way the training data was preprocessed
            app_df['income'] = '>50K' if self.application.loan_approved else '<=50K'
            from preprocessing import preprocess_adult
            app_processed = preprocess_adult(app_df)
            
            # Set this as the agent's current instance
            self.agent.current_instance = app_processed.drop('income', axis=1).iloc[0]
            self.agent.predicted_class = app_processed['income'].iloc[0]
            
            # CRITICAL: Pass the actual loan decision to agent for XAI explanations
            self.agent.loan_approved = self.application.loan_approved
            
        except Exception as e:
            # Fallback to random instance if user data setup fails
            print(f"Warning: Could not set up user instance, using random: {e}")
            self.agent.select_random_instance()

    def _get_next_question(self) -> str:
        """Get the next question to ask based on missing fields"""
        for field in self.field_order:
            if getattr(self.application, field) is None:
                self.current_field = field
                return self.field_prompts.get(field, f"Please provide your {field.replace('_', ' ')}:")
        
        self.current_field = None
        return ""

    def _show_progress(self) -> str:
        """Show current application progress with step information"""
        completion = self.application.calculate_completion()
        completed_fields = []
        missing_fields = []
        current_step = None
        
        for field in self.field_order:
            value = getattr(self.application, field)
            step_num = self.field_to_step.get(field, 0)
            step_desc = self.step_descriptions.get(step_num, field.replace('_', ' ').title())
            
            if value is not None:
                completed_fields.append(f"✅ **Step {step_num}:** {step_desc} - {value}")
            else:
                if current_step is None:  # First missing field is current step
                    current_step = step_num
                    missing_fields.append(f"🔄 **Step {step_num}:** {step_desc} *(Currently collecting)*")
                else:
                    missing_fields.append(f"⏳ **Step {step_num}:** {step_desc}")
        
        progress_msg = f"📊 **Application Progress: {completion:.0f}% Complete (Step {current_step or 10}/10)**\n\n"
        
        if completed_fields:
            progress_msg += "**✅ Completed Steps:**\n" + "\n".join(completed_fields) + "\n\n"
        
        if missing_fields:
            progress_msg += "**📋 Remaining Steps:**\n" + "\n".join(missing_fields) + "\n\n"
        
        if current_step:
            progress_msg += f"💡 **Next:** Continue with Step {current_step}\n\n"
        
        progress_msg += "Would you like to continue filling out the application?"
        return progress_msg

    def _show_application_summary(self) -> str:
        """Show complete application summary"""
        summary = "**Application Summary:**\n\n"
        
        for field in self.field_order:
            value = getattr(self.application, field)
            if value is not None:
                # Get friendly field name
                friendly_field = field.replace('_', ' ').title()
                
                # Format value based on field type
                if field in ['capital_gain', 'capital_loss']:
                    friendly_value = f"${value:,}" if isinstance(value, (int, float)) else str(value)
                elif field == 'hours_per_week':
                    friendly_value = f"{value} hours/week"
                elif field == 'age':
                    friendly_value = f"{value} years old"
                elif field in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'native_country']:
                    # For categorical values, try to get friendly name from FEATURE_DISPLAY_NAMES
                    friendly_value = get_friendly_feature_name(f"{field}_{value}")
                    # If no mapping found, use original value
                    if friendly_value.startswith(field.title()) or friendly_value == f"{field}_{value}":
                        friendly_value = str(value)
                else:
                    friendly_value = str(value)
                
                summary += f"• {friendly_field}: {friendly_value}\n"
        
        return summary

    def _process_application(self) -> str:
        """Process the loan application using the ML model"""
        try:
            # Convert application to DataFrame for model prediction
            app_data = self.application.to_dict()
            app_df = pd.DataFrame([app_data])
            
            # Add a dummy income column for preprocessing (required by preprocess_adult)
            app_df['income'] = '>50K'  # Dummy value, will be ignored
            
            # Ensure all required columns are present for preprocessing
            required_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                              'marital_status', 'occupation', 'relationship', 'race', 'sex',
                              'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
            
            # Add missing columns with sensible defaults
            if 'fnlwgt' not in app_df.columns:
                app_df['fnlwgt'] = 100000  # Default final weight
            
            if 'education_num' not in app_df.columns:
                # Map education level to education_num
                education_mapping = {
                    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
                    '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
                    'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
                    'Prof-school': 15, 'Doctorate': 16
                }
                education_value = app_df['education'].iloc[0] if 'education' in app_df.columns else 'HS-grad'
                app_df['education_num'] = education_mapping.get(education_value, 9)
            
            # Now preprocess the data to match the model's expected format
            from preprocessing import preprocess_adult
            app_df_processed = preprocess_adult(app_df)
            
            # Remove the dummy income column before prediction
            X_features = app_df_processed.drop('income', axis=1)
            
            # Check for and handle any remaining NaN values (failsafe)
            if X_features.isnull().any().any():
                X_features = X_features.fillna(0)
            
            # Ensure all values are finite (no inf values)
            if not np.isfinite(X_features.values).all():
                X_features = X_features.replace([np.inf, -np.inf], 0)
            
            # Make prediction using the agent's model
            if self.agent.clf_display is not None:
                # Ensure feature alignment with training data
                # Get the expected features from training data
                if hasattr(self.agent, 'data') and 'X_display' in self.agent.data:
                    # Get training features for reference
                    train_df = pd.concat([self.agent.data['X_display'], self.agent.data['y_display']], axis=1)
                    train_df_processed = preprocess_adult(train_df)
                    expected_features = train_df_processed.drop('income', axis=1).columns.tolist()
                    
                    # Align prediction features with training features
                    for col in expected_features:
                        if col not in X_features.columns:
                            X_features[col] = 0  # Add missing columns with default value
                    
                    # Reorder columns to match training order
                    X_features = X_features[expected_features]
                
                prediction = self.agent.clf_display.predict(X_features)[0]
                
                # Map prediction to loan decision
                approved = prediction in ['>50K', '1', 'true', 'True', 'Approved']
                decision = "APPROVED" if approved else "NOT APPROVED"
                
                # Update the application with the loan decision
                self.application.loan_approved = approved
                
                self.conversation_state = ConversationState.COMPLETE
                
                result_msg = f"🎉 **Loan Application Result: {decision}**\n\n"
                
                if approved:
                    result_msg += ("Congratulations! Your loan application has been approved based on your profile. "
                                 "You should hear from our lending team within 2-3 business days.\n\n")
                else:
                    # Simple message for non-explanation conditions
                    if config.explanation == "none":
                        result_msg += ("Based on the data you provided, you could not qualify for the loan at this time. "
                                     "For further details, please visit your local branch.\n\n")
                    else:
                        result_msg += ("Unfortunately, your loan application was not approved at this time. "
                                     "This decision is based on various factors in your application.\n\n")
                
                # Only offer explanations if explanation mode is enabled
                if config.show_any_explanation:
                    result_msg += ("Would you like me to explain how this decision was made? "
                                 "Just ask me 'why' or 'explain the decision'.")
                else:
                    result_msg += "Would you like to start a new application?"
                
                return result_msg
            else:
                return "I'm sorry, there was an issue processing your application. Please try again later."
                
        except Exception as e:
            return f"Sorry, there was an error processing your application: {str(e)}"

    def _generate_explanation(self) -> str:
        """Generate explanation using existing XAI methods"""
        try:
            # Set current instance for explanation
            self.agent.current_instance = self.application.to_dict()
            self.agent.df_display_instance = pd.DataFrame([self.application.to_dict()])
            
            # Use existing XAI explanation
            explanation = self.agent.handle_user_input("Why was this decision made?")
            return f"**Decision Explanation:**\n\n{explanation}"
            
        except Exception as e:
            return "I'm sorry, I couldn't generate an explanation right now. The decision was based on various factors in your application profile."

    def _restart_application(self) -> str:
        """Restart the application process"""
        self.application = LoanApplication()
        self.conversation_state = ConversationState.GREETING
        self.current_field = None
        self.field_attempts = {}
        self.show_what_if_lab = False
        return "Great! Let's start a new loan application. Hi! I'm your loan application assistant. Would you like to start your loan application?"

    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state for debugging/monitoring"""
        current_step = None
        if self.current_field and self.current_field in self.field_to_step:
            current_step = self.field_to_step[self.current_field]
        
        return {
            'state': self.conversation_state.value,
            'completion_percentage': self.application.calculate_completion(),
            'current_field': self.current_field,
            'current_step': current_step,
            'step_description': self.step_descriptions.get(current_step, '') if current_step else '',
            'is_complete': self.application.is_complete
        }
    
    def _get_progress_celebration(self, completion: float) -> str:
        """Return appropriate celebration message based on progress"""
        if config.show_anthropomorphic:
            # High anthropomorphism: Warm, encouraging
            if completion >= 75:
                return "We're almost done! 🎊"
            elif completion >= 50:
                return "Great progress! We're halfway there! 🚀"
            elif completion >= 25:
                return "Excellent! You're making great progress! 💪"
            else:
                return "Thank you so much! 😊"
        else:
            # Low anthropomorphism: Technical, factual
            if completion >= 75:
                return "Data collection 75% complete."
            elif completion >= 50:
                return "50% progress milestone reached."
            elif completion >= 25:
                return "25% data fields collected."
            else:
                return "Data received."
    
    def _get_success_message(self, field: str, normalized_value: str, completion: float, next_step: int = None) -> str:
        """Generate natural success message using LLM enhancement"""
        from ab_config import config
        
        # Build base message
        field_friendly = field.replace('_', ' ').title()
        celebration = self._get_progress_celebration(completion)
        
        if config.show_anthropomorphic:
            base_msg = f"Perfect! {celebration} I've recorded your {field_friendly}: {normalized_value}.\n\nProgress: {completion:.1f}% complete."
        else:
            base_msg = f"Data received. {field_friendly}: {normalized_value}. Progress: {completion:.1f}% complete."
        
        # Add next step info if available
        if next_step:
            step_desc = self.step_descriptions.get(next_step, '')
            if config.show_anthropomorphic:
                base_msg += f"\n\n📋 **Step {next_step}/10:** {step_desc}"
            else:
                base_msg += f"\n\nStep {next_step}/10: {step_desc}"
        
        # Enhance with LLM for natural conversation
        if NATURAL_CONVERSATION_AVAILABLE:
            try:
                context = {
                    'field': field,
                    'value': normalized_value,
                    'completion': completion,
                    'step': next_step
                }
                enhanced = enhance_response(
                    base_msg,
                    context,
                    "success_confirmation",
                    high_anthropomorphism=config.show_anthropomorphic
                )
                if enhanced and len(enhanced.strip()) > 10:
                    return enhanced
            except Exception:
                pass  # Fall back to base message
        
        return base_msg
    
    def _get_smart_validation_message(self, field: str, user_input: str, attempt: int) -> str:
        """Provide smart, context-aware validation messages"""
        from ab_config import config
        
        if config.show_anthropomorphic:
            # Try to get dynamic LLM-generated message first
            if NATURAL_CONVERSATION_AVAILABLE:
                try:
                    # Define expected format for each field
                    expected_formats = {
                        'age': "a number between 17-90, e.g., '25', '30', or '45'",
                        'workclass': "one of: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked, or '?'",
                        'education': "education level like 'HS-grad', 'Bachelors', 'Masters', 'Some-college', etc.",
                        'sex': "'Male' or 'Female'",
                        'marital_status': "one of: 'Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'",
                        'occupation': "a job category or '?' if uncertain",
                        'hours_per_week': "number of hours like '40', '35', or '50'",
                        'capital_gain': "amount like '5000' or '0' if none",
                        'capital_loss': "amount like '2000' or '0' if none",
                        'race': "one of: Black, Asian-Pac-Islander, Amer-Indian-Eskimo, White, or Other",
                        'native_country': "a country name from the 42 supported countries",
                        'relationship': "one of: Husband, Wife, Own-child, Not-in-family, Other-relative, Unmarried"
                    }
                    
                    expected_format = expected_formats.get(field, f"valid {field.replace('_', ' ')}")
                    llm_message = enhance_validation_message(field, user_input, expected_format, attempt, high_anthropomorphism=True)
                    
                    if llm_message:
                        return llm_message
                except Exception:
                    pass  # Fall back to hardcoded messages
            
            # Fallback: Hardcoded warm messages if LLM not available
            # High anthropomorphism: Warm, friendly, understanding
            field_examples = {
                'age': "I need a number for your age, please! 😊 For example: '25', '30', or '45'",
                'workclass': "No worries! Please choose from: Private sector, Self-employed, Federal/Local/State government, etc.\n\n💡 **Type '?' to see all valid options!**",
                'education': "I'd love to know your education level! Try: 'High school graduate', 'Bachelor's', 'Master's', 'Some college', etc.\n\n💡 **Type '?' to see all valid options you can copy-paste!** 📚",
                'sex': "Please let me know: 'Male' or 'Female' - these are the only categories in my dataset, I apologize for the limitation",
                'marital_status': "I need your marital status! Please choose: 'Married', 'Never married', 'Divorced', 'Separated', 'Widowed', etc.\n\n💡 **Type '?' for the full list!**",
                'occupation': "Tell me about your job! Please pick from the available categories.\n\n💡 **Type '?' to see all valid options you can copy-paste!** 💼",
                'hours_per_week': "How many hours do you work per week? Just give me a number like: '40', '35', or '50' ⏰",
                'capital_gain': "I need the amount of capital gains, or just '0' if you don't have any. For example: '5000' or '0'",
                'capital_loss': "Please tell me your capital losses, or '0' if none. For example: '2000' or '0'",
                'race': "Please help me understand your race/ethnicity.\n\n💡 **Type '?' to see all valid options!**",
                'native_country': "Which country are you from? I support all 42 countries in my training data! 🌍\n\n💡 **Type '?' to see the full list!**",
                'relationship': "What's your household relationship?\n\n💡 **Type '?' to see all valid options!**"
            }
            
            base_msg = f"Oops! I didn't quite catch that - '{user_input}' doesn't seem right for {field.replace('_', ' ')}. 🤔"
            help_msg = field_examples.get(field, f"Please share your {field.replace('_', ' ')} with me!")
            
            if attempt == 2:
                return f"{base_msg}\n\n💡 **Here's a tip:** {help_msg}\n\nLet's try again, or say 'help' if you need more options! 😊"
            else:
                return f"{base_msg}\n\n{help_msg}"
        else:
            # Low anthropomorphism: Technical, concise
            # Try to get dynamic LLM-generated message first
            if NATURAL_CONVERSATION_AVAILABLE:
                try:
                    # Define expected format for each field
                    expected_formats = {
                        'age': "a number between 17-90, e.g., '25', '30', or '45'",
                        'workclass': "one of: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked, or '?'",
                        'education': "education level like 'HS-grad', 'Bachelors', 'Masters', 'Some-college', etc.",
                        'sex': "'Male' or 'Female'",
                        'marital_status': "one of: 'Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'",
                        'occupation': "a job category or '?' if uncertain",
                        'hours_per_week': "number of hours like '40', '35', or '50'",
                        'capital_gain': "amount like '5000' or '0' if none",
                        'capital_loss': "amount like '2000' or '0' if none",
                        'race': "one of: Black, Asian-Pac-Islander, Amer-Indian-Eskimo, White, or Other",
                        'native_country': "a country name from the 42 supported countries",
                        'relationship': "one of: Husband, Wife, Own-child, Not-in-family, Other-relative, Unmarried"
                    }
                    
                    expected_format = expected_formats.get(field, f"valid {field.replace('_', ' ')}")
                    llm_message = enhance_validation_message(field, user_input, expected_format, attempt, high_anthropomorphism=False)
                    
                    if llm_message:
                        return llm_message
                except Exception:
                    pass  # Fall back to hardcoded messages
            
            # Fallback: Hardcoded technical messages if LLM not available
            field_examples = {
                'age': "I need a number for your age. For example: '25', '30', or '45'",
                'workclass': "Please choose from the available employment types.\n\nType '?' to see all valid options.",
                'education': "Please specify your education level.\n\nType '?' to see all valid options you can copy-paste.",
                'sex': "Please specify 'Male' or 'Female' - these are the only categories in the dataset I was trained on",
                'marital_status': "Please choose from the available marital status options.\n\nType '?' for the full list.",
                'occupation': "Please describe your job from the available categories.\n\nType '?' to see all valid options you can copy-paste.",
                'hours_per_week': "I need the number of hours you work per week. For example: '40', '35', or '50'",
                'capital_gain': "Enter the amount of capital gains, or '0' if none. For example: '5000' or '0'",
                'capital_loss': "Enter the amount of capital losses, or '0' if none. For example: '2000' or '0'",
                'race': "Please choose from the available race/ethnicity options.\n\nType '?' to see all valid options.",
                'native_country': "Please specify your country from the 42 supported countries.\n\nType '?' to see the full list.",
                'relationship': "Please choose from the available relationship options.\n\nType '?' for the full list."
            }
            
            base_msg = f"I didn't quite understand '{user_input}' for {field.replace('_', ' ')}."
            help_msg = field_examples.get(field, f"Please provide your {field.replace('_', ' ')}.")
            
            if attempt == 2:
                return f"{base_msg}\n\n💡 **Tip:** {help_msg}\n\nTry again, or say 'help' for more options."
            else:
                return f"{base_msg} {help_msg}"
    
    def _get_copyable_options(self, field: str) -> str:
        """Provide copyable list of valid options for categorical fields"""
        from ab_config import config
        
        if field not in self.allowed_values:
            # Numeric field - provide range info
            if field in self.validation_rules:
                rules = self.validation_rules[field]
                min_val = rules.get('min', 'any')
                max_val = rules.get('max', 'any')
                if config.show_anthropomorphic:
                    return f"📋 **{field.replace('_', ' ').title()}** is a number field.\n\nJust enter a value between **{min_val}** and **{max_val}**.\n\nFor example: `{(min_val + max_val) // 2 if isinstance(min_val, int) and isinstance(max_val, int) else '40'}`"
                else:
                    return f"**{field.replace('_', ' ').title()}:** Numeric field.\n\nRange: {min_val} to {max_val}\n\nExample: `{(min_val + max_val) // 2 if isinstance(min_val, int) and isinstance(max_val, int) else '40'}`"
        
        # Categorical field - provide full list
        options = self.allowed_values[field]
        
        if config.show_anthropomorphic:
            header = f"📋 **Here are all the valid options for {field.replace('_', ' ')}:**\n\n"
            header += "💡 **Tip:** You can copy-paste any of these exactly:\n\n"
        else:
            header = f"**Valid options for {field.replace('_', ' ')}:**\n\n"
            header += "Copy-paste one of these options:\n\n"
        
        # Format options in a clean, copyable list
        options_list = "\n".join([f"• `{opt}`" for opt in options])
        
        if config.show_anthropomorphic:
            footer = "\n\n✨ **Or just describe it naturally** - I can understand variations like 'private', 'bachelor', 'single', etc.!"
        else:
            footer = "\n\nNote: Natural language variations (e.g., 'private', 'bachelor', 'single') are also accepted."
        
        return header + options_list + footer
    
    def _get_field_help(self, field: str) -> str:
        """Provide detailed help for specific fields"""
        help_texts = {
            'age': "**Age Examples:**\n• Just enter a number: 25, 30, 45\n• Must be between 17-90 years old",
            'workclass': "**Employment Type (9 categories):**\n• Private sector (most common)\n• Self-employed\n• Federal/Local/State government\n• Without pay, Never worked\n• '?' if unknown/missing",
            'education': "**Education Level (16 categories):**\n• High school graduate, Bachelor's, Master's, Doctorate\n• Some college, Associate's degree\n• 1st-4th grade, 5th-6th grade, 7th-8th grade\n• 9th, 10th, 11th, 12th grade\n• Preschool, Professional school",
            'sex': "**Gender (2 categories only):**\n• Male\n• Female\n\n⚠️ This dataset was created in 1994 and only includes these binary options. We acknowledge this limitation and are working toward more inclusive data.",
            'marital_status': "**Marital Status (7 categories):**\n• Married\n• Never married\n• Divorced, Separated, Widowed\n• Married but spouse absent\n• Married to armed forces spouse",
            'occupation': "**Occupation (15 categories):**\n• Professional, Management, Tech support\n• Sales, Craft/repair, Administrative\n• Service, Farming/fishing\n• Handlers/cleaners, Machine operators\n• Transportation, Protective services\n• Private household service, Armed Forces\n• '?' if unknown",
            'race': "**Race/Ethnicity (5 categories):**\n• White, Black\n• Asian-Pacific Islander\n• Indigenous American\n• Other ✅",
            'native_country': "**Native Country (42 countries!):**\n• United States, Canada, Mexico\n• England, Scotland, Germany, France, Italy, Greece, Ireland, Portugal, Poland, Hungary, Netherlands\n• Philippines, India, China, Japan, Taiwan, Thailand, Vietnam, Cambodia, Laos, Hong Kong, South Korea\n• Jamaica, Haiti, Cuba, Puerto Rico, Dominican Republic, Trinidad & Tobago, Guatemala, Honduras, El Salvador, Nicaragua, Ecuador, Peru, Colombia\n• Iran, Former Yugoslavia\n• Use clickable buttons or type country name\n• '?' if unknown",
            'relationship': "**Household Relationship (6 categories):**\n• Husband, Wife\n• Own-child\n• Not in family\n• Other relative\n• Unmarried"
        }
        
        return help_texts.get(field, f"Please provide your {field.replace('_', ' ')} information.")