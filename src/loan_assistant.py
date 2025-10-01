"""
Loan Application Assistant - Multi-turn Conversational Agent
This module handles the conversational flow for collecting loan application information.
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

# Import natural conversation enhancer
try:
    from natural_conversation import enhance_response
    NATURAL_CONVERSATION_AVAILABLE = True
except ImportError:
    NATURAL_CONVERSATION_AVAILABLE = False
    def enhance_response(response, context=None, response_type="loan"):
        return response

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
        
        # Field collection order and prompts
        self.field_order = [
            'age', 'workclass', 'education', 'marital_status', 
            'occupation', 'hours_per_week', 'sex', 'race',
            'native_country', 'relationship', 'capital_gain', 'capital_loss'
        ]
        
        self.field_prompts = {
            'age': "What's your age?",
            'workclass': "What's your employment type? (e.g., Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)",
            'education': "What's your highest education level? (e.g., Bachelors, HS-grad, 11th, Masters, 9th, Some-college, Assoc-acdm, Assoc-voc, 7th-8th, Doctorate, Prof-school, 5th-6th, 10th, 1st-4th, Preschool, 12th)",
            'marital_status': "What's your marital status? (e.g., Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)",
            'occupation': "What's your occupation? (e.g., Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)",
            'hours_per_week': "How many hours per week do you work?",
            'sex': "What's your gender? (Male/Female)",
            'race': "What's your race? (e.g., White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)",
            'native_country': "What's your native country? (e.g., United-States, Cambodia, England, Puerto-Rico, Canada, Germany, etc.)",
            'relationship': "What's your relationship status? (e.g., Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)",
            'capital_gain': "Do you have any capital gains this year? (Enter amount or 0 if none)",
            'capital_loss': "Do you have any capital losses this year? (Enter amount or 0 if none)"
        }
        
        self.validation_rules = {
            'age': {'type': 'int', 'min': 17, 'max': 90},
            'hours_per_week': {'type': 'int', 'min': 1, 'max': 99},
            'capital_gain': {'type': 'int', 'min': 0, 'max': 99999},
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

    def handle_message(self, user_input: str) -> str:
        """Main message handler with enhanced natural conversation and XAI routing"""
        if not user_input.strip():
            return "I didn't catch that. Could you please say something?"
        
        # Check for XAI questions regardless of current state (if application is complete)
        user_lower = user_input.lower()
        xai_keywords = ['what if', 'why', 'explain', 'how', 'which factors', 'feature importance', 
                       'counterfactual', 'simple rules', 'anchor', 'what changes', 'what would happen']
        
        if (self.conversation_state == ConversationState.COMPLETE and 
            any(keyword in user_lower for keyword in xai_keywords)):
            # Route to explanation handler
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
        
        if any(keyword in user_input.lower() for keyword in greeting_keywords) or user_input.lower() in ['yes', 'y']:
            self.conversation_state = ConversationState.COLLECTING_INFO
            return ("Hello! I'm Luna, your personal loan application assistant. I will process your information and provide you with your loan qualification results. If you have any questions about the results, feel free to ask.\n\n"
                   "**I will collect information step by step** (not all at once) and you can **track your progress on the Progress Tracker** in the sidebar:\n"
                   "• **Step 1-2:** Personal Information (Age, Gender, etc.)\n"
                   "• **Step 3-5:** Employment Details (Work Class, Occupation, Hours)\n"
                   "• **Step 6:** Education Level\n"
                   "• **Step 7-8:** Financial Information (Capital Gains/Losses)\n"
                   "• **Step 9-10:** Background & Relationship Status\n\n"
                   "**Check the blue progress bar on the left to see your completion status!**\n\n"
                   "Let's start with **Step 1:**\n\n" + self._get_next_question())
        else:
            return ("Hi there! I'm Luna, your personal loan application assistant. I will process your information and provide you with your loan qualification results. If you have any questions about the results, feel free to ask.\n\n"
                   "**I will collect your information step by step** (not all at once) and you can **track your progress on the Progress Tracker** in the sidebar:\n"
                   "• **Step 1-2:** Personal Information (Age, Gender, etc.)\n"
                   "• **Step 3-5:** Employment Details (Work Class, Occupation, Hours)\n"
                   "• **Step 6:** Education Level\n"
                   "• **Step 7-8:** Financial Information (Capital Gains/Losses)\n"
                   "• **Step 9-10:** Background & Relationship Status\n\n"
                   "**Watch the blue progress bar fill up as we complete each step!**\n\n"
                   "Would you like to start your loan application? Just say 'yes' or 'start' to begin!")

    def _handle_info_collection(self, user_input: str) -> str:
        """Handle information collection phase"""
        if user_input.lower() in ['quit', 'exit', 'stop', 'cancel']:
            return "Application cancelled. Feel free to start again anytime by saying 'hi'!"
        
        if user_input.lower() in ['review', 'check', 'status']:
            return self._show_progress()
        
        if user_input.lower() in ['help', 'help me', 'what do i do', 'stuck', 'confused']:
            if self.current_field:
                return f"No problem! Let me help you with {self.current_field.replace('_', ' ')}:\n\n{self._get_field_help(self.current_field)}\n\n✨ You can also use the quick-select buttons if available!"
            else:
                return "I'm here to help! I'm collecting information for your loan application step by step. You can say 'review' to see your progress, or just answer the current question."
        
        # Process the current field
        if self.current_field:
            result = self._process_field_input(self.current_field, user_input)
            if result['success']:
                completion = self.application.calculate_completion()
                
                # Check if we have all required information
                if self.application.is_complete:
                    self.conversation_state = ConversationState.REVIEWING
                    return (f"🎉 Fantastic! I've collected all the necessary information ({completion:.0f}% complete).\n\n"
                           + self._show_application_summary() + 
                           "\n\n💼 Would you like me to process your loan application now? (yes/no)")
                else:
                    next_question = self._get_next_question()
                    if next_question:
                        # Get step information for the next field
                        next_step = self.field_to_step.get(self.current_field, 0) if self.current_field else 0
                        step_desc = self.step_descriptions.get(next_step, '')
                        # Progress celebrations at milestones
                        celebration = self._get_progress_celebration(completion)
                        return f"✅ Perfect! {celebration} Progress: {completion:.1f}% complete.\n\n📋 **Step {next_step}/10:** {step_desc}\n{next_question}"
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
        if user_input.lower() in ['explain', 'why', 'how', 'reason', 'details']:
            self.conversation_state = ConversationState.EXPLAINING
            return self._generate_explanation()
        elif user_input.lower() in ['new', 'another', 'restart', 'again']:
            return self._restart_application()
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
        elif user_lower in ['explain', 'why', 'how', 'reason', 'details', 'decision']:
            return self._generate_explanation()
        elif any(word in user_lower for word in ['what if', 'change', 'different']):
            return self._handle_explanation(user_input)
        else:
            # Check if the application has been processed
            if self.application.loan_approved is None:
                return ("It looks like your application hasn't been processed yet. Please complete the application first, "
                       "or if you believe this is an error, try saying 'start over' to begin a new application.")
            
            # Provide helpful guidance
            result = self.application.loan_approved
            status = "approved" if result else "denied"
            return (f"Your loan application has been {status}! Here's what you can do:\n\n"
                   "🔍 **Ask for explanations:** 'Why was I approved/denied?' or 'Explain the decision'\n"
                   "🔧 **What-if analysis:** 'What if my income was higher?' or 'What changes would help?'\n"
                   "📊 **Feature importance:** 'Which factors were most important?'\n"
                   "🆕 **New application:** 'Start a new application'\n\n"
                   "Just ask me anything about your loan decision!")

    def _handle_explanation(self, user_input: str) -> str:
        """Handle explanation requests using the XAgent-compatible approach"""
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
                    label = self.agent.nlu_model.df.query('Question == @matched_question')['Label'].iloc[0]
                    xai_method = self.agent.nlu_model.map_label_to_xai_method(label)
                    
                    print(f"🔍 DEBUG: Label: {label}, XAI method: {xai_method}")
                    
                    # Create intent result in the format expected by route_to_xai_method
                    intent_result = {
                        'intent': xai_method,
                        'label': label,
                        'matched_question': matched_question
                    }
                    
                    # Route to the correct XAI method
                    from xai_methods import route_to_xai_method
                    explanation_result = route_to_xai_method(self.agent, intent_result)
                    
                    # Debug logging
                    print(f"🔍 DEBUG: XAI result: {explanation_result}")
                    
                    explanation = explanation_result.get('explanation', 'Sorry, I could not generate an explanation.')
                    
                    # Store SHAP results for visualization (if available)
                    if (xai_method == 'shap' and 
                        isinstance(explanation_result, dict) and 
                        ('feature_impacts' in explanation_result or 'shap_values' in explanation_result)):
                        self.last_shap_result = explanation_result
                        print(f"🔍 DEBUG: Stored SHAP result for visualization")
                    else:
                        self.last_shap_result = None
                    
                    # Debug logging
                    print(f"🔍 DEBUG: Final explanation: {explanation}")
                    
                    context_msg = f"*I understood this as: \"{matched_question}\"*\n\n"
                    
                except Exception as e:
                    explanation = f"Sorry, I couldn't generate that explanation right now. Error: {str(e)}"
                    context_msg = ""
                
            else:
                # No good match found - fallback to general agent
                print(f"🔍 DEBUG: No good match found, falling back to general agent")
                explanation = self.agent.handle_user_input(user_input)
                context_msg = ""
            
            # Format the explanation nicely
            if isinstance(explanation, dict) and 'explanation' in explanation:
                formatted_explanation = explanation['explanation']
            elif isinstance(explanation, str):
                formatted_explanation = explanation
            else:
                formatted_explanation = str(explanation)
            
            return (f"{context_msg}**AI Explanation:**\n\n{formatted_explanation}\n\n"
                   "Would you like another explanation or start a new application? "
                   "You can ask questions like:\n"
                   "• 'What should change to get approved?' (counterfactual)\n"
                   "• 'Why was I denied/approved?' (feature importance)\n" 
                   "• 'What are the simple rules?' (rule-based explanations)\n"
                   "• 'New application' to start over")
            
        except Exception as e:
            return ("I'm sorry, I couldn't generate that explanation right now. "
                   "Would you like to try asking differently or start a new application? "
                   f"Error: {str(e)}")

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
                return {
                    'success': False,
                    'message': validation_result['message']
                }
            
            # Set the value
            setattr(self.application, field, value)
            self.field_attempts[field] = 0  # Reset attempts on success
            
            return {
                'success': True,
                'message': f"Got it! {field.replace('_', ' ').title()}: {value}"
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
                            'message': f"Please enter a number between {rules.get('min', 'any')} and {rules.get('max', 'any')}."
                        }
                except ValueError:
                    return {
                        'valid': False,
                        'message': f"Please enter a valid number for {field.replace('_', ' ')}."
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
                            'message': f"I understand you may prefer not to specify, but unfortunately the dataset requires specific values for {field.replace('_', ' ')}. This field doesn't support 'unknown' values. Could you please select the option that best fits your situation from the available choices?"
                        }
                    else:
                        return {
                            'valid': False,
                            'message': f"I'm sorry, but the Adult (Census Income) dataset that trained this model doesn't include 'Other' as a category for {field.replace('_', ' ')}. The historical dataset only contains specific predefined categories. Could you please select from the available options, or choose the closest match to your situation?"
                        }
            elif field == 'occupation':
                # Occupation has "Other-service" but not plain "Other"
                return {
                    'valid': False,
                    'message': f"For occupation, the dataset has 'Other-service' but not plain 'Other'. Would you like to select 'Other-service' or choose a more specific occupation from the available options?"
                }
            elif field == 'relationship':
                # Relationship has "Other-relative" but not plain "Other"  
                return {
                    'valid': False,
                    'message': f"For relationship, the dataset has 'Other-relative' but not plain 'Other'. Would you like to select 'Other-relative' or choose from the available relationship options?"
                }
        
        return {'valid': True, 'message': ''}

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
                summary += f"• {field.replace('_', ' ').title()}: {value}\n"
        
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
                    result_msg += ("Unfortunately, your loan application was not approved at this time. "
                                 "This decision is based on various factors in your application.\n\n")
                
                result_msg += ("Would you like me to explain how this decision was made? "
                             "Just ask me 'why' or 'explain the decision'.")
                
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
        if completion >= 75:
            return "We're almost done! 🎊"
        elif completion >= 50:
            return "Great progress! We're halfway there! 🚀"
        elif completion >= 25:
            return "Excellent! You're making great progress! 💪"
        else:
            return "Thank you so much!"
    
    def _get_smart_validation_message(self, field: str, user_input: str, attempt: int) -> str:
        """Provide smart, context-aware validation messages"""
        field_examples = {
            'age': "I need a number for your age. For example: '25', '30', or '45'",
            'workclass': "Please choose from: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked, or '?' if unknown",
            'education': "Please specify your education level like: 'HS-grad', 'Bachelors', 'Masters', 'Some-college', etc. Click the buttons below for all options!",
            'sex': "Please specify 'Male' or 'Female' - these are the only categories in the dataset I was trained on",
            'marital_status': "Please choose: 'Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', or 'Married-AF-spouse'",
            'occupation': "Please describe your job from the available categories, or '?' if uncertain. Click the buttons below for all options!",
            'hours_per_week': "I need the number of hours you work per week. For example: '40', '35', or '50'",
            'capital_gain': "Enter the amount of capital gains, or '0' if none. For example: '5000' or '0'",
            'capital_loss': "Enter the amount of capital losses, or '0' if none. For example: '2000' or '0'",
            'race': "Please choose from: White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, or Other",
            'native_country': "Please specify your country - I support all 42 countries in my training data! Click the buttons below or type the country name.",
            'relationship': "Please choose from: Husband, Wife, Own-child, Not-in-family, Other-relative, or Unmarried"
        }
        
        base_msg = f"I didn't quite understand '{user_input}' for {field.replace('_', ' ')}."
        help_msg = field_examples.get(field, f"Please provide your {field.replace('_', ' ')}.")
        
        if attempt == 2:
            return f"{base_msg}\n\n💡 **Tip:** {help_msg}\n\nTry again, or say 'help' for more options."
        else:
            return f"{base_msg} {help_msg}"
    
    def _get_field_help(self, field: str) -> str:
        """Provide detailed help for specific fields"""
        help_texts = {
            'age': "**Age Examples:**\n• Just enter a number: 25, 30, 45\n• Must be between 17-90 years old",
            'workclass': "**Employment Type (9 categories):**\n• Private (most common)\n• Self-emp-not-inc, Self-emp-inc\n• Federal-gov, Local-gov, State-gov\n• Without-pay, Never-worked\n• '?' if unknown/missing",
            'education': "**Education Level (16 categories):**\n• HS-grad, Bachelors, Masters, Doctorate\n• Some-college, Assoc-acdm, Assoc-voc\n• 1st-4th, 5th-6th, 7th-8th, 9th, 10th, 11th, 12th\n• Preschool, Prof-school",
            'sex': "**Gender (2 categories only):**\n• Male\n• Female\n\n⚠️ This dataset was created in 1994 and only includes these binary options. We acknowledge this limitation and are working toward more inclusive data.",
            'marital_status': "**Marital Status (7 categories):**\n• Married-civ-spouse (civilian spouse)\n• Never-married\n• Divorced, Separated, Widowed\n• Married-spouse-absent\n• Married-AF-spouse (armed forces spouse)",
            'occupation': "**Occupation (15 categories):**\n• Prof-specialty, Exec-managerial, Tech-support\n• Sales, Craft-repair, Adm-clerical\n• Other-service, Farming-fishing\n• Handlers-cleaners, Machine-op-inspct\n• Transport-moving, Protective-serv\n• Priv-house-serv, Armed-Forces\n• '?' if unknown",
            'race': "**Race/Ethnicity (5 categories):**\n• White, Black\n• Asian-Pac-Islander\n• Amer-Indian-Eskimo\n• Other ✅",
            'native_country': "**Native Country (42 countries!):**\n• United-States (most common)\n• All major countries supported\n• Use clickable buttons or type country name\n• '?' if unknown",
            'relationship': "**Household Relationship (6 categories):**\n• Husband, Wife\n• Own-child\n• Not-in-family\n• Other-relative\n• Unmarried"
        }
        
        return help_texts.get(field, f"Please provide your {field.replace('_', ' ')} information.")