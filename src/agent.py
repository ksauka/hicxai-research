import logging
import json
import random
import re
import os
import pandas as pd
import shap
import sklearn
import pickle
from constraints import *
from nlu import NLU
import json
from answer import Answers
from utils import print_log

# Import natural conversation enhancer
try:
    from natural_conversation import enhance_response
    NATURAL_CONVERSATION_AVAILABLE = True
except ImportError:
    NATURAL_CONVERSATION_AVAILABLE = False
    def enhance_response(response, context=None, response_type="explanation"):
        return response

class Agent:
    def __init__(self, nlu_model=None):
        # Core state
        self.dataset = "adult"
        self.current_instance = None
        self.clf = None
        self.predicted_class = None
        self.mode = None
        self.data = {"X": None, "y": None, "features": None, "classes": None}

        # NLU setup: prefer provided model, else use config, else default
        config_path = os.path.join(os.path.dirname(__file__), 'nlu_config.json')
        if nlu_model is not None:
            self.nlu_model = nlu_model
        elif os.path.exists(config_path):
            with open(config_path, 'r') as f:
                nlu_config = json.load(f)
            self.nlu_model = NLU(model_type=nlu_config.get('model_type', 'simcse'), model_path=nlu_config.get('model_path'))
        else:
            self.nlu_model = NLU()

        # UI/state helpers
        self.list_node = []
        self.clf_display = None
        self.l_exist_classes = None
        self.l_exist_features = None
        self.l_instances = None
        self.df_display_instance = None
        self.current_feature = None
        self.preprocessor = None

        # Feature requirements for user input flows
        self.required_features = [
            'age', 'workclass', 'education', 'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain',
            'capital_loss', 'hours_per_week', 'native_country'
        ]
        self.user_features = {}

        # Load data and train model (sets self.clf and self.clf_display)
        self.load_adult_dataset()
        self.train_model()

    def load_adult_dataset(self):
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'adult.data')
        info_path = os.path.join(os.path.dirname(__file__), '..', 'dataset_info', 'adult.json')
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
            'hours_per_week', 'native_country', 'income'
        ]
        self.data['X_display'] = pd.read_csv(data_path, names=columns, skipinitialspace=True)
        self.data['y_display'] = self.data['X_display']['income']
        self.data['X_display'].drop(['income'], axis=1, inplace=True)
        with open(info_path, 'r') as f:
            self.data['info'] = json.load(f)
        self.data['classes'] = ['<=50K', '>50K']
        self.data['features'] = self.data['X_display'].columns.tolist()
        self.data['feature_names'] = self.data['features']
        self.data['map'] = {}

    def train_model(self):
        # Ensure model directory exists
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'RandomForest.pkl')
        if os.path.exists(model_path):
            self.clf = pickle.load(open(model_path, 'rb'))
            self.clf_display = self.clf
        else:
            from preprocessing import preprocess_adult
            df = pd.concat([self.data['X_display'], self.data['y_display']], axis=1)
            df_clean = preprocess_adult(df)
            X = df_clean.drop('income', axis=1)
            y = df_clean['income']
            from sklearn.ensemble import RandomForestClassifier
            self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.clf.fit(X, y)
            # Persist the trained model for faster subsequent runs
            with open(model_path, 'wb') as f:
                pickle.dump(self.clf, f)
            self.clf_display = self.clf

    # (Removed duplicate __init__; initialization handled above)

    def handle_user_input(self, user_input):
        """Handle user input for XAI explanations (used by loan assistant for explanations)"""
        # Step 1: Intent classification and XAI routing using enhanced NLU
        try:
            intent_result, confidence, suggestions = self.nlu_model.classify_intent(user_input)
            from constraints import SUGGEST_SIMILAR_QUESTIONS_MSG, REPHRASE_QUESTION_MSG
            
            # Route to appropriate XAI method based on intent
            if isinstance(intent_result, dict) and 'intent' in intent_result:
                # Ensure we have a current instance for explanation
                if self.current_instance is None:
                    self.select_random_instance()
                
                # Import the routing function
                try:
                    from xai_methods import route_to_xai_method
                    explanation_result = route_to_xai_method(self, intent_result)
                    base_explanation = explanation_result.get('explanation', 'Sorry, I could not generate an explanation.')
                    
                    # Enhance with natural conversation if available
                    if NATURAL_CONVERSATION_AVAILABLE:
                        context = {
                            'explanation_type': intent_result.get('intent', 'general'),
                            'user_question': user_input,
                            'confidence': intent_result.get('confidence', 0)
                        }
                        return enhance_response(base_explanation, context, "explanation")
                    
                    return base_explanation
                except ImportError:
                    # Fallback if routing function not available
                    base_explanation = self._generate_basic_explanation(intent_result)
                    
                    # Enhance fallback explanation too
                    if NATURAL_CONVERSATION_AVAILABLE:
                        context = {
                            'explanation_type': 'basic',
                            'user_question': user_input,
                            'confidence': 0.5
                        }
                        return enhance_response(base_explanation, context, "explanation")
                    
                    return base_explanation
                
            elif intent_result == 'unknown' and suggestions:
                suggestions_str = "\n".join([f"{idx}. {q}" for idx, q in enumerate(suggestions, 1)])
                return SUGGEST_SIMILAR_QUESTIONS_MSG.format(suggestions=suggestions_str)
            else:
                return REPHRASE_QUESTION_MSG
                
        except Exception as e:
            return f"I'm having trouble processing that question. Could you try asking it differently? Error: {str(e)}"
    
    def _generate_basic_explanation(self, intent_result):
        """Generate basic explanation when XAI methods are not available"""
        if self.current_instance is None or self.predicted_class is None:
            return "I need a specific instance to explain. Please make sure a prediction has been made."
        
        # Basic explanation based on the current instance
        explanation = f"Based on your profile, the decision was: {self.predicted_class}\n\n"
        explanation += "Key factors in this decision include:\n"
        
        # Highlight some key features
        key_features = ['age', 'education', 'hours_per_week', 'occupation', 'marital_status']
        for feature in key_features:
            if feature in self.current_instance:
                value = self.current_instance[feature]
                explanation += f"• {feature.replace('_', ' ').title()}: {value}\n"
        
        explanation += "\nThis is a simplified explanation. For more detailed analysis, specific XAI methods would provide deeper insights."
        return explanation
    
    def select_random_instance(self):
        """Select a random instance from the dataset for explanation"""
        if self.data.get('X_display') is not None and len(self.data['X_display']) > 0:
            random_idx = random.randint(0, len(self.data['X_display']) - 1)
            self.df_display_instance = self.data['X_display'].iloc[[random_idx]]
            self.current_instance = self.df_display_instance.iloc[0].to_dict()
            
            # Make prediction for this instance
            if self.clf_display is not None:
                self.predicted_class = self.clf_display.predict(self.df_display_instance)[0]

    def get_visualization(self, viz_type, instance_df=None):
        """
        Route advanced visualization requests to Answers class.
        viz_type: 'shap_advanced' or 'dtreeviz'
        instance_df: DataFrame for the instance to visualize
        """
        answers = Answers(
            list_node=self.list_node,
            clf=self.clf,
            clf_display=self.clf_display,
            current_instance=self.current_instance,
            question=None,
            l_exist_classes=self.l_exist_classes,
            l_exist_features=self.l_exist_features,
            l_instances=self.l_instances,
            data=self.data,
            df_display_instance=self.df_display_instance,
            predicted_class=self.predicted_class,
            preprocessor=self.preprocessor
        )
        return answers.answer(viz_type, instance_df=instance_df)

    def handle_user_input(self, user_input, instance_df=None):
        # Step 1: Refined feature extraction using regex and synonyms
        feature_synonyms = {
            'age': ['age', 'years old'],
            'workclass': ['workclass', 'work type', 'job type'],
            'education': ['education', 'degree'],
            'education_num': ['education num', 'education number', 'years of education'],
            'marital_status': ['marital status', 'married', 'single', 'relationship status'],
            'occupation': ['occupation', 'job', 'profession'],
            'relationship': ['relationship'],
            'race': ['race', 'ethnicity'],
            'sex': ['sex', 'gender'],
            'capital_gain': ['capital gain', 'gain'],
            'capital_loss': ['capital loss', 'loss'],
            'hours_per_week': ['hours per week', 'weekly hours', 'work hours'],
            'native_country': ['native country', 'country', 'nationality']
        }
        # Try to extract feature-value pairs from user input
        for feature, synonyms in feature_synonyms.items():
            for syn in synonyms:
                pattern = rf"{syn}[:=]?\s*([\w\-\+]+)"
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    self.user_features[feature] = match.group(1)
        # Check for missing features
        from constraints import CLARIFY_FEATURE_MSG
        missing = [f for f in self.required_features if f not in self.user_features]
        if missing:
            next_feat = missing[0]
            return CLARIFY_FEATURE_MSG.format(feature=next_feat.replace('_', ' '))
        # Step 2: Robust validation using adult dataset metadata
        from constraints import REPEAT_NUM_FEATURES, REPEAT_CAT_FEATURES
        info = self.data.get('info', {})
        for feature in self.required_features:
            value = self.user_features.get(feature)
            if value is None:
                continue
            # Numeric validation
            if feature in info.get('num_features', []):
                try:
                    val = float(value)
                    minv, maxv = info.get('feature_ranges', {}).get(feature, (None, None))
                    if minv is not None and (val < minv or val > maxv):
                        del self.user_features[feature]
                        return REPEAT_NUM_FEATURES.format(f"{minv}-{maxv}")
                except Exception:
                    del self.user_features[feature]
                    return REPEAT_NUM_FEATURES.format("valid number")
            # Categorical validation
            if feature in info.get('cat_features', []):
                valid = info.get('feature_values', {}).get(feature, [])
                if valid and value not in valid:
                    del self.user_features[feature]
                    return REPEAT_CAT_FEATURES.format(", ".join(valid))
        # Step 3: Intent classification and XAI routing using enhanced NLU
        intent_result, confidence, suggestions = self.nlu_model.classify_intent(user_input)
        from constraints import SUGGEST_SIMILAR_QUESTIONS_MSG, REPHRASE_QUESTION_MSG
        from xai_methods import route_to_xai_method
        # Route to appropriate XAI method based on intent
        if isinstance(intent_result, dict) and 'intent' in intent_result:
            if self.current_instance is None:
                self.select_random_instance()
            # Advanced visualization intents
            if intent_result['intent'] in ['shap_advanced', 'dtreeviz']:
                return self.get_visualization(intent_result['intent'], instance_df)
            # Standard explanation routing
            explanation_result = route_to_xai_method(self, intent_result)
            return explanation_result.get('explanation', 'Sorry, I could not generate an explanation.')
        elif intent_result == 'unknown' and suggestions:
            suggestions_str = "\n".join([f"{idx}. {q}" for idx, q in enumerate(suggestions, 1)])
            return SUGGEST_SIMILAR_QUESTIONS_MSG.format(suggestions=suggestions_str)
        else:
            return REPHRASE_QUESTION_MSG
