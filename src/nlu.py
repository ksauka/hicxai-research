
# NLU module for sentence-transformers-based semantic similarity and intent extraction
import pandas as pd
import os
import numpy as np
from constraints import L_SUPPORT_QUESTIONS_IDS, INTENT_TO_XAI_METHOD

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from simcse import SimCSE
    SIMCSE_AVAILABLE = True
except ImportError:
    SimCSE = None
    SIMCSE_AVAILABLE = False

class NLU:
    def __init__(self, model_type="sentence_transformers", model_path=None):
        self.model_type = model_type
        self.df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_questions', 'Median_4.csv'), index_col=0).drop_duplicates()
        self.questions = list(self.df['Question'])

        # Prefer sentence-transformers; use GPU if available, otherwise CPU (Streamlit Cloud has no GPU)
        if model_type == "sentence_transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                print("⚠️ sentence-transformers not available, trying SimCSE...")
                self.model_type = "simcse"
            else:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
                # Lightweight, fast model for semantic similarity
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                print(f"✅ Loaded sentence-transformers model on {device}")
                # Pre-compute embeddings for all questions
                self.question_embeddings = self.model.encode(self.questions, convert_to_numpy=True, show_progress_bar=False)
                print(f"✅ Pre-computed embeddings for {len(self.questions)} questions")

        # Optional SimCSE fallback for legacy envs
        if self.model_type == "simcse" or (model_type == "sentence_transformers" and not SENTENCE_TRANSFORMERS_AVAILABLE):
            if not SIMCSE_AVAILABLE:
                print("⚠️ SimCSE not available, falling back to simple keyword matching")
                self.model_type = "fallback"
                self.model = None
            else:
                self.model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
                self.model.build_index(self.questions)
                self.model_type = "simcse"

        elif model_type == "fallback":
            self.model = None
        elif self.model_type not in {"sentence_transformers", "simcse", "fallback"}:
            raise ValueError(f"Unsupported NLU model type: {model_type}. Supported: 'sentence_transformers', 'simcse', 'fallback'")

    def classify_intent(self, user_input, top_k=5):
        # Dynamic, model-driven intent extraction
        # Fast keyword heuristics ensure clear phrases immediately map to an XAI method
        try:
            text = (user_input or "").lower()
            # Heuristics for common phrasing
            rule_keywords = ["rule-based", "rule based", "rules", "conditions", "if then", "anchor"]
            shap_keywords = ["feature", "importance", "impact", "influence", "contribute", "shap", "why", "explain", "decision", "factors", "affected"]
            dice_keywords = ["what if", "counterfactual", "change", "modify", "different", "should i", "how to get"]
            if any(k in text for k in rule_keywords):
                return {
                    'intent': 'anchor',
                    'label': None,
                    'confidence': 0.95,
                    'matched_question': "Provide a simple rule-based explanation for this decision."
                }, 0.95, []
            if any(k in text for k in shap_keywords):
                return {
                    'intent': 'shap',
                    'label': None,
                    'confidence': 0.9,
                    'matched_question': "Which features were most important for this prediction?"
                }, 0.9, []
            if any(k in text for k in dice_keywords):
                return {
                    'intent': 'dice',
                    'label': None,
                    'confidence': 0.9,
                    'matched_question': "How should the instance be changed to get a different prediction?"
                }, 0.9, []
        except Exception:
            pass

        # sentence-transformers path
        if self.model_type == "sentence_transformers" and hasattr(self, 'question_embeddings'):
            try:
                query_emb = self.model.encode([user_input], convert_to_numpy=True, show_progress_bar=False)[0]
                # Cosine similarity
                q_norm = np.linalg.norm(self.question_embeddings, axis=1) + 1e-12
                u_norm = np.linalg.norm(query_emb) + 1e-12
                sims = (self.question_embeddings @ query_emb) / (q_norm * u_norm)
                # Top-k indices
                top_idx = np.argsort(-sims)[:top_k]
                match_question = self.questions[top_idx[0]]
                score = float(sims[top_idx[0]])
                label = self.df.iloc[top_idx[0]]['Label']
                xai_method = self.map_label_to_xai_method(label)
                suggestions = [self.questions[i] for i in top_idx]
                return {
                    'intent': xai_method,
                    'label': label,
                    'confidence': score,
                    'matched_question': match_question
                }, score, suggestions
            except Exception as e:
                print(f"sentence-transformers classify failed: {e}")

        # Legacy SimCSE path
        if self.model_type == "simcse" and self.model is not None:
            # Always get top matches without initial threshold filtering
            match_results = self.model.search(user_input, threshold=0, top_k=top_k)
            
            if len(match_results) > 0:
                match_question, score = match_results[0]
                
                # Get the label for the matched question
                label = self.df.query('Question == @match_question')['Label'].iloc[0]
                # Map label to XAI method if supported
                xai_method = self.map_label_to_xai_method(label)
                
                # Normalize confidence score to 0-1 range for consistency
                # SimCSE scores can be very high, so we'll use relative confidence
                normalized_confidence = min(1.0, score / 1e20) if score > 1 else score
                
                # Always return the best match but indicate confidence level
                return {
                    'intent': xai_method,
                    'label': label,
                    'confidence': normalized_confidence,
                    'matched_question': match_question
                }, normalized_confidence, []
        
        # Fallback to simple keyword matching when SimCSE is not available
        elif self.model_type == "fallback" or self.model is None:
            return self._fallback_classify_intent(user_input, top_k)
            
        # No matches found at all
            return 'unknown', 0.0, []
        else:
            return 'unknown', 0.0, []

    def match(self, user_input, features=None, prediction=None, current_instance=None, labels=None):
        """Hybrid approach: Fuzzy first (primary), Intent classifier fallback"""
        
        # PRIMARY: Try fuzzy matching first (fast and reliable)
        fuzzy_result = self._fuzzy_match_fallback(user_input)
        if fuzzy_result != "unknown":
            print(f"🔤 Fuzzy match (primary): {fuzzy_result}")
            return fuzzy_result
        
        # FALLBACK 1: Try intent classifier (65% accuracy)
        intent_result = self._classify_with_intent_classifier(user_input)
        if intent_result != "unknown":
            print(f"🧠 Intent classifier (fallback): {intent_result}")
            return intent_result
        
        # FALLBACK 2: Try embedding search if available (ST first, then SimCSE)
        if self.model_type == "sentence_transformers" and hasattr(self, 'question_embeddings'):
            try:
                query_emb = self.model.encode([user_input], convert_to_numpy=True, show_progress_bar=False)[0]
                q_norm = np.linalg.norm(self.question_embeddings, axis=1) + 1e-12
                u_norm = np.linalg.norm(query_emb) + 1e-12
                sims = (self.question_embeddings @ query_emb) / (q_norm * u_norm)
                best_idx = int(np.argmax(sims))
                match_question = self.questions[best_idx]
                print(f"🔍 ST match (last resort): {match_question}")
                return match_question
            except Exception as e:
                print(f"ST search failed: {e}")

        if hasattr(self, 'model') and self.model_type == "simcse" and self.model is not None:
            try:
                threshold = 0.6
                match_results = self.model.search(user_input, threshold=threshold)
                
                if len(match_results) > 0:
                    match_question, score = match_results[0]
                    print(f"🔍 SimCSE match (last resort): {match_question}")
                    return match_question
                else:
                    # Try with no threshold
                    match_results = self.model.search(user_input, threshold=0, top_k=5)
                    if len(match_results) > 0:
                        match_question, score = match_results[0]
                        print(f"🔍 SimCSE fallback: {match_question}")
                        return match_question
            except Exception as e:
                print(f"SimCSE search failed: {e}")
        
        print(f"❓ No match found for: '{user_input}'")
        return "unknown"
    
    def _fuzzy_match_fallback(self, user_input):
        """Fallback fuzzy matching using simple string similarity"""
        try:
            from difflib import SequenceMatcher
            
            user_lower = user_input.lower()
            best_match = None
            best_score = 0
            
            # Define key patterns for different XAI methods
            shap_patterns = [
                "feature", "important", "impact", "contribute", "influence", "matter", "weigh", "explain"
            ]
            dice_patterns = [
                "change", "different", "modify", "counterfact", "should", "what if", "approved", "denied"
            ]
            anchor_patterns = [
                "rule", "condition", "guarantee", "necessary", "sufficient", "always", "simple"
            ]
            
            # Check for pattern matches
            if any(pattern in user_lower for pattern in shap_patterns):
                # Return a representative SHAP question
                return "What features of this instance lead to the system's prediction?"
            elif any(pattern in user_lower for pattern in dice_patterns):
                # Return a representative DiCE question  
                return "How should the instance be changed to get a different (better or worse) prediction?"
            elif any(pattern in user_lower for pattern in anchor_patterns):
                # Return a representative Anchor question
                return "What is the minimum requirement for the prediction to stay the same?"
            
            # If no patterns match, try fuzzy string matching with dataset questions
            for _, row in self.df.iterrows():
                question = row['Question']
                similarity = SequenceMatcher(None, user_lower, question.lower()).ratio()
                if similarity > best_score:
                    best_score = similarity
                    best_match = question
            
            # Return best match if similarity is reasonable
            if best_score > 0.4:  # 40% similarity threshold
                return best_match
                
        except Exception as e:
            print(f"Fuzzy matching failed: {e}")
        
        return "unknown"
    
    def get_question_suggestions(self, match_results):
        """Extract question suggestions from match results"""
        suggestions = []
        for question, _ in match_results:
            if len(suggestions) < 5:  # Limit to 5 suggestions
                suggestions.append(question)
        return suggestions
    
    def map_label_to_xai_method(self, label):
        """Map question label to appropriate XAI method (adopted from XAgent logic)"""
        from constraints import L_SHAP_QUESTION_IDS, L_DICE_QUESTION_IDS, L_ANCHOR_QUESTION_IDS
        
        if label in L_SHAP_QUESTION_IDS:
            return "shap"
        elif label in L_DICE_QUESTION_IDS:
            return "dice"  
        elif label in L_ANCHOR_QUESTION_IDS:
            return "anchor"
        else:
            return "general"
    
    def replace_information(self, question, features=None, prediction=None, current_instance=None, labels=None):
        """Replace template variables in questions (adopted from XAgent)"""
        if features and "{X}" in question:
            feature_str = f"{{{features[0]},{features[1]}, ...}}" if len(features) > 1 else f"{features[0]}"
            question = question.replace("{X}", feature_str)
        if prediction and "{P}" in question:
            question = question.replace("{P}", str(prediction))
        if labels and prediction and "{Q}" in question:
            other_labels = [label for label in labels if str(label) != str(prediction)]
            question = question.replace("{Q}", str(other_labels))
        return question

    def _classify_with_intent_classifier(self, user_input):
        """Use the trained intent classifier (65% accuracy) as fallback"""
        try:
            # Try to load intent classifier if not already loaded
            if not hasattr(self, 'intent_classifier') or self.intent_classifier is None:
                self._load_intent_classifier()
            
            if self.intent_classifier is None:
                return "unknown"
            
            # Generate embedding for user input
            embedding = self.intent_simcse.encode([user_input])
            
            # Convert to tensor
            import torch
            import numpy as np
            embedding_tensor = torch.FloatTensor(embedding)
            
            # Get classifier prediction
            with torch.no_grad():
                outputs = self.intent_classifier(embedding_tensor)
                probabilities = outputs[0].numpy()
                
                # Get the class with highest probability
                predicted_class_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_class_idx]
                
                # Use lower threshold since this is fallback
                if confidence >= 0.3:  # Lower threshold for fallback
                    # Convert back to intent
                    predicted_intent = self.intent_label_encoder.inverse_transform([predicted_class_idx])[0]
                    
                    # Map intent to representative question
                    if predicted_intent == 'shap':
                        return "What features of this instance lead to the system's prediction?"
                    elif predicted_intent == 'dice':
                        return "How should the instance be changed to get a different (better or worse) prediction?"
                    elif predicted_intent == 'anchor':
                        return "What is the minimum requirement for the prediction to stay the same?"
                    # Don't return anything for 'other' - let it fall through
                
        except Exception as e:
            print(f"Intent classifier failed: {e}")
        
        return "unknown"
    
    def _load_intent_classifier(self):
        """Load the trained intent classifier (65% accuracy model)"""
        try:
            import torch
            import torch.nn as nn
            import pickle
            import numpy as np
            from simcse import SimCSE
            
            # Define the classifier architecture (matching the training script)
            class IntentClassifier(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_classes=4):
                    super(IntentClassifier, self).__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim // 2, num_classes),
                        nn.Softmax(dim=1)
                    )
                
                def forward(self, x):
                    return self.network(x)
            
            # Load metadata
            with open('models/intent_classifier_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            # Load label encoder
            with open('models/intent_label_encoder.pkl', 'rb') as f:
                self.intent_label_encoder = pickle.load(f)
            
            # Initialize and load classifier
            self.intent_classifier = IntentClassifier(
                metadata['input_dim'],
                metadata['hidden_dim'], 
                metadata['num_classes']
            )
            self.intent_classifier.load_state_dict(torch.load('models/intent_classifier_best.pth', map_location='cpu'))
            self.intent_classifier.eval()
            
            # Initialize SimCSE for embedding generation
            self.intent_simcse = SimCSE("princeton-nlp/sup-simcse-roberta-large")
            
            print(f"✅ Loaded intent classifier (accuracy: {metadata.get('best_accuracy', 'unknown'):.4f})")
            
        except Exception as e:
            print(f"⚠️ Could not load intent classifier: {e}")
            self.intent_classifier = None
            self.intent_label_encoder = None
            self.intent_simcse = None
