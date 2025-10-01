

# Flexible, template-based constraint messages for dynamic, model-driven NLU

WELCOME_MSG = "Welcome to the HicXAI agent! Ask me about the model's predictions."
DATASET_ERROR_MSG = "I only support the adult dataset. Please type a correct name."
WAIT_MSG = "Wait a moment, I need to learn it."
RECORD_INFO_MSG = "I recorded the information: {}."
PREDICT_MSG = "You have: {}."
QUESTION_MSG = (
	"You can ask me questions about a machine learning model, such as: \n"
	"Why was the prediction made? \nWhy was Y not predicted? \n"
	"What should change in order to make prediction Y? \nPlease type your question."
)
REPHRASE_QUESTION_MSG = "Sorry, I don't understand your question. Please rephrase your question."
NO_CF_MSG = "Sorry, I couldn't find a way to modify {} to change the label."
CANT_ANSWER_MSG = "I am not capable of answering your question. Questions of this type can currently not be answered by an explainable AI method."
REPEAT_CAT_FEATURES = "The input value is not valid, please choose one of the following values: {}."
REPEAT_NUM_FEATURES = "The input value is not valid, please type a value in the range: {}."
REQUEST_NUMBER_MSG = "That is not a valid number. Please choose another number."

# Dynamic clarification/feedback templates (to be filled by agent/NLU at runtime)
CLARIFY_FEATURE_MSG = "What is your {feature}?"
CLARIFY_AMBIGUOUS_MSG = "I detected ambiguity in your input: {detail}. Could you clarify?"
SUGGEST_SIMILAR_QUESTIONS_MSG = (
	"I'm not sure I understood. Did you mean one of these?\n{suggestions}\nPlease type the number of the closest question, or rephrase your question."
)

# XAI method routing constants (adopted from XAgent)
L_SHAP_QUESTION_IDS = [3, 5, 6, 8, 26, 67, 69]
L_SHAP_QUESTION_FEATURE = [3, 5, 69]
L_SHAP_QUESTION_SINGLE_FEATURE = [6]
L_DICE_QUESTION_IDS = [11, 12, 14, 71]
L_DICE_QUESTION_RELATION_IDS = [71] 
L_ANCHOR_QUESTION_IDS = [20, 15, 13]
L_FEATURE_QUESTIONS_IDS = [6, 12]
L_NEW_PREDICT_QUESTION_IDS = [64]
L_SUPPORT_QUESTIONS_IDS = L_SHAP_QUESTION_IDS + L_DICE_QUESTION_IDS + L_ANCHOR_QUESTION_IDS

# Intent to XAI method mapping
INTENT_TO_XAI_METHOD = {
    "feature_importance": "shap",
    "counterfactual": "dice", 
    "local_explanation": "anchor",
    "prototype": "cfproto",
    "what_if": "interactive"
}

# Example usage in agent/NLU:
#   msg = CLARIFY_FEATURE_MSG.format(feature='age')
#   method = INTENT_TO_XAI_METHOD.get(intent, "unknown")
#   msg = CLARIFY_AMBIGUOUS_MSG.format(detail='multiple possible occupations')
#   msg = SUGGEST_SIMILAR_QUESTIONS_MSG.format(suggestions='1. ...\n2. ...')
