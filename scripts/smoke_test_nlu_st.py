import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlu import NLU

if __name__ == "__main__":
    nlu = NLU(model_type="sentence_transformers")
    tests = [
        "What features affect income the most?",
        "Which variables impact salary?",
        "How should the instance be changed to get a different prediction?",
        "What is the minimum requirement for the prediction to stay the same?",
    ]
    for q in tests:
        result, conf, suggestions = nlu.classify_intent(q)
        print("Q:", q)
        print("→ intent:", result.get('intent'), "label:", result.get('label'), "conf:", round(conf, 3))
        print("→ matched:", result.get('matched_question'))
        print("→ suggestions[0..2]:", suggestions[:3])
        print("---")
