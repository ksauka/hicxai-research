import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from agent import Agent

if __name__ == "__main__":
    a = Agent()
    print("NLU type:", a.nlu_model.model_type)
    print("Model trained:", a.clf is not None)
    # Simple route test
    from xai_methods import route_to_xai_method
    out = route_to_xai_method(a, {'intent':'shap','label':8,'matched_question':'feature importance'})
    print("SHAP result type:", out.get('type'))
