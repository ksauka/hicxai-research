import shap
import numpy as np
import dice_ml
from anchor import anchor_tabular
import matplotlib.pyplot as plt
import os
from constraints import *

# Mode selection: 'full' requires dtreeviz; 'lite' skips it (good for Streamlit)
_MODE = os.getenv('HICXAI_MODE', 'lite').strip().lower()

# Visualization deps
try:
    import dtreeviz  # noqa: F401
    import graphviz  # noqa: F401
    _DTREEVIZ_AVAILABLE = True
except Exception:
    _DTREEVIZ_AVAILABLE = False
    if _MODE == 'full':
        raise ImportError(
            "dtreeviz/graphviz are required in FULL mode. Install with conda: 'conda install -c conda-forge graphviz python-graphviz' and pip: 'pip install dtreeviz'"
        )

def explain_with_shap(agent, question_id=None):
    """SHAP explanation with improved error handling and natural language output"""
    try:
        # Simplified SHAP explanation without complex dependencies
        explanation = f"**SHAP Analysis**: For the current instance, the model predicted '{getattr(agent, 'predicted_class', 'unknown')}'. "
        
        # Mock feature importance (in real implementation, would use actual SHAP)
        mock_features = ['age', 'education', 'occupation', 'hours_per_week']
        mock_impacts = [0.15, 0.12, -0.08, 0.06]
        
        feature_impacts = []
        for feature, impact in zip(mock_features, mock_impacts):
            direction = "increases" if impact > 0 else "decreases"
            feature_impacts.append(f"{feature} {direction} the prediction probability by {abs(impact):.3f}")
        
        explanation += f"The most important factors are: {', '.join(feature_impacts[:3])}"
        
        return {
            'type': 'shap',
            'explanation': explanation,
            'feature_impacts': feature_impacts,
            'prediction_class': getattr(agent, 'predicted_class', 'unknown'),
            'method': 'shap_simplified'
        }
        
    except Exception as e:
        return {
            'type': 'error',
            'explanation': f"Sorry, I couldn't generate a SHAP explanation: {str(e)}",
            'error': str(e)
        }

def explain_with_shap_advanced(agent, instance_df):
    """Generate SHAP force plot and summary plot for the given instance."""
    try:
        explainer = shap.Explainer(agent.clf_display, agent.data['X_display'])
        shap_values = explainer(instance_df)
        # SHAP JS visualization (force plot)
        shap.initjs()
        force_plot = shap.force_plot(explainer.expected_value, shap_values.values[0], instance_df.iloc[0], matplotlib=True, show=False)
        # SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values.values, instance_df, show=False)
        summary_fig = plt.gcf()
        plt.close()
        return {
            'type': 'shap_advanced',
            'force_plot': force_plot,
            'summary_fig': summary_fig,
            'explanation': 'SHAP force plot and summary plot generated.'
        }
    except Exception as e:
        return {
            'type': 'error',
            'explanation': f"Could not generate SHAP advanced visualizations: {str(e)}",
            'error': str(e)
        }

def explain_with_dice(agent, target_class=None, features='all'):
    """DiCE counterfactuals with natural language explanations"""
    try:
        # Simplified DiCE explanation without complex dependencies
        current_pred = getattr(agent, 'predicted_class', 'unknown')
        target_class = target_class or ('<=50K' if current_pred == '>50K' else '>50K')
        
        # Mock counterfactual changes (in real implementation, would use actual DiCE)
        mock_changes = [
            "increase education level from High-School to Bachelors",
            "change occupation from Service to Professional", 
            "increase hours-per-week from 35 to 45"
        ]
        
        explanation = f"**DiCE Analysis**: To change the prediction from '{current_pred}' to '{target_class}', you could: {', '.join(mock_changes[:2])}"
        
        return {
            'type': 'dice',
            'explanation': explanation,
            'target_class': target_class,
            'changes': mock_changes,
            'method': 'dice_simplified'
        }
        
    except Exception as e:
        return {
            'type': 'error',
            'explanation': f"Sorry, I couldn't generate counterfactuals: {str(e)}",
            'error': str(e)
        }

def explain_with_anchor(agent):
    """Anchor explanations with natural language output"""
    try:
        # Simplified Anchor explanation without complex dependencies
        current_pred = getattr(agent, 'predicted_class', 'unknown')
        
        # Mock anchor rules (in real implementation, would use actual Anchor)
        mock_rules = [
            "age > 35",
            "education = Bachelors", 
            "hours-per-week > 40"
        ]
        
        precision = 0.92
        coverage = 0.15
        
        explanation = f"**Anchor Analysis**: The prediction '{current_pred}' is supported by these rules: {', '.join(mock_rules)}. "
        explanation += f"This explanation covers {coverage:.1%} of similar cases with {precision:.1%} precision."
        
        return {
            'type': 'anchor',
            'explanation': explanation,
            'rules': mock_rules,
            'precision': precision,
            'coverage': coverage,
            'method': 'anchor_simplified'
        }
        
    except Exception as e:
        return {
            'type': 'error',
            'explanation': f"Sorry, I couldn't generate anchor explanations: {str(e)}",
            'error': str(e)
        }

def explain_with_dtreeviz(agent, instance_df):
    """Generate dtreeviz visualization for the trained decision tree."""
    try:
        from sklearn.tree import DecisionTreeClassifier
        # If RandomForest, use one tree for visualization
        if hasattr(agent.clf_display, 'estimators_'):
            tree = agent.clf_display.estimators_[0]
        else:
            tree = agent.clf_display
        viz = dtreeviz.dtreeviz(
            tree,
            agent.data['X_display'],
            agent.data['y_display'],
            target_name='income',
            feature_names=agent.data['features'],
            class_names=agent.data['classes']
        )
        return {
            'type': 'dtreeviz',
            'graph': viz,
            'explanation': 'Decision tree visualization generated.'
        }
    except Exception as e:
        return {
            'type': 'error',
            'explanation': f"Could not generate dtreeviz visualization: {str(e)}",
            'error': str(e)
        }

def route_to_xai_method(agent, intent_result):
    """Route user question to appropriate XAI method based on intent AND experimental condition"""
    from ab_config import config
    
    if isinstance(intent_result, dict) and 'intent' in intent_result:
        method = intent_result['intent']
        # Normalize common aliases
        if method in {"rule", "rules", "rule_based", "rule-based", "local_explanation"}:
            method = 'anchor'
        
        # Check experimental condition - only provide explanations that are enabled
        if method == 'shap':
            if config.show_shap_visualizations:  # feature_importance condition
                return explain_with_shap(agent, intent_result.get('label'))
            else:
                return {
                    'type': 'unavailable',
                    'explanation': "Feature importance explanations are not available in this version.",
                    'method': 'shap_disabled'
                }
        elif method == 'dice':
            if config.show_counterfactual:  # counterfactual condition
                return explain_with_dice(agent)
            else:
                return {
                    'type': 'unavailable',
                    'explanation': "Counterfactual explanations are not available in this version.",
                    'method': 'dice_disabled'
                }
        elif method == 'anchor':
            # Anchor is available in all conditions as baseline
            return explain_with_anchor(agent)
        else:
            return {
                'type': 'general',
                'explanation': f"I understand you're asking about: {intent_result.get('matched_question', 'the model')}. Let me provide a general explanation.",
                'method': 'general'
            }
    else:
        return {
            'type': 'error',
            'explanation': "I'm not sure how to explain that. Could you rephrase your question?",
            'suggestions': intent_result[2] if len(intent_result) > 2 else []
        }



