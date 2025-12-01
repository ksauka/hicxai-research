# Migrated and adapted from XAgent/Agent/answer.py for adult-only use
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from xai_methods import (
    explain_with_shap, explain_with_dice, explain_with_anchor,
    explain_with_shap_advanced, explain_with_dtreeviz
)
from constraints import *

class Answers:
    def __init__(self, list_node, clf, clf_display, current_instance, question, l_exist_classes, l_exist_features,
                 l_instances, data, df_display_instance, predicted_class, preprocessor=None):
        self.list_node = list_node
        self.clf = clf
        self.clf_display = clf_display
        self.question = question
        self.current_instance = current_instance
        self.l_exist_classes = l_exist_classes
        self.l_exist_features = l_exist_features
        self.l_instances = l_instances
        self.l_classes = data['classes']
        self.l_features = data['features']
        self.data = data
        self.df_display_instance = df_display_instance
        self.predicted_class = predicted_class
        self.preprocessor = preprocessor

    def answer(self, intent, conversations=[], instance_df=None, **kwargs):
        """
        Route to the correct XAI method based on dynamic intent/label from NLU.
        intent: predicted label from NLU (e.g., 'predict', 'shap_explain', 'dice_explain', 'anchor_explain', 'cf_proto', 'shap_advanced', 'dtreeviz')
        """
        if intent == 'predict':
            return f"Based on your input, the predicted income is {self.predicted_class}."
        elif intent == 'shap_explain':
            return explain_with_shap(self)
        elif intent == 'dice_explain':
            return explain_with_dice(self)
        elif intent == 'anchor_explain':
            return explain_with_anchor(self)
        elif intent == 'cf_proto':
            # CounterfactualProto (alibi) removed; optionally replace with dice-ml or handle gracefully
            return None
        elif intent == 'shap_advanced':
            if instance_df is not None:
                return explain_with_shap_advanced(self, instance_df)
            else:
                return {'type': 'error', 'explanation': 'No instance provided for SHAP advanced.'}
        elif intent == 'dtreeviz':
            if instance_df is not None:
                return explain_with_dtreeviz(self, instance_df)
            else:
                return {'type': 'error', 'explanation': 'No instance provided for dtreeviz.'}
        else:
            return "Sorry, I can't answer that question yet."
