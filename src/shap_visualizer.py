"""
SHAP Visualization Component for XAI Explanations
Generates visual SHAP plots and explanations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import io
import base64

def create_shap_bar_plot(feature_impacts, prediction_class, title="Feature Importance Analysis"):
    """
    Create a SHAP-style bar plot showing feature impacts
    
    Args:
        feature_impacts: List of strings like "age increases the prediction probability by 0.150"
        prediction_class: The predicted class (e.g., ">50K" or "<=50K")
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    try:
        # Parse feature impacts
        features = []
        impacts = []
        
        for impact_str in feature_impacts:
            # Parse strings like "age increases the prediction probability by 0.150"
            parts = impact_str.split()
            if len(parts) >= 2:
                feature = parts[0]
                try:
                    # Find the numeric value
                    value = None
                    for part in parts:
                        try:
                            value = float(part)
                            break
                        except ValueError:
                            continue
                    
                    if value is not None:
                        # Determine if positive or negative impact
                        if "increases" in impact_str:
                            impacts.append(value)
                        elif "decreases" in impact_str:
                            impacts.append(-value)
                        else:
                            impacts.append(value)
                        features.append(feature.capitalize())
                except ValueError:
                    continue
        
        if not features:
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by absolute impact
        sorted_data = sorted(zip(features, impacts), key=lambda x: abs(x[1]), reverse=True)
        features_sorted, impacts_sorted = zip(*sorted_data)
        
        # Create colors: red for negative, blue for positive
        colors = ['red' if impact < 0 else 'blue' for impact in impacts_sorted]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(features_sorted)), impacts_sorted, color=colors, alpha=0.7)
        
        # Customize the plot
        ax.set_yticks(range(len(features_sorted)))
        ax.set_yticklabels(features_sorted)
        ax.set_xlabel('Impact on Prediction Probability')
        ax.set_title(f'{title}\nPrediction: {prediction_class}', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, impact) in enumerate(zip(bars, impacts_sorted)):
            width = bar.get_width()
            label_x = width + (0.01 if width >= 0 else -0.01)
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{impact:.3f}', ha='left' if width >= 0 else 'right', 
                   va='center', fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Increases Probability'),
            Patch(facecolor='red', alpha=0.7, label='Decreases Probability')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Style improvements
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating SHAP plot: {e}")
        return None

def create_shap_waterfall_plot(feature_impacts, base_probability=0.5, prediction_class="<=50K"):
    """
    Create a SHAP-style waterfall plot showing cumulative feature impacts
    """
    try:
        # Parse feature impacts
        features = []
        impacts = []
        
        for impact_str in feature_impacts:
            parts = impact_str.split()
            if len(parts) >= 2:
                feature = parts[0]
                try:
                    value = None
                    for part in parts:
                        try:
                            value = float(part)
                            break
                        except ValueError:
                            continue
                    
                    if value is not None:
                        if "decreases" in impact_str:
                            value = -value
                        features.append(feature.capitalize())
                        impacts.append(value)
                except ValueError:
                    continue
        
        if not features:
            return None
        
        # Create waterfall data
        cumulative = [base_probability]
        for impact in impacts:
            cumulative.append(cumulative[-1] + impact)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Draw the waterfall
        x_pos = range(len(features) + 2)
        colors = ['gray'] + ['red' if impact < 0 else 'blue' for impact in impacts] + ['green']
        
        # Base probability bar
        ax.bar(0, base_probability, color='gray', alpha=0.7, label='Base Probability')
        ax.text(0, base_probability/2, f'{base_probability:.3f}', ha='center', va='center', fontweight='bold')
        
        # Feature impact bars
        for i, (feature, impact, cum_val) in enumerate(zip(features, impacts, cumulative[1:-1])):
            start_height = cumulative[i]
            ax.bar(i+1, impact, bottom=start_height, 
                   color='red' if impact < 0 else 'blue', alpha=0.7)
            
            # Add connecting lines
            if i > 0:
                ax.plot([i, i+1], [cumulative[i], cumulative[i]], 'k--', alpha=0.5)
            
            # Add value label
            label_y = start_height + impact/2
            ax.text(i+1, label_y, f'{impact:+.3f}', ha='center', va='center', 
                   fontweight='bold', color='white')
        
        # Final prediction bar
        final_prob = cumulative[-1]
        ax.bar(len(features)+1, final_prob, color='green', alpha=0.7, label='Final Prediction')
        ax.text(len(features)+1, final_prob/2, f'{final_prob:.3f}', ha='center', va='center', fontweight='bold')
        
        # Customize plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Base'] + features + ['Final'], rotation=45, ha='right')
        ax.set_ylabel('Probability')
        ax.set_title(f'SHAP Waterfall Plot - Prediction: {prediction_class}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating waterfall plot: {e}")
        return None

def display_shap_explanation(explanation_result):
    """
    Display SHAP explanation with visualizations (only called when show_shap_visualizations=True)
    
    Args:
        explanation_result: Dict with SHAP explanation data
    """
    if explanation_result.get('type') != 'shap':
        return
    
    # Visual explanations - show plots
    if 'feature_impacts' in explanation_result and explanation_result['feature_impacts']:
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["📊 Feature Impact", "🌊 Waterfall Analysis"])
        
        with tab1:
            st.write("**How each feature affects the prediction:**")
            try:
                fig1 = create_shap_bar_plot(
                    explanation_result['feature_impacts'],
                    explanation_result.get('prediction_class', 'Unknown'),
                    "Feature Importance Analysis"
                )
                if fig1:
                    st.pyplot(fig1)
                    plt.close(fig1)  # Clean up memory
                else:
                    st.warning("Unable to generate feature impact chart")
            except Exception as e:
                st.error(f"Error creating feature impact chart: {str(e)}")
        
        with tab2:
            st.write("**Step-by-step impact on prediction probability:**")
            try:
                fig2 = create_shap_waterfall_plot(
                    explanation_result['feature_impacts'],
                    base_probability=0.5,
                    prediction_class=explanation_result.get('prediction_class', 'Unknown')
                )
                if fig2:
                    st.pyplot(fig2)
                    plt.close(fig2)  # Clean up memory
                else:
                    st.warning("Unable to generate waterfall chart")
            except Exception as e:
                st.error(f"Error creating waterfall chart: {str(e)}")
        
        # Feature impact breakdown
        st.write("### 📋 Detailed Feature Impacts")
        try:
            impacts_df = pd.DataFrame({
                'Feature Impact': explanation_result['feature_impacts']
            })
            st.dataframe(impacts_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying feature impacts table: {str(e)}")

def explain_shap_visualizations():
    """Provide educational content about SHAP visualizations"""
    with st.expander("ℹ️ Understanding SHAP Visualizations"):
        st.write("""
        **SHAP (SHapley Additive exPlanations)** helps you understand how each feature contributed to your prediction:
        
        **📊 Feature Impact Chart:**
        - **Blue bars** = Features that *increase* the likelihood of approval
        - **Red bars** = Features that *decrease* the likelihood of approval  
        - **Longer bars** = Stronger impact on the decision
        
        **🌊 Waterfall Analysis:**
        - Shows step-by-step how each feature moves the probability up or down
        - Starts with base probability and shows cumulative effect
        - Final bar shows the overall prediction probability
        
        **Why this matters:**
        - Understand *exactly* what factors influenced your decision
        - See which changes would have the biggest impact
        - Make informed decisions about improving your profile
        """)