import streamlit as st
from agent import Agent
from nlu import NLU
from answer import Answers
from github_saver import save_to_github
from loan_assistant import LoanAssistant
import os
import pandas as pd

# Streamlit compatibility function
def st_rerun():
    """Compatibility function for Streamlit rerun across versions"""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

# Configure page
st.set_page_config(page_title="AI Loan Assistant - Complete Solution", layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .progress-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 10px 0;
    }
    .progress-fill {
        background-color: #4caf50;
        height: 100%;
        transition: width 0.3s ease;
    }
    .main-tabs .stTabs [role="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize agent and components
@st.cache_resource
def initialize_system():
    """Initialize the agent and all components"""
    agent = Agent()
    answers = Answers(
        list_node=[],
        clf=agent.clf,
        clf_display=agent.clf_display,
        current_instance=agent.current_instance,
        question=None,
        l_exist_classes=agent.l_exist_classes,
        l_exist_features=agent.l_exist_features,
        l_instances=agent.l_instances,
        data=agent.data,
        df_display_instance=agent.df_display_instance,
        predicted_class=agent.predicted_class,
        preprocessor=agent.preprocessor
    )
    return agent, answers

agent, answers = initialize_system()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'loan_assistant' not in st.session_state:
    st.session_state['loan_assistant'] = LoanAssistant(agent)

if 'whatif_instance' not in st.session_state and hasattr(agent, 'data') and agent.data.get('X_display') is not None:
    st.session_state['whatif_instance'] = agent.data['X_display'].iloc[0].to_dict()

# App header
st.title("🏦 AI Loan Assistant - Complete Solution")
st.markdown("*Multi-turn conversational loan applications with advanced AI explanations*")

# Single conversational interface
st.markdown("---")
st.header("Conversational Loan Application")
st.markdown("Apply for a loan through natural conversation with AI-powered decisions and explanations.")

# Sidebar for application status
with st.sidebar:
        st.header("📊 Application Status")
        
        # Get current state info
        state_info = st.session_state.loan_assistant.get_conversation_state()
        completion = st.session_state.loan_assistant.application.calculate_completion()
        
        # Progress bar
        st.markdown("**Progress**")
        progress_html = f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {completion}%"></div>
        </div>
        <p style="text-align: center; margin: 5px 0;">{completion:.0f}% Complete</p>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        
        # Current state
        st.markdown(f"**Current State:** {state_info['state'].replace('_', ' ').title()}")
        
        if state_info['current_field']:
            st.markdown(f"**Collecting:** {state_info['current_field'].replace('_', ' ').title()}")
        
        # Quick actions
        st.markdown("---")
        st.markdown("**Quick Actions**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 Show Progress", key="sidebar_progress"):
                response = st.session_state.loan_assistant._show_progress()
                st.session_state.chat_history.append(("show progress", response))
                st_rerun()
        
        with col2:
            if st.button("🔄 Restart", key="sidebar_restart"):
                st.session_state.loan_assistant = LoanAssistant(agent)
                st.session_state.chat_history = []
                st_rerun()

# Chat interface
st.markdown("### 💬 Chat with Your Loan Assistant")

# Display chat history
chat_container = st.container()
with chat_container:
        for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
            # User message
            if user_msg:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {user_msg}
                </div>
                """, unsafe_allow_html=True)
            
            # Assistant message
            if assistant_msg:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong><br>
                    {assistant_msg}
                </div>
                """, unsafe_allow_html=True)
    
    # Welcome message for first-time visitors
    if len(st.session_state.chat_history) == 0:
        welcome_msg = st.session_state.loan_assistant.handle_message("hello")
        st.session_state.chat_history.append((None, welcome_msg))
        st_rerun()
    
    # Chat input
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message here...", 
            key="chat_user_input",
            placeholder="Type your message and press Enter or click Send..."
        )
    
    with col2:
        send_button = st.button("Send", type="primary", key="chat_send")
    
    # Handle user input
    if (send_button or user_input) and user_input.strip():
        # Add user message to history
        user_message = user_input.strip()
        
        # Determine if this should go to loan assistant or XAI agent
        loan_keywords = ['application', 'apply', 'loan', 'start', 'my age', 'my education', 'my job', 'hello', 'hi']
        is_loan_flow = any(keyword in user_input.lower() for keyword in loan_keywords)
        current_state = st.session_state.loan_assistant.conversation_state.value
        
        # Get assistant response
        with st.spinner("Processing..."):
            if is_loan_flow or current_state in ['greeting', 'collecting_info', 'reviewing']:
                # Use loan assistant for application flow
                assistant_response = st.session_state.loan_assistant.handle_message(user_message)
            else:
                # Use XAI agent for explanations
                if hasattr(st.session_state.loan_assistant.application, 'to_dict'):
                    app_dict = st.session_state.loan_assistant.application.to_dict()
                    agent.current_instance = app_dict
                    agent.df_display_instance = pd.DataFrame([app_dict])
                    if agent.clf_display:
                        agent.predicted_class = agent.clf_display.predict(agent.df_display_instance)[0]
                
                assistant_response = agent.handle_user_input(user_message)
        
        # Add to chat history
        st.session_state.chat_history.append((user_message, assistant_response))
        st_rerun()
    
    # Quick reply buttons based on current state
    st.markdown("---")
    st.markdown("**Quick Replies:**")
    
    current_state = st.session_state.loan_assistant.conversation_state.value
    
    if current_state == 'greeting':
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👋 Start Application", key="quick_start"):
                response = st.session_state.loan_assistant.handle_message("start")
                st.session_state.chat_history.append(("start", response))
                st_rerun()
    
    elif current_state == 'collecting_info':
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Check Progress", key="quick_progress"):
                response = st.session_state.loan_assistant.handle_message("review")
                st.session_state.chat_history.append(("check progress", response))
                st_rerun()
        with col2:
            if st.button("❓ Help", key="quick_help"):
                help_msg = ("I'm collecting information for your loan application. Please answer the questions "
                           "as accurately as possible. You can say 'review' to see your progress.")
                st.session_state.chat_history.append(("help", help_msg))
                st_rerun()
    
    elif current_state == 'complete':
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Explain Decision", key="quick_explain"):
                response = st.session_state.loan_assistant.handle_message("explain")
                st.session_state.chat_history.append(("explain", response))
                st_rerun()
        with col2:
            if st.button("🆕 New Application", key="quick_new"):
                response = st.session_state.loan_assistant.handle_message("new")
                st.session_state.chat_history.append(("new application", response))
                st_rerun()

# TAB 2: WHAT-IF ANALYSIS (Original functionality)
with tab2:
    st.header("Interactive What-If Analysis")
    st.markdown("Adjust parameters and see how they affect the loan decision in real-time.")
    
    # Load dataset info and current instance from agent
    if hasattr(agent, 'data') and agent.data.get('X_display') is not None:
        features = agent.data['features']
        X_display = agent.data['X_display']
        instance = st.session_state['whatif_instance']

        new_instance = {}
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Split features into two columns
        mid_point = len(features) // 2
        
        with col1:
            st.markdown("#### Personal & Employment")
            for feat in features[:mid_point]:
                col = X_display[feat]
                if str(col.dtype).startswith('float') or str(col.dtype).startswith('int'):
                    min_val = float(col.min())
                    max_val = float(col.max())
                    step = (max_val - min_val) / 100 if max_val > min_val else 1
                    val = st.slider(f"{feat}", min_value=min_val, max_value=max_val, 
                                  value=float(instance.get(feat, min_val)), step=step, key=f"slider_{feat}")
                    new_instance[feat] = val
                else:
                    options = list(col.unique())
                    val = st.selectbox(f"{feat}", options, 
                                     index=options.index(instance.get(feat, options[0])), key=f"select_{feat}")
                    new_instance[feat] = val
        
        with col2:
            st.markdown("#### Demographics & Financial")
            for feat in features[mid_point:]:
                col = X_display[feat]
                if str(col.dtype).startswith('float') or str(col.dtype).startswith('int'):
                    min_val = float(col.min())
                    max_val = float(col.max())
                    step = (max_val - min_val) / 100 if max_val > min_val else 1
                    val = st.slider(f"{feat}", min_value=min_val, max_value=max_val, 
                                  value=float(instance.get(feat, min_val)), step=step, key=f"slider2_{feat}")
                    new_instance[feat] = val
                else:
                    options = list(col.unique())
                    val = st.selectbox(f"{feat}", options, 
                                     index=options.index(instance.get(feat, options[0])), key=f"select2_{feat}")
                    new_instance[feat] = val
        
        st.session_state['whatif_instance'] = new_instance

        # Predict and explain with updated instance
        st.markdown("---")
        st.markdown("### 🎯 Updated Prediction and Explanation")
        
        instance_df = pd.DataFrame([new_instance])
        
        # Predict
        if hasattr(agent, 'clf_display') and agent.clf_display is not None:
            pred = agent.clf_display.predict(instance_df)[0]
            # Map underlying model class to a loan decision label for demo purposes
            pred_str = str(pred)
            approved = pred_str in {">50K", "1", "true", "True", "Approved"}
            decision = "✅ APPROVED" if approved else "❌ NOT APPROVED"
            
            # Display result prominently
            if approved:
                st.success(f"**Loan Decision:** {decision}")
            else:
                st.error(f"**Loan Decision:** {decision}")
            
            st.caption(f"(Model prediction: {pred_str})")
        
        # SHAP explanation (if available)
        st.markdown("#### 📊 SHAP Waterfall Explanation")
        try:
            import shap
            explainer = shap.Explainer(agent.clf_display, agent.data['X_display'])
            shap_values = explainer(instance_df)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(shap.plots.waterfall(shap_values[0], show=False))
        except Exception as e:
            st.info(f"SHAP explanation not available: {e}")

# TAB 3: ADVANCED EXPLANATIONS
with tab3:
    st.header("Advanced AI Explanations")
    st.markdown("Get detailed explanations using different AI methods: SHAP, DiCE, and Anchor.")
    
    # Use current whatif instance or loan application data
    if hasattr(st.session_state.loan_assistant.application, 'to_dict') and st.session_state.loan_assistant.application.calculate_completion() > 50:
        # Use loan application data if available
        app_dict = st.session_state.loan_assistant.application.to_dict()
        instance_df = pd.DataFrame([app_dict])
        st.info("Using your loan application data for explanations.")
    elif 'whatif_instance' in st.session_state:
        # Use what-if analysis data
        instance_df = pd.DataFrame([st.session_state['whatif_instance']])
        app_dict = st.session_state['whatif_instance']
        st.info("Using what-if analysis data for explanations.")
    else:
        st.warning("Please complete a loan application or use the What-If Analysis tab first.")
        app_dict = None
        instance_df = None
    
    if app_dict and instance_df is not None:
        # Show current prediction
        if agent.clf_display:
            pred = agent.clf_display.predict(instance_df)[0]
            approved = str(pred) in {">50K", "1", "true", "True", "Approved"}
            decision = "APPROVED" if approved else "NOT APPROVED"
            
            if approved:
                st.success(f"**Current Prediction:** {decision}")
            else:
                st.error(f"**Current Prediction:** {decision}")
        
        st.markdown("---")
        
        # Three columns for different explanation methods
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 SHAP Analysis")
            st.markdown("Shows which features most influenced your decision")
            if st.button("Generate SHAP Explanation", key="shap_btn"):
                try:
                    result = answers.answer('shap_advanced', instance_df=instance_df)
                    if isinstance(result, dict) and result.get('type') == 'shap_advanced':
                        if result.get('force_plot'):
                            st.pyplot(result.get('force_plot'))
                        if result.get('summary_fig'):
                            st.pyplot(result.get('summary_fig'))
                    else:
                        # Fallback explanation
                        st.markdown("**Key factors influencing your decision:**")
                        factors = []
                        if app_dict.get('age', 0) > 30:
                            factors.append("• Age shows financial maturity")
                        if app_dict.get('education') in ['Bachelors', 'Masters', 'Doctorate']:
                            factors.append("• Education level is favorable")
                        if app_dict.get('hours_per_week', 0) >= 35:
                            factors.append("• Work hours indicate stability")
                        
                        for factor in factors:
                            st.markdown(factor)
                except Exception as e:
                    st.error(f"SHAP analysis error: {e}")
        
        with col2:
            st.markdown("#### 🔄 DiCE Counterfactuals")
            st.markdown("Shows what changes could flip your decision")
            if st.button("Generate Counterfactuals", key="dice_btn"):
                try:
                    result = answers.answer('dice', instance_df=instance_df)
                    if isinstance(result, dict) and result.get('explanation'):
                        st.markdown(result.get('explanation'))
                    else:
                        # Fallback counterfactual
                        opposite = "approved" if not approved else "denied"
                        st.markdown(f"**To get {opposite}, consider:**")
                        st.markdown("• Increasing education level")
                        st.markdown("• Working more hours per week")
                        st.markdown("• Different occupation category")
                except Exception as e:
                    st.error(f"DiCE analysis error: {e}")
        
        with col3:
            st.markdown("#### ⚓ Anchor Rules")
            st.markdown("Shows simple rules that explain your decision")
            if st.button("Generate Anchor Rules", key="anchor_btn"):
                try:
                    result = answers.answer('anchor', instance_df=instance_df)
                    if isinstance(result, dict) and result.get('explanation'):
                        st.markdown(result.get('explanation'))
                    else:
                        # Fallback anchor rules
                        st.markdown("**Key decision rules:**")
                        if app_dict.get('age', 0) >= 25:
                            st.markdown(f"• Age ≥ 25 (yours: {app_dict.get('age')})")
                        if app_dict.get('hours_per_week', 0) >= 30:
                            st.markdown(f"• Hours ≥ 30/week (yours: {app_dict.get('hours_per_week')})")
                except Exception as e:
                    st.error(f"Anchor analysis error: {e}")
        
        # Advanced visualizations
        st.markdown("---")
        st.markdown("### 📈 Advanced Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Show Decision Tree Visualization", key="tree_viz"):
                result = answers.answer('dtreeviz', instance_df=instance_df)
                if isinstance(result, dict) and result.get('type') == 'dtreeviz':
                    graph_svg = result.get('graph', None)
                    if graph_svg is not None:
                        st.graphviz_chart(graph_svg.svg())
                    else:
                        st.error("No graph available.")
                else:
                    st.error("Decision tree visualization not available.")
        
        with col2:
            if st.button("Show Feature Importance Plot", key="feature_importance"):
                try:
                    # Simple feature importance visualization
                    if hasattr(agent.clf_display, 'feature_importances_'):
                        import matplotlib.pyplot as plt
                        importances = agent.clf_display.feature_importances_
                        features = agent.data['features']
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        indices = importances.argsort()[::-1][:10]  # Top 10 features
                        ax.bar(range(len(indices)), importances[indices])
                        ax.set_xticks(range(len(indices)))
                        ax.set_xticklabels([features[i] for i in indices], rotation=45)
                        ax.set_title('Top 10 Feature Importances')
                        st.pyplot(fig)
                    else:
                        st.info("Feature importance not available for this model type.")
                except Exception as e:
                    st.error(f"Feature importance error: {e}")

# TAB 4: FEEDBACK & ANALYTICS
with tab4:
    st.header("User Feedback & System Analytics")
    st.markdown("Help us improve the AI loan assistant by providing feedback.")
    
    # User feedback section (enhanced from original)
    st.markdown("### 💬 We'd love your feedback!")
    st.markdown("Help us make the loan assistant better. Share anything—from a confusing answer to a feature you wish existed.")

    col1, col2 = st.columns(2)
    with col1:
        fb_question = st.text_area(
            "What did you ask the assistant? (Optional)",
            placeholder="E.g. Why was I approved/denied?",
            key="fb_question",
            height=100
        )
        fb_actual = st.text_area(
            "What response did you get? (Optional)",
            placeholder="Paste the assistant's answer here if you want.",
            key="fb_actual",
            height=100
        )

    with col2:
        fb_expected = st.text_area(
            "What would have been a more helpful answer? (Optional)",
            placeholder="Describe what you expected or wanted instead.",
            key="fb_expected",
            height=100
        )
        fb_comments = st.text_area(
            "Anything else you'd like to tell us? (Optional)",
            placeholder="Suggestions, bugs, or general thoughts...",
            key="fb_comments",
            height=100
        )

    if st.button("📤 Send Feedback", type="primary") and (fb_question.strip() or fb_expected.strip() or fb_actual.strip() or fb_comments.strip()):
        import time
        feedback_structured = (
            f"User Question: {str(fb_question) if fb_question is not None else ''}\n"
            f"Assistant Response: {str(fb_actual) if fb_actual is not None else ''}\n"
            f"Better/Expected Answer: {str(fb_expected) if fb_expected is not None else ''}\n"
            f"Other Comments: {str(fb_comments) if fb_comments is not None else ''}\n"
            f"Application State: {st.session_state.loan_assistant.get_conversation_state()}\n"
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        success = save_to_github(
            repo="yourusername/yourrepo",
            path=f"feedback/comprehensive_app_{st.session_state.get('user_id', 'anon')}.txt",
            content=feedback_structured,
            commit_message="Comprehensive app user feedback",
            github_token=str(st.secrets.get("GITHUB_TOKEN", ""))
        )
        if success:
            st.success("✅ Thank you! Your feedback was sent successfully.")
        else:
            st.warning("⚠️ Feedback saved locally. Thank you!")
    
    # System analytics
    st.markdown("---")
    st.markdown("### 📊 System Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Chat Messages", len(st.session_state.chat_history))
    
    with col2:
        completion = st.session_state.loan_assistant.application.calculate_completion()
        st.metric("Application Progress", f"{completion:.0f}%")
    
    with col3:
        current_state = st.session_state.loan_assistant.conversation_state.value
        st.metric("Current State", current_state.replace('_', ' ').title())
    
    # Debug information
    if st.checkbox("Show Debug Information"):
        st.markdown("#### 🔧 Debug Information")
        st.json(st.session_state.loan_assistant.get_conversation_state())
        
        if hasattr(st.session_state.loan_assistant.application, 'to_dict'):
            st.markdown("#### Application Data")
            st.json(st.session_state.loan_assistant.application.to_dict())

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🔒 Your information is secure and used only for loan processing simulation.</p>
    <p>This comprehensive system combines conversational AI with advanced explainability methods (SHAP, DiCE, Anchor).</p>
    <p>Built with the UCI Adult dataset for demonstration purposes.</p>
</div>
""", unsafe_allow_html=True)