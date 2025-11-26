import streamlit as st

# Load environment variables from .env file
import env_loader

# Configure page FIRST - before any other Streamlit commands
st.set_page_config(page_title="AI Loan Assistant - Complete Solution", layout="wide")

# Now import everything else
from agent import Agent
from nlu import NLU
from answer import Answers
from github_saver import save_to_github
from loan_assistant import LoanAssistant
from ab_config import config
from shap_visualizer import display_shap_explanation, explain_shap_visualizations
import os
import pandas as pd

# Define field options for quick selection (based on actual Adult dataset analysis)
field_options = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
    'education': ['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '11th', '9th', '10th', '12th', '7th-8th', 'Doctorate', '1st-4th', '5th-6th', 'Preschool', 'Prof-school'],
    'marital_status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Armed-Forces', 'Priv-house-serv', 'Protective-serv', 'Transport-moving', '?'],
    'sex': ['Male', 'Female'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    'native_country': ['United-States', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'Vietnam', 'Yugoslavia', '?'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
}

# Str            <h3 style="margin: 0; color: white;">Hi! I'm Luna</h3>amlit compatibility function
def st_rerun():
    """Compatibility function for Streamlit rerun across versions"""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

# Custom CSS for better appearance with chat bubbles
st.markdown("""
<style>
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%);
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .chat-message {
        display: flex;
        margin: 0.8rem 0;
        align-items: flex-end;
        clear: both;
    }
    .user-message {
        justify-content: flex-end;
        flex-direction: row-reverse;
    }
    .assistant-message {
        justify-content: flex-start;
        flex-direction: row;
    }
    .message-bubble {
        padding: 10px 14px;
        border-radius: 18px;
        max-width: 65%;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
        line-height: 1.4;
        font-size: 14px;
    }
    .user-bubble {
        background: #007bff;
        color: white;
        border-bottom-right-radius: 4px;
        margin-right: 8px;
    }
    .user-bubble::after {
        content: '';
        position: absolute;
        right: -8px;
        bottom: 0;
        width: 0;
        height: 0;
        border-left: 8px solid #007bff;
        border-bottom: 8px solid transparent;
    }
    .assistant-bubble {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
        border-bottom-left-radius: 4px;
        margin-left: 8px;
    }
    .assistant-bubble::after {
        content: '';
        position: absolute;
        left: -9px;
        bottom: 0;
        width: 0;
        height: 0;
        border-right: 8px solid white;
        border-bottom: 8px solid transparent;
        border-top: 1px solid transparent;
    }
    .assistant-bubble::before {
        content: '';
        position: absolute;
        left: -10px;
        bottom: 0;
        width: 0;
        height: 0;
        border-right: 8px solid #e0e0e0;
        border-bottom: 8px solid transparent;
    }
    .profile-pic {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 5px;
        border: 2px solid #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        flex-shrink: 0;
    }
    .user-icon {
        width: 45px;
        height: 40px;
        border-radius: 50%;
        background: #007bff;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 11px;
        margin: 0 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        flex-shrink: 0;
    }
    .progress-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 3px;
        border: 1px solid #dee2e6;
    }
    .progress-fill {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        height: 22px;
        border-radius: 7px;
        text-align: center;
        line-height: 22px;
        color: white;
        font-weight: bold;
        font-size: 12px;
        box-shadow: 0 2px 4px rgba(0,123,255,0.2);
    }
    .status-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .luna-intro {
        display: flex;
        align-items: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
        .luna-intro img {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        margin-right: 15px;
        border: 3px solid white;
    }
    .option-button {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 3px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 13px;
        color: #495057;
        display: inline-block;
    }
    .option-button:hover {
        background: #e9ecef;
        border-color: #007bff;
        color: #007bff;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,123,255,0.2);
    }
    .option-button:active {
        background: #007bff;
        color: white;
        transform: translateY(0);
    }
    .options-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def initialize_system():
    """Initialize the agent and all components"""
    try:
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
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        st.error("Please check the console for more details.")
        import traceback
        st.code(traceback.format_exc())
        # Return None values to prevent further errors
        return None, None

# Initialize system
if 'agent' not in st.session_state:
    st.session_state.agent, st.session_state.answers = initialize_system()

# Check if initialization was successful
if st.session_state.agent is None:
    st.error("System initialization failed. Please check the error messages above and try refreshing the page.")
    st.stop()

agent = st.session_state.agent
answers = st.session_state.answers

# Initialize loan assistant
if 'loan_assistant' not in st.session_state:
    st.session_state.loan_assistant = LoanAssistant(agent)
    st.session_state.chat_history = []

# App header
st.title("🏦 AI Loan Assistant - Complete Solution")
st.markdown("*Multi-turn conversational loan applications with advanced AI explanations*")

# Assistant Introduction (A/B testing)
assistant_avatar = config.get_assistant_avatar()
if assistant_avatar and os.path.exists(assistant_avatar):
    import base64
    with open(assistant_avatar, "rb") as f:
        avatar_pic_b64 = base64.b64encode(f.read()).decode()
    
    st.markdown(f"""
    <div class="luna-intro">
        <img src="data:image/png;base64,{avatar_pic_b64}" alt="{config.assistant_name}">
        <div>
            <h3 style="margin: 0; color: white;">Hi! I'm {config.assistant_name}</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">{config.assistant_intro}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback without image
    st.markdown(f"""
    <div class="luna-intro">
        <div style="width: 60px; height: 60px; border-radius: 50%; margin-right: 15px; border: 3px solid white; background: #f093fb; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 24px;">{config.assistant_name[0]}</div>
        <div>
            <h3 style="margin: 0; color: white;">Hi! I'm {config.assistant_name}</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">{config.assistant_intro}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Single conversational interface
st.markdown("---")

# Sidebar for application status
with st.sidebar:
    st.header("Luna's Progress Tracker")
    
    # Get current state info
    state_info = st.session_state.loan_assistant.get_conversation_state()
    completion = st.session_state.loan_assistant.application.calculate_completion()
    
    # Progress bar
    st.markdown("### Progress")
    progress_html = f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {completion:.2f}%">{completion:.2f}%</div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # Current status
    st.markdown("### Current Status")
    current_state = state_info['state']  # Keep original state value for logic
    current_state_display = current_state.replace('_', ' ').title()
    st.markdown(f"**State:** {current_state_display}")
    
    if state_info.get('current_step'):
        current_step = state_info['current_step']
        step_desc = state_info.get('step_description', '')
        st.markdown(f"**Current Step:** {current_step}/10")
        if step_desc:
            st.markdown(f"**Collecting:** {step_desc}")
    elif state_info['current_field']:
        st.markdown(f"**Collecting:** {state_info['current_field'].replace('_', ' ').title()}")
    
    # Quick actions
    st.markdown("---")
    st.markdown("**Quick Actions**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Progress", key="sidebar_progress"):
            response = st.session_state.loan_assistant._show_progress()
            st.session_state.chat_history.append(("show progress", response))
            st_rerun()
    
    with col2:
        if st.button("Restart", key="sidebar_restart"):
            st.session_state.loan_assistant = LoanAssistant(agent)
            st.session_state.chat_history = []
            st_rerun()

    # High anthropomorphism: Explanation style selector (applies to generative rewriter where available)
    if config.show_anthropomorphic:
        st.markdown("---")
        st.markdown("**Explanation Style**")
        style = st.selectbox(
            "Tone and detail",
            options=["detailed", "short", "actionable"],
            index=0,
            help=f"Choose how {config.assistant_name} summarizes explanations."
        )
        # Make available to the backend via env var
        os.environ["HICXAI_STYLE"] = style
    
    # A/B Testing Debug Info (only for development/testing - hidden from users)
    # Uncomment the lines below only when debugging A/B testing locally
    # if config.show_debug_info and os.getenv('HICXAI_DEBUG_MODE', 'false').lower() == 'true':
    # What‑if Lab (V1 only, shown after user asks what-if)
    if config.show_counterfactual and getattr(st.session_state.loan_assistant, 'show_what_if_lab', False):
        st.markdown("---")
        st.subheader("🧪 What‑if Lab")
        st.caption("Adjust inputs to see how the predicted probability changes.")

        # Prepare a baseline instance from current app state if available
        app_state = st.session_state.loan_assistant.application
        def default(v, fallback):
            return v if v is not None else fallback

        # Core numerics
        age = st.slider("Age", min_value=17, max_value=90, value=int(default(app_state.age, 35)))
        hours = st.slider("Hours per week", min_value=1, max_value=99, value=int(default(app_state.hours_per_week, 40)))
        gain = st.number_input("Capital Gain", min_value=0, max_value=99999, step=100, value=int(default(app_state.capital_gain, 0)))
        loss = st.number_input("Capital Loss", min_value=0, max_value=4356, step=50, value=int(default(app_state.capital_loss, 0)))

        # Categorical selectors using known field options
        edu = st.selectbox("Education", options=field_options['education'], index=field_options['education'].index(default(app_state.education, 'HS-grad')))
        occ = st.selectbox("Occupation", options=field_options['occupation'], index=field_options['occupation'].index(default(app_state.occupation, 'Sales')))
        workclass = st.selectbox("Workclass", options=field_options['workclass'], index=field_options['workclass'].index(default(app_state.workclass, 'Private')))
        marital = st.selectbox("Marital Status", options=field_options['marital_status'], index=field_options['marital_status'].index(default(app_state.marital_status, 'Never-married')))
        relationship = st.selectbox("Relationship", options=field_options['relationship'], index=field_options['relationship'].index(default(app_state.relationship, 'Not-in-family')))
        sex = st.selectbox("Sex", options=field_options['sex'], index=field_options['sex'].index(default(app_state.sex, 'Male')))
        race = st.selectbox("Race", options=field_options['race'], index=field_options['race'].index(default(app_state.race, 'White')))
        country = st.selectbox("Native Country", options=field_options['native_country'], index=field_options['native_country'].index(default(app_state.native_country, 'United-States')))

        # Build a hypothetical instance and predict
        try:
            # Start from existing application dict (fill minimal defaults)
            hypo = app_state.to_dict()
            hypo['age'] = age
            hypo['hours_per_week'] = hours
            hypo['education'] = edu
            hypo['occupation'] = occ
            hypo['workclass'] = workclass
            hypo['marital_status'] = marital
            hypo['relationship'] = relationship
            hypo['sex'] = sex
            hypo['race'] = race
            hypo['native_country'] = country
            hypo['capital_gain'] = gain
            hypo['capital_loss'] = loss
            if hypo.get('education_num') is None:
                edu_map = {
                    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
                    '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
                    'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
                    'Prof-school': 15, 'Doctorate': 16
                }
                hypo['education_num'] = edu_map.get(edu, 9)
            # Ensure required fields have plausible defaults
            hypo.setdefault('workclass', 'Private')
            hypo.setdefault('marital_status', 'Never-married')
            hypo.setdefault('relationship', 'Not-in-family')
            hypo.setdefault('race', 'White')
            hypo.setdefault('sex', 'Male')
            hypo.setdefault('capital_gain', 0)
            hypo.setdefault('capital_loss', 0)
            hypo.setdefault('native_country', 'United-States')

            import pandas as pd
            app_df = pd.DataFrame([hypo])
            app_df['income'] = '<=50K'  # dummy
            from preprocessing import preprocess_adult
            processed = preprocess_adult(app_df)
            X = processed.drop('income', axis=1)
            # Align with training features
            train_df = pd.concat([agent.data['X_display'], agent.data['y_display']], axis=1)
            train_df_processed = preprocess_adult(train_df)
            expected = train_df_processed.drop('income', axis=1).columns.tolist()
            for col in expected:
                if col not in X.columns:
                    X[col] = 0
            X = X[expected]
            # Predict probability if available
            prob = None
            if hasattr(agent.clf_display, 'predict_proba'):
                p = agent.clf_display.predict_proba(X)
                # Assume class index 1 corresponds to '>50K'
                prob = float(p[0][1]) if p.shape[1] > 1 else float(p[0][0])
            st.metric(label="Estimated P(>50K)", value=f"{(prob if prob is not None else 0.5)*100:.1f}%")

            # Optional: refresh SHAP visuals for hypo profile (textual SHAP for now)
            # We keep visuals in the main flow; here we just indicate changes
            st.caption("Adjust inputs to explore their impact. Use chat for detailed explanations and visuals.")
        except Exception as e:
            st.caption(f"What‑if Lab unavailable: {e}")
    # Otherwise, no What‑if panel is shown until triggered by user
    #     st.markdown("---")
    #     st.markdown("**🧪 Debug Info**")
    #     st.markdown(f"Version: **{config.version}**")
    #     st.markdown(f"Assistant: **{config.assistant_name}**")
    #     st.markdown(f"SHAP Visuals: **{config.show_shap_visualizations}**")

# Chat interface - Display chat history with enhanced bubbles
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
    # User message (right side, blue bubble)
    if user_msg:
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="user-icon">You</div>
            <div class="message-bubble user-bubble">
                {user_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Assistant message with profile picture (left side, white bubble)
    if assistant_msg:
        assistant_avatar = config.get_assistant_avatar()
        if assistant_avatar and os.path.exists(assistant_avatar):
            import base64
            with open(assistant_avatar, "rb") as f:
                avatar_pic_b64 = base64.b64encode(f.read()).decode()
            avatar_pic_element = f'<img src="data:image/png;base64,{avatar_pic_b64}" class="profile-pic" alt="{config.assistant_name}">'
        else:
            avatar_pic_element = f'<div class="profile-pic" style="background: #f093fb; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 16px;">{config.assistant_name[0]}</div>'
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            {avatar_pic_element}
            <div class="message-bubble assistant-bubble">
                {assistant_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Initialize with welcome message
if len(st.session_state.chat_history) == 0:
    welcome_msg = st.session_state.loan_assistant.handle_message("hello")
    st.session_state.chat_history.append((None, welcome_msg))
    st_rerun()

# Chat input (form enables Enter-to-send and clears on submit automatically)
# Check if current field has clickable options for placeholder
current_field = getattr(st.session_state.loan_assistant, 'current_field', None)
if current_field and current_field in field_options:
    placeholder_text = "💬 Type your answer or use the clickable buttons below..."
else:
    placeholder_text = "Type your message to Luna..."

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_message = st.text_input("Message to Luna", key="user_input", placeholder=placeholder_text, label_visibility="collapsed")
    with col2:
        send_button = st.form_submit_button("Send", use_container_width=True)

# Add helper text for clickable features
if current_field and current_field in field_options:
    st.markdown('<div style="text-align: center; color: #666; font-size: 0.85em; margin-top: 5px;">👆 Use the clickable buttons below for faster selection!</div>', unsafe_allow_html=True)

# Show clickable options right after chat input (for immediate visibility)
if current_field and current_field in field_options:
    st.markdown("---")
    st.markdown(f"### 🎯 Quick Select: {current_field.replace('_', ' ').title()}")
    st.markdown("**💡 Click any option below instead of typing:**")
    st.markdown('<div class="options-container">', unsafe_allow_html=True)
    
    options = field_options[current_field]
    
    # Create buttons in rows with enhanced styling
    cols_per_row = 4 if len(options) > 8 else 3
    for i in range(0, len(options), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, option in enumerate(options[i:i+cols_per_row]):
            with cols[j]:
                # Enhanced button styling based on option type
                if option == "Other":
                    button_text = f"🔄 {option}"
                    button_type = "primary"
                elif option == "?":
                    button_text = f"❓ Unknown/Prefer not to say"
                    button_type = "primary"
                elif option in ["Male", "Female"]:
                    button_text = f"👤 {option}"
                    button_type = "secondary"
                elif option == "United-States":
                    button_text = f"🇺🇸 {option}"
                    button_type = "primary"
                elif option in ["Private", "Self-emp-not-inc", "Self-emp-inc"]:
                    button_text = f"💼 {option}"
                    button_type = "secondary"
                elif "gov" in option.lower():
                    button_text = f"🏛️ {option}"
                    button_type = "secondary"
                else:
                    button_text = f"✨ {option}"
                    button_type = "secondary"
                
                if st.button(button_text, key=f"option_top_{current_field}_{option}", use_container_width=True, type=button_type):
                    st.session_state.option_clicked = option
                    st_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("*💬 Or you can still type your answer in the chat box above*")

# Process user input
if send_button and user_message:
    # Handle the message through loan assistant
    assistant_response = st.session_state.loan_assistant.handle_message(user_message)
    # Add to chat history (form clears input on submit)
    st.session_state.chat_history.append((user_message, assistant_response))
    st_rerun()

# Handle option clicks
if 'option_clicked' in st.session_state and st.session_state.option_clicked:
    option_value = st.session_state.option_clicked
    assistant_response = st.session_state.loan_assistant.handle_message(option_value)
    
    # Add to chat history
    st.session_state.chat_history.append((option_value, assistant_response))
    st.session_state.option_clicked = None  # Reset
    st_rerun()

# Persistent SHAP visuals section: render when feature_importance explanation is enabled
if config.show_shap_visualizations:
    shap_data = getattr(st.session_state.loan_assistant, 'last_shap_result', None)
    if shap_data:
        st.markdown("---")
        st.subheader("🔎 Visual Explanations")
        display_shap_explanation(shap_data)
        explain_shap_visualizations()

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
        if st.button("Check Progress", key="quick_progress"):
            response = st.session_state.loan_assistant.handle_message("review")
            st.session_state.chat_history.append(("check progress", response))
            st_rerun()
    with col2:
        if st.button("Help", key="quick_help"):
            # Get context-aware help
            current_field = getattr(st.session_state.loan_assistant, 'current_field', None)
            if current_field:
                help_msg = st.session_state.loan_assistant._get_field_help(current_field)
                help_msg += f"\n\n💡 **You can also:**\n• Say 'review' to see your progress\n• Click the quick-select buttons below\n• Ask for specific examples"
            else:
                help_msg = ("I'm collecting information for your loan application. Please answer the questions "
                           "as accurately as possible. You can say 'review' to see your progress.")
            st.session_state.chat_history.append(("help", help_msg))
            st_rerun()

elif current_state == 'complete':
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Explain Decision", key="quick_explain"):
            response = st.session_state.loan_assistant.handle_message("explain")
            st.session_state.chat_history.append(("explain", response))
            st_rerun()
    with col2:
        if st.button("New Application", key="quick_new"):
            response = st.session_state.loan_assistant.handle_message("new")
            st.session_state.chat_history.append(("new application", response))
            st_rerun()
    with col3:
        if st.button("🔧 What If Analysis", key="quick_whatif"):
            # Turn on What‑if Lab and prompt guidance
            try:
                st.session_state.loan_assistant.show_what_if_lab = True
            except Exception:
                pass
            response = "What‑if Lab enabled in the sidebar. Adjust Age, Hours, Education, or Occupation to see how the probability changes."
            st.session_state.chat_history.append(("what if analysis", response))
            st_rerun()

# Clickable Options for Current Field (if collecting info)
if current_state == 'collecting_info' and hasattr(st.session_state.loan_assistant, 'current_field') and st.session_state.loan_assistant.current_field:
    current_field = st.session_state.loan_assistant.current_field
    
    if current_field in field_options:
        st.markdown("---")
        st.markdown(f"### 🎯 Quick Select: {current_field.replace('_', ' ').title()}")
        st.markdown("**💡 Click any option below instead of typing:**")
        st.markdown('<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 15px; border-radius: 10px; margin: 10px 0; border: 1px solid #dee2e6;">', unsafe_allow_html=True)
        
        options = field_options[current_field]
        
        # Create buttons in rows with enhanced styling
        cols_per_row = 4 if len(options) > 8 else 3
        for i in range(0, len(options), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, option in enumerate(options[i:i+cols_per_row]):
                with cols[j]:
                    # Enhanced button styling based on option type
                    if option == "Other":
                        button_text = f"🔄 {option}"
                        button_type = "primary"
                    elif option == "?":
                        button_text = f"❓ Unknown/Prefer not to say"
                        button_type = "primary"
                    elif option in ["Male", "Female"]:
                        button_text = f"👤 {option}"
                        button_type = "secondary"
                    elif option == "United-States":
                        button_text = f"🇺🇸 {option}"
                        button_type = "primary"
                    elif option in ["Private", "Self-emp-not-inc", "Self-emp-inc"]:
                        button_text = f"💼 {option}"
                        button_type = "secondary"
                    elif "gov" in option.lower():
                        button_text = f"🏛️ {option}"
                        button_type = "secondary"
                    else:
                        button_text = f"✨ {option}"
                        button_type = "secondary"
                    
                    if st.button(button_text, key=f"option_{current_field}_{option}", use_container_width=True, type=button_type):
                        st.session_state.option_clicked = option
                        st_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("*💬 Or you can still type your answer in the chat box above*")

# XAI Analysis Options (available once application is complete)
if current_state == 'complete':
    # Show explanation options only if any explanations are enabled
    if config.show_any_explanation:
        st.markdown("---")
        st.markdown("### 🔍 AI Explanation Options")
        st.markdown("Now that your application is complete, you can ask for detailed explanations:")
        
        # Dynamically create columns based on enabled explanation types
        enabled_explanations = []
        if config.show_shap_visualizations:
            enabled_explanations.append('shap')
        if config.show_counterfactual:
            enabled_explanations.append('dice')
        # Note: Anchor is always available when any explanation is enabled
        
        if enabled_explanations or config.show_any_explanation:
            num_cols = len(enabled_explanations) if enabled_explanations else 1
            cols = st.columns(num_cols)
            col_idx = 0
            
            # Show SHAP option only if feature_importance explanation enabled
            if config.show_shap_visualizations:
                with cols[col_idx]:
                    st.markdown("**🎯 Feature Importance (SHAP)**")
                    st.markdown("• 'Which factors affected my decision?'")
                    st.markdown("• 'Why was I approved/denied?'")
                    if st.button("Ask Feature Importance", key="shap_question"):
                        query = "Which factors most affected my loan decision?"
                        response = st.session_state.loan_assistant.handle_message(query)
                        
                        # Add SHAP visualizations if available
                        if (config.show_shap_visualizations and 
                            hasattr(st.session_state.loan_assistant, 'last_shap_result') and 
                            st.session_state.loan_assistant.last_shap_result):
                            shap_data = st.session_state.loan_assistant.last_shap_result
                            # Render visuals in the UI (function returns None), do not append to text
                            display_shap_explanation(shap_data)
                            explain_shap_visualizations()
                        
                        st.session_state.chat_history.append((query, response))
                        st_rerun()
                col_idx += 1
            
            # Show counterfactual option only if counterfactual explanation enabled
            if config.show_counterfactual:
                with cols[col_idx]:
                    st.markdown("**🔄 What-If Analysis (DiCE)**")
                    st.markdown("• 'What if my income was higher?'")
                    st.markdown("• 'What changes would get me approved?'")
                    if st.button("Ask What-If", key="dice_question"):
                        response = st.session_state.loan_assistant.handle_message("What changes would help me get approved?")
                        st.session_state.chat_history.append(("What changes would help me get approved?", response))
                        st_rerun()
                col_idx += 1

# Feedback section (appears after application is complete)
if current_state == 'complete' and len(st.session_state.chat_history) > 5:
    st.markdown("---")
    st.markdown("### 📝 Your Feedback")
    st.markdown("Help us improve by sharing your experience:")
    
    with st.form("feedback_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            rating = st.select_slider(
                "How would you rate your experience?",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
            
            ease_of_use = st.radio(
                "Was the application process easy to understand?",
                ["Very Easy", "Easy", "Neutral", "Difficult", "Very Difficult"]
            )
        
        with col2:
            explanation_clarity = st.radio(
                "Were the AI explanations helpful?",
                ["Very Helpful", "Helpful", "Neutral", "Not Helpful", "Confusing"]
            )
            
            would_recommend = st.radio(
                "Would you recommend this service?",
                ["Definitely", "Probably", "Maybe", "Probably Not", "Definitely Not"]
            )
        
        feedback_text = st.text_area(
            "Additional comments (optional):",
            placeholder="“What feature would help you most next time?”\n“What would make this agent's explanations more useful?”..."
        )
        
        submitted = st.form_submit_button("Submit Feedback 🚀")
        
        if submitted:
            feedback_data = {
                "rating": rating,
                "ease_of_use": ease_of_use,
                "explanation_clarity": explanation_clarity,
                "would_recommend": would_recommend,
                "additional_comments": feedback_text,
                "conversation_length": len(st.session_state.chat_history),
                "completion_percentage": completion,
                # A/B Testing metadata
                "ab_version": config.version,
                "session_id": config.session_id,
                "assistant_name": config.assistant_name,
                "had_shap_visualizations": config.show_shap_visualizations,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Save feedback
            try:
                # Try GitHub first (if configured)
                github_token = os.getenv('GITHUB_TOKEN')
                github_repo = os.getenv('GITHUB_REPO', 'your-username/your-repo')
                
                if github_token:
                    import json
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"feedback/session_{config.session_id}_{timestamp}.json"
                    
                    success = save_to_github(
                        repo=github_repo,
                        path=filename,
                        content=json.dumps(feedback_data, indent=2),
                        commit_message=f"User feedback - {config.version} - {timestamp}",
                        github_token=github_token
                    )
                    
                    if success:
                        st.success("Thank you for your feedback! 🎉")
                    else:
                        raise Exception("GitHub save failed")
                else:
                    raise Exception("No GitHub token configured")
                    
            except Exception as e:
                st.warning("Feedback saved locally. Thank you!")
                # Fallback: save to local file
                import json
                os.makedirs('feedback', exist_ok=True)
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"feedback/session_{config.session_id}_{timestamp}.json"
                
                with open(filename, "w") as f:
                    f.write(json.dumps(feedback_data, indent=2))

# Footer with detailed dataset information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏦 AI Loan Assistant | Powered by Explainable AI</p>
    <p><small>Features: Multi-turn conversation • SHAP explanations • DiCE what-if analysis • Anchor rules • Natural language processing</small></p>
    <p><small>🔬 Algorithm trained on the Adult (Census Income) dataset with 32,561 records from the UCI Machine Learning Repository</small></p>
</div>
""", unsafe_allow_html=True)

# Expandable dataset details
with st.expander("📊 Dataset Information - Adult Census Income Dataset"):
    st.markdown("""
    **Dataset Overview:**
    
    The Adult Census Income Dataset is a popular benchmark dataset from the UCI Machine Learning Repository, 
    sometimes referred to as the Census Income or Adult dataset. It includes **32,561 records** and **15 attributes**, 
    each representing a person's social, employment, and demographic information. The dataset originates from the 
    U.S. Census database from 1994.
    
    **Prediction Task:**
    
    The main goal is to determine whether an individual makes more than $50,000 per year based on their attributes. 
    The income is the target variable with two possible classes:
    - **≤50K**: Income less than or equal to $50,000
    - **>50K**: Income greater than $50,000
    
    **Dataset Features:**
    
    The dataset contains both qualitative and numerical attributes:
    
    - **Age**: Numerical value indicating person's age
    - **Workclass**: Type of employment (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
    - **Education / Education-num**: Highest education level (both textual and numerical representation)
    - **Marital-status**: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, etc.)
    - **Occupation**: Work area (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, etc.)
    - **Relationship**: Family role (Husband, Wife, Own-child, Not-in-family, Other-relative, Unmarried)
    - **Race**: Ethnic background (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
    - **Sex**: Gender (Male, Female)
    - **Capital-gain / Capital-loss**: Investment gains or losses
    - **Hours-per-week**: Number of working hours per week
    - **Native-country**: Country of origin
    - **Income**: Target label (≤50K or >50K)
    
    **Model Performance:**
    
    Our trained RandomForest classifier achieves **85.94% accuracy** on this dataset, making it suitable for 
    demonstrating explainable AI techniques and understanding feature importance in income prediction.
    """)

# A/B Testing Debug Info (only for development - hidden from users)
# Only show when HICXAI_DEBUG_MODE environment variable is set to 'true'
if os.getenv('HICXAI_DEBUG_MODE', 'false').lower() == 'true':
    st.markdown("---")
    st.markdown("### 🧪 A/B Testing Information (Debug Mode)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Version:** {config.version}")
        st.markdown(f"**Session ID:** {config.session_id}")
    with col2:
        st.markdown(f"**Assistant:** {config.assistant_name}")
        st.markdown(f"**SHAP Visuals:** {config.show_shap_visualizations}")
    with col3:
        st.markdown(f"**Concurrent Testing:** ✅ Enabled")
        st.markdown(f"**User Isolation:** ✅ Session-based")