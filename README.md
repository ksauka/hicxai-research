# HicXAI_agent

A robust, modular, and production-ready explainable AI (XAI) agent for intent-to-XAI-method mapping, powered by **SimCSE semantic similarity**. Uses pre-trained models without fine-tuning, providing immediate deployment capability with natural language explanations.

## Features
- **SimCSE-based NLU**: Semantic similarity for intent classification and XAI method routing (no training required)
- **Natural Language Explanations**: Human-readable explanations instead of just technical plots
- **Modular Architecture**: Easily extend datasets, XAI methods, and agent logic
- **XAgent Integration**: Enhanced modules adopted from XAgent for robust explanation generation
- **Environment Management**: Automated scripts for consistent execution across environments
- **Feedback Loop**: Collects structured user feedback via conversational Streamlit UI
- **Instant Deployment**: No model training - ready to use immediately

## Quick Start

1. **Setup Environment & GitHub Integration**
   ```bash
   # Setup GitHub data collection (creates .env file)
   ./setup_github.sh
   
   # Test the system
   ./run_in_env.sh test_enhanced_nlu.py
   ```

2. **Deploy A/B Testing**
   ```bash
   # Start both versions concurrently
   ./deploy_ab_testing.sh
   
   # Monitor progress
   ./monitor_progress.sh
   ```

3. **Share with Test Subjects**
   ```bash
   # Get shareable links
   ./generate_user_links.sh
   
   # Check status
   ./check_ab_status.sh
   ```

## Feedback
- The app collects user feedback with conversational prompts. All fields are optional.
- Feedback is saved to GitHub for later processing and model improvement.

## How It Works
1. **Question Understanding**: SimCSE finds semantically similar questions in the knowledge base
2. **Intent Classification**: Maps matched questions to XAI method intents (SHAP, DiCE, Anchor)
3. **Natural Explanations**: Generates human-readable explanations instead of technical outputs
4. **Ambiguity Handling**: Provides suggestions when user intent is unclear

## Key Components
- `src/nlu.py`: SimCSE-based semantic similarity and intent classification
- `src/xai_methods.py`: Natural language explanation generation for SHAP, DiCE, Anchor
- `src/agent.py`: Main orchestrator for user interaction and response generation
- `src/constraints.py`: Intent-to-method mapping and user messages
- `run_in_env.sh`: Environment management utility

## Extending
- Add new questions to `bert_data/Median_4.csv` (no retraining needed)
- Extend XAI methods in `src/xai_methods.py` 
- Add new intent mappings in `src/constraints.py`

## Environment Management
The included `run_in_env.sh` script ensures consistent execution:
- Automatically navigates to project directory
- Activates the correct conda environment
- Handles dependency conflicts
- Provides colored status output

## License
MIT License
`conda activate xagent`
`python -m streamlit run src/app.py`

this information below should be included on the line: 🔬 Algorithm trained on the Adult (Census Income) dataset with 32,561 records from the UCI Machine Learning Repository
Income Dataset from the Adult Census (Details)

One popular benchmark dataset from the UCI Machine Learning Repository is the Adult Census Income Dataset, sometimes referred to as the Census Income or Adult dataset. It includes 32,561 records and 15 qualities, each of which represents a person's social, employment, and demographic information. The initial source of the dataset was the U.S. Census database from 1994.

This dataset's main goal is to determine, using an individual's qualities, whether or not they make more than $50,000 per year. Income is the target variable, and there are two possible classes:
<=50K &>50K

Overview of Features
Both qualitative and numerical attributes are present in the dataset, including:

Age:A numerical value that indicates how old a person is.

Workclass:Type of employment (e.g., government, self-employed, private).

Education / Education.num:The highest level of education attained, expressed both numerically and textually.

Marital.status:Marital status (e.g., Married, Divorced, Widowed).

Occupation:Work area (e.g., craftsmanship, tech assistance, executive management)

Relationship:Family role (e.g., Husband, Wife, Own-child).

Race:Ethnic background.

Sex:Male or Female.

Capital.gain / Capital.loss:Investment gains or losses.

Hours.per.week:Number of working hours per week.

Native.country:Origin country.

Income:Target label (less than or more than $50,000).