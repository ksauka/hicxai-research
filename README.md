# HicXAI Research Platform

A Human-Computer Interaction research platform for studying user needs and preferences in explainable AI (XAI) for sensitive domains. This project investigates how different explanation modalities affect user trust and decision-making in AI-assisted financial services.

## Research Objectives

This platform conducts user studies on:
- **User preferences** for different XAI explanation types
- **Trust calibration** in AI explanations for sensitive financial decisions  
- **Effectiveness** of anthropomorphic vs. minimal AI assistant interfaces
- **UX design patterns** for explainable AI in high-stakes domains

## Attribution & Acknowledgments

This work builds upon and adapts components from:
- **[XAgent](https://github.com/bach1292/XAGENT)** by bach1292 - Original XAI agent framework and Adult dataset integration
- **Adult Dataset** from UCI Machine Learning Repository via XAgent implementation
- **SimCSE** semantic similarity model for zero-shot intent classification

## Key Research Contributions

### 1. A/B Testing Framework
- **Control Group (v0)**: Minimal AI assistant interface
- **Treatment Group (v1)**: Anthropomorphic "Luna" assistant with SHAP visualizations
- Concurrent deployment system for randomized user assignment

### 2. Human-Centered Design
- **Fuzzy matching** for natural language query understanding
- **Conversational UI** in Streamlit for accessible user interaction
- **Optional feedback collection** with privacy-preserving data handling

### 3. Production-Ready Architecture
- **Streamlined dataset focus**: Adult dataset only (vs. multiple datasets in original XAgent)
- **Zero-shot deployment**: No model training required using SimCSE
- **Secure data separation**: Public app repository + private research data repository

## Dataset

🔬 **Adult (Census Income) Dataset**: 32,561 records from the UCI Machine Learning Repository
- **Source**: U.S. Census database (1994) via [XAgent implementation](https://github.com/bach1292/XAGENT)
- **Task**: Binary classification (income ≤$50K vs >$50K) 
- **Features**: 15 demographic, social, and employment attributes
- **Usage**: Sensitive financial decision-making context for trust studies

## Architecture

### Core Components
- `src/app.py`: Main Streamlit application with A/B testing logic
- `src/ab_config.py`: A/B testing configuration and version control
- `src/nlu.py`: SimCSE-based semantic similarity and intent classification
- `src/xai_methods.py`: Natural language explanation generation (SHAP, DiCE, Anchor)
- `src/shap_visualizer.py`: SHAP visualization components for treatment group
- `app_v0.py` / `app_v1.py`: Streamlit Cloud deployment entry points

### Research Infrastructure
- `src/github_saver.py`: Secure feedback collection to private repository
- `data_questions/Median_4.csv`: Curated question-intent pairs for semantic matching
- `.streamlit/config.toml`: UI customization for research study presentation

## Deployment

### Local Development
```bash
conda activate xagent
streamlit run app_v0.py  # Control group
streamlit run app_v1.py  # Treatment group
```

### Production (Streamlit Cloud)
- **Control**: Deploy `app_v0.py` → Minimal AI assistant interface
- **Treatment**: Deploy `app_v1.py` → Luna with SHAP visualizations
- **Secrets**: Configure `GITHUB_TOKEN` and `GITHUB_REPO` for data collection

## Research Ethics & Privacy

- **Informed Consent**: Clear explanation of research purpose in UI
- **Optional Participation**: All feedback collection is voluntary
- **Data Minimization**: Only essential interaction data collected
- **Secure Storage**: Research data stored in private GitHub repository
- **Anonymization**: No personally identifiable information collected

## Technical Implementation

### How It Works
1. **Question Understanding**: SimCSE finds semantically similar questions in knowledge base
2. **Intent Classification**: Maps matched questions to XAI method intents (SHAP, DiCE, Anchor)
3. **Natural Explanations**: Generates human-readable explanations instead of technical outputs
4. **A/B Assignment**: Routes users to control vs. treatment interfaces based on configuration

### Extending the Platform
- Add new questions to `data_questions/Median_4.csv` (no retraining needed)
- Extend XAI methods in `src/xai_methods.py` 
- Add new intent mappings in `src/constraints.py`
- Modify A/B testing variants in `src/ab_config.py`

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{hicxai_research_2024,
  title={HicXAI Research Platform: A/B Testing Framework for Explainable AI User Studies},
  author={[Your Name]},
  year={2024},
  url={https://github.com/ksauka/hicxai-research},
  note={Adapted from XAgent by bach1292: https://github.com/bach1292/XAGENT}
}
```

## License

MIT License - See LICENSE file for details