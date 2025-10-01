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
- **Question-Intent Dataset** (`data_questions/Median_4.csv`) curated by [XAgent](https://github.com/bach1292/XAGENT), adapted from original work by [Liao et al. (2020)](https://arxiv.org/abs/2001.02478)
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

## References

### Primary Sources
- **Liao, Q.V., Gruen, D., & Miller, S.** (2020). Questioning the AI: Informing Design Practices for Explainable AI User Experiences. *CHI '20: CHI Conference on Human Factors in Computing Systems*. [https://arxiv.org/abs/2001.02478](https://arxiv.org/abs/2001.02478)
- **bach1292** - XAgent: Explainable AI Agent Framework. [https://github.com/bach1292/XAGENT](https://github.com/bach1292/XAGENT)

### Dataset Attribution
- **Adult (Census Income) Dataset**: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. [http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml)
- **Question-Intent Pairs** (`Median_4.csv`): Curated by [XAgent](https://github.com/bach1292/XAGENT), adapted from original research by Liao et al. (2020)

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{hicxai_research_2024,
  title={HicXAI Research Platform: A/B Testing Framework for Explainable AI User Studies},
  author={[Your Name]},
  year={2024},
  url={https://github.com/ksauka/hicxai-research},
  note={Adapted from XAgent by bach1292 and datasets from Liao et al. 2020}
}
```

## License
MIT License
