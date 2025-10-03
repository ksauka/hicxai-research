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

### Local Development

1. **Setup Environment**
   ```bash
   # Activate the conda environment
   conda activate xagent
   
   # Navigate to project directory
   cd /path/to/hicxai-research
   ```

2. **Run A/B Testing Locally**
   ```bash
   # Control Group (v0) - Minimal interface
   streamlit run app_v0.py --server.port 8501

   # Treatment Group (v1) - Luna with SHAP visualizations
   # Tip: enable full visualizations locally
   HICXAI_MODE=full streamlit run app_v1.py --server.port 8502
   ```

   Note: The entrypoints `app_v0.py` and `app_v1.py` force their respective versions regardless of environment variables. When running `src/app.py` directly, the `HICXAI_VERSION` environment variable takes precedence over CLI args; default is v0.

3. **Access Local Apps**
   - **Control Group**: http://localhost:8501
   - **Treatment Group**: http://localhost:8502

### Cloud Deployment (Streamlit Cloud)

1. **Create Apps**
   - Go to https://streamlit.io/cloud
   - Create app with repository: `ksauka/hicxai-research`
   - For Control: Use `app_v0.py` as main file
   - For Treatment: Use `app_v1.py` as main file

2. **Configure Secrets** (for both apps)
   ```toml
   GITHUB_TOKEN = "your_github_personal_access_token"
   GITHUB_REPO = "yourusername/your-private-data-repo"

3. **Environment & Python Version**
   - This repo includes `runtime.txt` (3.10) to avoid Python 3.13/build issues on Cloud (distutils removal).
   - If Cloud asks for a requirements file, point it to `requirements-streamlit.txt` for a CPU-friendly set.
   ```

## Feedback
- The app collects user feedback with conversational prompts. All fields are optional.
- Feedback is saved to GitHub for later processing and model improvement.

## How It Works
1. **Question Understanding**: sentence-transformers (all-MiniLM-L6-v2) finds semantically similar questions in the knowledge base (GPU locally; CPU on Streamlit)
2. **Intent Classification**: Maps matched questions to XAI method intents (SHAP, DiCE, Anchor)
3. **Natural Explanations**: Generates human-readable explanations instead of technical outputs
4. **Ambiguity Handling**: Provides suggestions when user intent is unclear

## Key Components
- `src/app.py`: Main Streamlit application with A/B testing logic
- `src/ab_config.py`: A/B testing configuration and version control
- `src/nlu.py`: sentence-transformers semantic similarity and intent classification
- `src/xai_methods.py`: Natural language explanation generation for SHAP, DiCE, Anchor
- `src/shap_visualizer.py`: SHAP visualization components for treatment group
- `src/github_saver.py`: Secure feedback collection to private repository
- `app_v0.py` / `app_v1.py`: Streamlit Cloud deployment entry points

## Extending the Platform
- Add new questions to `data_questions/Median_4.csv` (no retraining needed)
- Extend XAI methods in `src/xai_methods.py` 
- Add new intent mappings in `src/constraints.py`
- Modify A/B testing variants in `src/ab_config.py`
- Customize UI themes in `.streamlit/config.toml`

## Dependencies
- **Local GPU dev**: Python 3.10 (conda env `hicxai_rtx5070`)
   - Create env: `conda env create -f environment_rtx5070.yml`
   - Install cu128 torch trio: `pip install -r requirements_rtx5070_torch.txt`
- **Streamlit Cloud**: uses `requirements.txt` -> `requirements-streamlit.txt` (CPU wheels)
- Core packages: streamlit, pandas, scikit-learn, shap, sentence-transformers, dice-ml, anchor-exp, dtreeviz, graphviz

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

## Generative Rewriter (Optional)

You can enable a friendlier, style-controlled paraphrase of explanations. The default model is `gpt-4o-mini`.

Environment variables:

- `OPENAI_API_KEY` – required to enable
- `HICXAI_GENAI` – `on` (default) or `off`
- `HICXAI_OPENAI_MODEL` – defaults to `gpt-4o-mini`
- `HICXAI_STYLE` – `short` | `detailed` | `actionable` (v1 sidebar can set this at runtime)
- `HICXAI_TEMPERATURE` – defaults to `0.2`
- `HICXAI_MAX_TOKENS` – defaults to `300`
- `HICXAI_OPENAI_BASE_URL` – optional base URL for proxies (respects `OPENAI_BASE_URL` too)

Example (local):

```bash
export OPENAI_API_KEY=sk-...
export HICXAI_OPENAI_MODEL=gpt-4o-mini
streamlit run src/app.py
```

Streamlit Cloud (protecting tokens):

- Open your app → Settings → Secrets and add `OPENAI_API_KEY = your_key`.
- Do not print or log the key; this project never echoes it.
- The app automatically reads `st.secrets["OPENAI_API_KEY"]` when the env var is missing.
- Optionally add `HICXAI_OPENAI_BASE_URL` to Secrets if you use a proxy.

