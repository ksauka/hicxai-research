# HicXAI Research Platform

A Human-Computer Interaction research platform for studying user needs and preferences in explainable AI (XAI) for sensitive domains. This project implements a **3×2 factorial experiment** investigating how different explanation modalities and interface anthropomorphism affect user trust and decision-making in AI-assisted financial services.

## Experimental Design (3×2 Factorial)

**Factor 1 - Explanation Type (3 levels):**
- `none` - No explanations provided
- `counterfactual` - DiCE counterfactual explanations only
- `feature_importance` - SHAP feature importance visualizations only

**Factor 2 - Anthropomorphism (2 levels):**
- `low` - Minimal, technical AI assistant
- `high` - Luna with friendly, conversational tone and avatar

**6 Experimental Conditions:**
| Condition | Explanation | Anthropomorphism | App Entry Point | Condition Code |
|-----------|-------------|------------------|-----------------|----------------|
| 1 | none | low | `app_v0.py` | `E_none_A_low` |
| 2 | none | high | `app_condition_2.py` | `E_none_A_high` |
| 3 | counterfactual | low | `app_condition_3.py` | `E_cf_A_low` |
| 4 | counterfactual | high | `app_condition_4.py` | `E_cf_A_high` |
| 5 | feature_importance | low | `app_condition_5.py` | `E_shap_A_low` |
| 6 | feature_importance | high | `app_v1.py` | `E_shap_A_high` |

## Research Questions

This platform investigates:
- **RQ1**: How do different explanation types (none, counterfactual, feature importance) affect user trust and understanding?
- **RQ2**: How does anthropomorphism (low vs. high) influence user perception of AI explanations?
- **RQ3**: Are there interaction effects between explanation type and anthropomorphism?
- **RQ4**: Which combination optimizes user trust calibration in high-stakes financial decisions?

## Attribution & Acknowledgments

This work builds upon and adapts components from:
- **[XAgent](https://github.com/bach1292/XAGENT)** by bach1292 - Original XAI agent framework and Adult dataset integration
- **Adult Dataset** from UCI Machine Learning Repository via XAgent implementation
- **Question-Intent Dataset** (`data_questions/Median_4.csv`) curated by [XAgent](https://github.com/bach1292/XAGENT), adapted from original work by [Liao et al. (2020)](https://arxiv.org/abs/2001.02478)
- **sentence-transformers** (prefers all-MiniLM-L6-v2) for semantic similarity in intent classification. The original XAgent used SimCSE; we preserve the spirit of zero-shot matching while defaulting to sentence-transformers when available.

## Key Research Contributions

### 1. Full Factorial Experimental Design
- **3×2 factorial design** with 6 experimental conditions
- **Clean factor isolation**: Explanation type and anthropomorphism are independently manipulated
- **Condition tracking**: Each session logged with unique ID including condition code (e.g., `E_cf_A_high_1732612345_a1b2c3d4`)
- **Concurrent deployment**: Multiple Streamlit Cloud apps for balanced randomized assignment

### 2. Controlled XAI Comparison
- **Counterfactual explanations** (DiCE): "What changes would lead to approval?"
- **Feature importance** (SHAP): Visual bar charts showing factor contributions
- **No explanations baseline**: Control condition for measuring explanation value

### 3. Human-Centered Design
- **Fuzzy matching** for natural language query understanding via sentence-transformers
- **Conversational UI** in Streamlit for accessible user interaction
- **Adaptive interface**: UI elements shown/hidden based on experimental condition
- **Optional feedback collection** with privacy-preserving data handling


## Quick Start

### Local Development

1. **Setup Environment**
   ```bash
   # Activate the conda environment
   conda activate hicxai_rtx5070
   # or
   conda activate xagent
   
   # Navigate to project directory
   cd /path/to/hicxai-research
   ```

2. **Run Specific Experimental Conditions Locally**
   ```bash
   # Condition 1: No explanations, low anthropomorphism (v0)
   streamlit run app_v0.py --server.port 8501

   # Condition 2: No explanations, high anthropomorphism (Luna)
   streamlit run app_condition_2.py --server.port 8502

   # Condition 3: Counterfactual only, low anthropomorphism
   streamlit run app_condition_3.py --server.port 8503

   # Condition 4: Counterfactual only, high anthropomorphism (Luna + DiCE)
   streamlit run app_condition_4.py --server.port 8504

   # Condition 5: SHAP only, low anthropomorphism
   streamlit run app_condition_5.py --server.port 8505

   # Condition 6: SHAP only, high anthropomorphism (Luna + SHAP) - v1
   streamlit run app_v1.py --server.port 8506
   ```

   **Or use environment variables directly:**
   ```bash
   HICXAI_EXPLANATION=counterfactual HICXAI_ANTHRO=high streamlit run src/app.py
   ```

   Note: Entry point files (`app_v0.py`, `app_v1.py`, `app_condition_*.py`) force their respective conditions regardless of environment variables.

3. **Access Local Apps**
   - Open browser to the appropriate port (8501-8506)

### Cloud Deployment (Streamlit Cloud)

1. **Create 6 Separate Apps** (one per condition)
   - Go to https://streamlit.io/cloud
   - Create 6 apps with repository: `ksauka/hicxai-research`
   - Use corresponding entry points:
     - **Condition 1** (E_none_A_low): `app_v0.py`
     - **Condition 2** (E_none_A_high): `app_condition_2.py`
     - **Condition 3** (E_cf_A_low): `app_condition_3.py`
     - **Condition 4** (E_cf_A_high): `app_condition_4.py`
     - **Condition 5** (E_shap_A_low): `app_condition_5.py`
     - **Condition 6** (E_shap_A_high): `app_v1.py`

2. **Configure Secrets** (optional, per app - same for all 6 apps)
   Add only what you use:
   ```toml
   # Optional: generative rewriter
   OPENAI_API_KEY = "sk-..."
   # Optional: proxy/base URL
   HICXAI_OPENAI_BASE_URL = "https://api.openai.com/v1"

   # Optional: GitHub feedback saver
   GITHUB_TOKEN = "ghp_..."
   GITHUB_REPO = "yourusername/your-private-repo"
   ```

3. **Environment & Python Version**
   - Cloud installs from `requirements.txt` only (CPU-friendly, Python 3.13 compatible)
   - Apt packages from `packages.txt` are installed automatically (includes `graphviz`)
   - Local GPU tooling is optional and does not affect Cloud

4. **User Assignment Strategy**
   - **Option A**: Manual assignment via Prolific/survey platform (redirect users to specific app URLs)
   - **Option B**: Create a landing page/router that randomly assigns users to one of 6 app URLs
   - Track condition code in session IDs for analysis

## Feedback
- The app collects user feedback with conversational prompts. All fields are optional.
- Feedback is saved to GitHub for later processing and model improvement.

## How It Works
1. **Question Understanding**: sentence-transformers (all-MiniLM-L6-v2) finds semantically similar questions in the knowledge base (GPU locally; CPU on Streamlit)
2. **Intent Classification**: Maps matched questions to XAI method intents (SHAP, DiCE, Anchor)
3. **Natural Explanations**: Generates human-readable explanations instead of technical outputs
4. **Ambiguity Handling**: Provides suggestions when user intent is unclear

## Key Components
- `src/app.py`: Main Streamlit application with conditional rendering based on experimental factors
- `src/ab_config.py`: 3×2 factorial configuration with factor-based feature flags
- `src/loan_assistant.py`: Conversational flow manager with explanation routing
- `src/nlu.py`: sentence-transformers semantic similarity and intent classification
- `src/xai_methods.py`: Natural language explanation generation for SHAP, DiCE, Anchor
- `src/shap_visualizer.py`: SHAP visualization components (shown only in `feature_importance` conditions)
- `src/github_saver.py`: Secure feedback collection with condition tracking
- `app_v0.py`, `app_v1.py`, `app_condition_2.py` through `app_condition_5.py`: Deployment entry points for each condition

## Configuration

### Experimental Condition Variables (Primary)
- `HICXAI_EXPLANATION`: `none` | `counterfactual` | `feature_importance` (determines which XAI method to show)
- `HICXAI_ANTHRO`: `low` | `high` (determines assistant personality and interface style)

### Legacy Variables (Backward Compatibility)
- `HICXAI_VERSION`: `v0` or `v1` (maps to specific factor combinations)
  - `v0` → `explanation=none`, `anthro=low`
  - `v1` → `explanation=feature_importance`, `anthro=high`
- CLI flags also supported: `--explanation=counterfactual --anthro=high` or `--v0` / `--v1`

### Optional Features
- `OPENAI_API_KEY`: enables optional generative rewriter for more natural explanations
- `HICXAI_GENAI`: `on` (default) or `off` to disable the rewriter
- `HICXAI_OPENAI_MODEL`: defaults to `gpt-4o-mini`
- `HICXAI_STYLE`: `short` | `detailed` | `actionable` (explanation tone, configurable in high-anthro conditions)
- `HICXAI_OPENAI_BASE_URL` or `OPENAI_BASE_URL`: optional proxy/base URL for OpenAI SDK
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`: optional workaround for protobuf issues

## Data & Assets

Tracked data and assets for out-of-the-box operation:
- `data/adult.data`
- `dataset_info/adult.json`
- `data_questions/Median_4.csv`
- `assets/luna_avatar.png` (avatar used in v1). The app also checks `data_questions/*.png` and `images/assistant_avatar.png` if present.

## Troubleshooting

- FileNotFoundError for `data/adult.data` on Cloud: ensure dataset files are tracked (this repo includes them).
- Requirements parse errors (InvalidMarker / inline comments): avoid inline comments in `requirements.txt` version lines.
- Graphviz/dtreeviz issues: `packages.txt` includes `graphviz` for apt; Cloud installs it automatically.
- Protobuf errors: set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` in the environment.

## Extending the Platform
- Add new questions to `data_questions/Median_4.csv` (no retraining needed)
- Extend XAI methods in `src/xai_methods.py` 
- Add new intent mappings in `src/constraints.py`
- Modify A/B testing variants in `src/ab_config.py`
- Customize UI themes in `.streamlit/config.toml`

## Dependencies
- **Streamlit Cloud**: `requirements.txt` only (CPU wheels, Python 3.13 compatible). `packages.txt` installs apt `graphviz`.
- **Local GPU dev (optional)**: `environment_rtx5070.yml` + `requirements_rtx5070_torch.txt` for CUDA workflows.
- Core packages: streamlit, pandas, scikit-learn, shap, dice-ml, anchor-exp, dtreeviz, graphviz, openai (optional)

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

Enable a friendlier, style-controlled paraphrase of explanations. The default model is `gpt-4o-mini`.

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



