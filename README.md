# HicXAI Research Platform

A research platform for studying explainable AI in loan decision-making scenarios.

---

## Attribution & Acknowledgments

This work builds upon and adapts components from:

- **[XAgent](https://github.com/bach1292/XAgent)** by bach1292 - Original XAI agent framework and Adult dataset integration
- **Adult Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) via XAgent implementation
- **Question-Intent Dataset** (`data_questions/Median_4.csv`) curated by XAgent, adapted from original work by Liao et al. (2020)

This project builds upon several open-source libraries and frameworks:

- **[Streamlit](https://streamlit.io/)** - Web application framework for the interactive UI
- **[SHAP](https://github.com/slundberg/shap)** - SHapley Additive exPlanations for feature importance visualization
- **[DiCE](https://github.com/interpretml/DiCE)** - Diverse Counterfactual Explanations for generating what-if scenarios
- **[Anchor](https://github.com/marcotcr/anchor)** - High-precision model-agnostic explanations
- **[Scikit-learn](https://scikit-learn.org/)** - Machine learning model development

---

## Dependencies

- **Streamlit Cloud**: `requirements.txt` only (CPU wheels, Python 3.13 compatible). `packages.txt` installs apt `graphviz`.
- **Local GPU dev (optional)**: `environment_rtx5070.yml` + `requirements_rtx5070_torch.txt` for CUDA workflows.
- Core packages: streamlit, pandas, scikit-learn, shap, dice-ml, anchor-exp, dtreeviz, graphviz, openai (optional)

---

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

---

## References

### Primary Sources
- **Liao, Q.V., Gruen, D., & Miller, S.** (2020). Questioning the AI: Informing Design Practices for Explainable AI User Experiences. *CHI '20: CHI Conference on Human Factors in Computing Systems*. [https://arxiv.org/abs/2001.02478](https://arxiv.org/abs/2001.02478)
- **bach1292** - XAgent: Explainable AI Agent Framework. [https://github.com/bach1292/XAGENT](https://github.com/bach1292/XAGENT)

### Dataset Attribution
- **Adult (Census Income) Dataset**: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. [http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml)
- **Question-Intent Pairs** (`Median_4.csv`): Curated by [XAgent](https://github.com/bach1292/XAGENT), adapted from original research by Liao et al. (2020)

---

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{hicxai_research_platform,
  title = {HicXAI Research Platform},
  author = {[Kudzai Sauka]},
  year = {2025},
  url = {https://github.com/ksauka/hicxai-research}
}
```

### Related Publications

*Publications list to be updated upon publication.*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



