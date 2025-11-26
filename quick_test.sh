#!/bin/bash
# Quick validation test for Streamlit Cloud readiness
# Tests that all entry points can at least be imported

echo "🔍 Quick Validation Test - Streamlit Cloud Readiness"
echo "===================================================="
echo ""

# Activate conda environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  No conda environment active. Attempting to activate..."
    if conda env list | grep -q "hicxai_rtx5070"; then
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate hicxai_rtx5070
    fi
fi

echo ""

# Check Python version
echo "Python version:"
python --version
echo ""

# Check if required packages are installed
echo "Checking key dependencies..."
python -c "import streamlit; print(f'✓ Streamlit {streamlit.__version__}')" || echo "✗ Streamlit not installed"
python -c "import openai; print(f'✓ OpenAI {openai.__version__}')" || echo "✗ OpenAI not installed"
python -c "import shap; print(f'✓ SHAP {shap.__version__}')" || echo "✗ SHAP not installed"
python -c "import dice_ml; print('✓ DiCE-ML installed')" || echo "✗ DiCE-ML not installed"
python -c "import anchor; print('✓ Anchor installed')" || echo "✗ Anchor not installed"

echo ""

# Test that each app entry point can be imported
echo "Validating app entry points..."

for app in app_v0.py app_v1.py app_v2.py app_v3.py app_v4.py app_v5.py; do
    if [ -f "$app" ]; then
        echo -n "  $app: "
        python -c "import sys; sys.path.insert(0, '.'); exec(open('$app').read())" 2>&1 | head -1 || echo "✓ Can load"
    else
        echo "  $app: ✗ File not found"
    fi
done

echo ""

# Check .env file
if [ -f ".env" ]; then
    echo "✓ .env file found"
    if grep -q "OPENAI_API_KEY" .env; then
        echo "  ✓ OPENAI_API_KEY present"
    else
        echo "  ✗ OPENAI_API_KEY missing"
    fi
else
    echo "⚠️  .env file not found (OK for Streamlit Cloud - use Secrets instead)"
fi

echo ""

# Check config files
echo "Checking configuration..."
if [ -f ".streamlit/config.toml" ]; then
    echo "✓ .streamlit/config.toml exists"
else
    echo "✗ .streamlit/config.toml missing"
fi

if [ -f "requirements.txt" ]; then
    echo "✓ requirements.txt exists"
else
    echo "✗ requirements.txt missing"
fi

echo ""
echo "===================================================="
echo "✓ Validation complete!"
echo ""
echo "Next steps:"
echo "1. Fix any issues marked with ✗ above"
echo "2. Test locally: streamlit run app_v0.py"
echo "3. Deploy to Streamlit Cloud (see DEPLOYMENT.md)"
echo ""
