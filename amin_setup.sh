#!/bin/bash -i
set -e # exit as soon as any command in the script fails 

# Create a virtual environment with Python 3.10
python3.10 -m venv dnlp_env

# Activate the virtual environment
source dnlp_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Check for CUDA and install the appropriate PyTorch version
if command -v nvidia-smi &>/dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support."
    pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu121
else
    echo "CUDA not detected, installing CPU-only PyTorch."
    pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
fi

# Install additional packages
pip install tqdm==4.66.2 requests==2.31.0 transformers==4.38.2 tensorboard==2.16.2 tokenizers==0.15.1 explainaboard-client==0.1.4 sacrebleu==2.4.0
pip install pandas
pip uninstall --yes numpy
pip install "numpy<2.0"

#pip install ipywidgets


# Other suggestions
# Install Rainbow CSV Extension for VSCode