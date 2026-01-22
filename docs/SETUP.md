# Setup Guide

Detailed instructions for setting up the In-Context Representation Influence research environment.

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 16GB VRAM | NVIDIA A40 (48GB) |
| RAM | 32GB | 64GB |
| Storage | 50GB free | 100GB+ (for model caches) |

### Software

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Git

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd in_context_representation_influence
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# OR using conda
conda create -n icl-influence python=3.10
conda activate icl-influence
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: Install as editable package
pip install -e .
```

### 4. Verify Installation

```bash
# Test core imports
python -c "
from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import ClusterSeparation, ContextSensitivityScore
print('All imports successful!')
"

# Test with small model (no GPU required)
python experiments/core/run_experiment.py --model gpt2 --n-contexts 5 --layers 0,5,11
```

## Model Setup

### Downloading Models

Models are downloaded automatically from HuggingFace on first use. To pre-download:

```bash
# Pre-download Llama 3 8B (requires HuggingFace authentication)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'meta-llama/Meta-Llama-3-8B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')
print('Model downloaded successfully!')
"
```

### HuggingFace Authentication

For gated models (like Llama 3), you need to:

1. Create a HuggingFace account at https://huggingface.co
2. Accept the model license at the model page
3. Create an access token at https://huggingface.co/settings/tokens
4. Login via CLI:

```bash
pip install huggingface_hub
huggingface-cli login
# Enter your token when prompted
```

### Model Cache Location

By default, models are cached in `~/.cache/huggingface/hub/`. To change:

```bash
export HF_HOME=/path/to/custom/cache
# or
export TRANSFORMERS_CACHE=/path/to/custom/cache
```

## Optional Components

### Weights & Biases (Experiment Tracking)

```bash
pip install wandb

# Login
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### Dimensionality Reduction

```bash
pip install umap-learn scikit-learn
```

### SGLD Sampling (devinterp)

```bash
pip install devinterp
```

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

1. **Use bfloat16 precision**:
   ```bash
   python experiments/core/run_experiment.py --dtype bfloat16 ...
   ```

2. **Reduce batch size / context length**:
   ```bash
   python experiments/core/run_experiment.py --n-contexts 50 ...
   ```

3. **Extract fewer layers**:
   ```bash
   python experiments/core/run_experiment.py --layers 0,15,31 ...
   ```

### Import Errors

If you see `ModuleNotFoundError`:

1. Ensure you're in the project root directory
2. Verify the virtual environment is activated
3. Try reinstalling:
   ```bash
   pip install -e .
   ```

### Slow Tokenization

First run may be slow due to tokenizer downloads. Pre-cache with:

```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` |
| `WANDB_PROJECT` | Default W&B project name | `icl-influence` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All available |

Example `.env` file:

```bash
HF_HOME=/data/models/huggingface
WANDB_PROJECT=icl-influence
CUDA_VISIBLE_DEVICES=0
```

## Running on Clusters

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=icl-influence
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=64G

source /path/to/venv/bin/activate
cd /path/to/in_context_representation_influence

python experiments/core/run_multilayer_loo_experiment.py \
    --model meta-llama/Meta-Llama-3-8B \
    --dtype bfloat16
```

## Testing Your Setup

Run this verification script:

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import ClusterSeparation

print('All core modules imported successfully!')
print('Setup complete!')
"
```
