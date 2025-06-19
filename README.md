# SecuRL

A Reinforcement Learning project for network traffic analysis with multiple ML approaches.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# Train PPO model
python rl/train_ppo.py

# Train and evaluate traditional ML models
python analysis/model_comparison.py
```

## Evaluation & Analysis

```bash
# Evaluate trained PPO model
python analysis/evaluate.py

# Generate SHAP explanations
python analysis/explain_with_shap.py
```

## Project Structure

```
SecuRL/
├── rl/                    # Reinforcement Learning components
├── preprocessing/         # Data preprocessing utilities
├── config/               # Configuration parameters
├── analysis/             # Evaluation and analysis tools
└── data/                 # Dataset directory
```