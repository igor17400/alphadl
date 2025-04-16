# alphadl - Lightning AI for Cutting-Edge Deep Learning Models in Stock Markets

alphadl is a project that leverages Lightning AI to implement and compare various deep learning models for stock market prediction.

## Models

- LSTM
- XGBoost
- CatBoost
- MLP

## Data interface

- OpenBB

## Features

- Utilizes Lightning AI for efficient and scalable deep learning
- Implements ML/DL models for stock market prediction
- Uses OpenBB for stock market data
- Employs Hydra for configuration management
- Integrates Weights & Biases (wandb) for experiment tracking and visualization

## Installation

1. Clone the repository:

```bash
git clone https://github.com/igor17400/alphadl.git
cd alphadl
```

2. Configure Poetry to create virtual environment in project directory:
```bash
poetry config virtualenvs.in-project true
```

3. Install dependencies and create virtual environment:
```bash
# Create virtual environment and install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

4. Set up pre-commit hooks:
```bash
poetry run pre-commit install
```

5. You might need to install PyTorch with CUDA support using the following command to ensure GPU compatibility:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

P.S.: Be careful with your cuda version. 

To test if the installation worked correctly, run the following command:

```bash
python test_pytorch.py
```

This will verify that both PyTorch and Lightning AI are properly installed with GPU support.

## Contributions

If you're interested in contributing to alphadl, here are some steps to get you started:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Make sure to follow the existing code style and formatting conventions.
3. Write unit tests for your changes to ensure they are working as expected.
4. Submit a pull request with a clear description of your changes.

We appreciate all kinds of contributions, whether it's a bug fix, a new feature, or documentation improvements. If you're unsure where to start, feel free to reach out to us on our issue tracker or discussion forum.


