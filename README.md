# TCN_portfolio
## Introduction
This project impement a model using Temporal Convolutional Networks (Bai et al., 2018) for predicting the weight for each asset to maximize the Sharpe
ratio at future timestamps. It is trained directly through the gradient of the Sharpe ratio (Zhang et al., 2020) calculated from the predicted portfolio weights, which
can give the model a sound and robust performance. 

## Install project dependencies
```
conda create --name tcn_pf python=3.9
conda activate tcn_pf
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

## Usage
```
python main.py
```
