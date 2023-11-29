# TCN_portfolio
## Introduction
This project impement a model using Temporal Convolutional Networks (Bai et al., 2018) for predicting the weight for each asset to maximize the Sharpe
ratio at future timestamps. It is trained directly through the gradient of the Sharpe ratio calculated from the predicted portfolio weights, which
can give the model a sound and robust performance. (Zhang et al., 2020)

## Install project dependencies
```
conda create --name tcn_pf python=3.9
conda activate tcn_pf
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirements.txt
```

## Usage
```
python main.py --setting.save_dir "TCN_result" \
               --portfolio_config.model "TCN" \
               --portfolio_config.input_period 240 \
               --hyperparameters_config.epoch 1024 \
               --hyperparameters_config.lr 0.001 \
               --hyperparameters_config.optimization_target sharpe \
               --model_structure_config.$model.hidden_size 32 \
               --model_structure_config.$model.num_layers 2 \
               --model_structure_config.$model.dropout 0.1 \
               --model_structure_config.$model.kernel_size 4
```
