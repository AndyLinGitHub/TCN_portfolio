# TCN_portfolio
## Introduction
This project impement a model using Temporal Convolutional Networks (Bai et al., 2018) for predicting the weight for each asset to maximize the Sharpe
ratio at future timestamps. It is trained directly through the gradient of the Sharpe ratio (Zhang et al., 2020) calculated from the predicted portfolio weights, which
can give the model a sound and robust performance. 

## Asset Pool
- Original Pool: Among the top twenty cryptocurrencies by market capitalization, excluding CEX coins, stablecoins, and coins without perpetual contracts on Binance.
- Selected Pool: Top ten cryptocurrencies by market capitalization in the original pool.  
- Pools Update Frequency: Every Monday at 00:00 UTC+0.
- Trading Target: Perpetual contracts on Binance for cryptocurrencies in the selected pool.
- Data Resolution: **1 minute**

## Backtesting Method
- Rolling basis (walk forward optimization)
- Training Period: t-40320 ~ t-10081 (30240 minutes or 21 days in total)
- Validation Period: t-10080 ~ t-1 (10080 minutes or 7 days in total)
- Testing Period: t ~ t+10079 (10080 minutes or 7 days in total)
- t is the start of each week, i.e., 00:01 on Monday.

![image.png](https://github.com/AndyLinGitHub/TCN_portfolio/blob/main/image/bt_flow.png)

## Model Structure
- Given $N$ assets, at timestamp $\tau$, we takes the return time series ($N \times L$) from $\tau$ to $\tau - L + 1$ as the model input, uses the model to predict the optimal asset weight $w_{\tau}$ ($N \times 1$), and holds or shorts the assets with this weight to timestamp $\tau + 1$.
- Here, we focus on having a long-short portfolio that is dollar-neutral. Therefore, the weight predicted needs to satisfy the following constraint:
    - ${\displaystyle\sum_{i=1}^{N}} |{w_{\tau i}|} = 1, -1 \le w_{\tau i} \le 1$
    - $\sum_{i=1}^{N} \max(w_{\tau i}, 0) + \sum_{i=1}^{N} \min(w_{\tau i}, 0) = 0$

![image.png](https://github.com/AndyLinGitHub/TCN_portfolio/blob/main/image/model.png)

## Loss Function
- The model aims to predict the weight of each crypto swap to maximize the Sharpe ratio at future timestamps. The loss function for updating model for this purpose is defined as follows:
    - $R_{t} = \sum_{i = 0}^{N-1}w_{i(t-1)}R_{it}$
    - $E(R) = \frac{1}{T}\sum_{t=1}^{T}R_t$
    - $Std(R) = \frac{1}{T-1}\sum_{t=1}^{T}[R_t - E(R)]^2$
    - $Loss = -\frac{E(R)}{Std(R)}$

## Training Procedure
- For a given hyperparameter set, the model with the best validation loss will be saved during training.
- The hyperparameters are also tuned based on the best validation loss.

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
