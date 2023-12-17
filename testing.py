import os
import random
import pickle
import importlib
import argparse
import sys

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch

plt.rcParams.update({'font.size': 20})

import config
import model
import utility

if utility.is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

configs = config.configs

#load data
binance_swaps = pd.read_csv(os.path.join("data", "binance_swaps.csv"), index_col=0)
binance_swaps_return = (binance_swaps/binance_swaps.shift(1) - 1)[1:]
binance_swaps_return.index = pd.to_datetime(binance_swaps_return.index)
with open(os.path.join("data", "weekly_crypto_top20.pkl"), 'rb') as f:
    weekly_crypto_top20 = pickle.load(f)

#Prepare training, validation, and testing periods and tickers for each week.
stables = {"BUSD", "USDC", "UST", "DAI", "TUSD", "USDT"}
cexs = {"FTT", "BNB", "HT", "OKB", "CRO", "LEO"}
not_listed = {"WBTC", "TON"}
name_changed = {"SHIB": "1000SHIB"}
remove = stables | cexs | not_listed
setting_list = []
start_date = "2023-01-01"

for k, v in weekly_crypto_top20.items():
    valid_tickers = set(v) - remove
    t = pd.to_datetime(k) + pd.Timedelta("1min")
    if t <= pd.to_datetime(start_date):
        continue
    listed = list(binance_swaps_return.loc[t].dropna().index) # Check whether a swap is listed on Binance at the start of the week.
    
    valid_tickers_top10 = []
    for ticker in v:
        if ticker in valid_tickers and ticker in listed:
            if ticker in name_changed.keys():
                valid_tickers_top10.append(name_changed[ticker])
            else:
                valid_tickers_top10.append(ticker)
    valid_tickers_top10 = valid_tickers_top10[:10]

    testing_period = (t, t +  pd.Timedelta("10079min"))
    validation_period = (t - pd.Timedelta("10080min"), t - pd.Timedelta("1min"))
    training_period = (t - pd.Timedelta("40320min"), t - pd.Timedelta("10081min"))
    
    setting_list.append([training_period, validation_period, testing_period, valid_tickers_top10])

tcn_result_dir = sys.argv[1]
MPT_result_dir =sys.argv[2]

input_length = configs["portfolio_config"]["input_length"]
input_td = pd.Timedelta(f"{input_length}min")
testing_return_series_tcn = pd.Series()
testing_return_series_mpt = pd.Series()
testing_return_series_rp = pd.Series()

print("Start Testing")
for setting in tqdm(setting_list):
    #Prepare data for each week
    training_period, validation_period, testing_period, tickers = setting
    training_return = binance_swaps_return[tickers].loc[training_period[0]-input_td:training_period[1]].fillna(0)
    validation_return = binance_swaps_return[tickers].loc[validation_period[0]-input_td:validation_period[1]].fillna(0)
    testing_return = binance_swaps_return[tickers].loc[testing_period[0]-input_td:testing_period[1]].fillna(0)
    data_list = [training_return, validation_return]

    tcn_model_dir = os.path.join(tcn_result_dir, testing_period[0].isoformat().replace(":", "") + "_" + testing_period[1].isoformat().replace(":", ""), "TCN_best_set_params.pth")
    mpt_dir = os.path.join(MPT_result_dir, testing_period[0].isoformat().replace(":", "") + "_" + testing_period[1].isoformat().replace(":", ""), "training_result.pkl")

    nn_model = model.NNModel(data_list, configs, "TCN")
    weight_tcn = nn_model.predict_weight(testing_return, tcn_model_dir)
    
    with open(mpt_dir, "rb") as input_file:
        mpt_weight = pickle.load(input_file)
    
    testing_return_tcn = (weight_tcn*testing_return.iloc[input_length:]).sum(axis=1)
    testing_return_series_tcn = pd.concat([testing_return_series_tcn, testing_return_tcn])

    testing_return_mpt = testing_return.mul(mpt_weight).sum(axis=1)
    testing_return_series_mpt = pd.concat([testing_return_series_mpt, testing_return_mpt])

tcn_sharpe = testing_return_series_tcn.mean() / testing_return_series_tcn.std()
mpt_sharpe = testing_return_series_mpt.mean() / testing_return_series_mpt.std()
print(f"Sharpe Ratio: {tcn_sharpe} (TCN), {mpt_sharpe} (MPT)")


fig0, ax0 = plt.subplots(figsize=(16, 9))
testing_return_series_tcn.add(1).cumprod().plot(label="TCN", ax=ax0, color="r")
testing_return_series_mpt.add(1).cumprod().plot(label="MPT", ax=ax0, color="b")
ax0.legend()
ax0.set_title("Cumulative Return")
ax0.set_xlabel("Date")
ax0.set_ylabel("Return")
plt.savefig("cumulative_return.png")
#plt.show()

tl_lists = []
vl_lists = []

for setting in setting_list:
    training_period, validation_period, testing_period, tickers = setting
    tcn_model_dir = os.path.join(tcn_result_dir, testing_period[0].isoformat().replace(":", "") + "_" + testing_period[1].isoformat().replace(":", ""), "TCN_best_set_params.pth")
    checkpoint = torch.load(tcn_model_dir)
    tl_lists.append(checkpoint["train_loss_hist"])
    vl_lists.append(checkpoint["valid_loss_hist"])

df = pd.DataFrame(tl_lists).T
df.index.name = "Epoch"
df.columns.name = "Interval"
df = df.stack()
df.name = "Training Loss"
df = pd.DataFrame(df).reset_index()

df2 = pd.DataFrame(vl_lists).T
df2.index.name = "Epoch"
df2.columns.name = "Interval"
df2 = df2.stack()
df2.name = "Validation Loss"
df2 = pd.DataFrame(df2).reset_index()

fig0, ax0 = plt.subplots(figsize=(16, 9))
ax1 = ax0.twinx()

sns.lineplot(data=df, x="Epoch", y="Training Loss", ax=ax0, color="blue", label="Training")
sns.lineplot(data=df2, x="Epoch", y="Validation Loss", ax=ax1, color="red", label="Validation")

ax0.get_legend().remove()
ax1.get_legend().remove()
fig0.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax0.transAxes)
ax0.set_title("Average Learning Curve")
plt.savefig("learning_curve.png")
#plt.show()