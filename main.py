import os
import random
import pickle
import importlib
import argparse

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch

import config
import model
import utility

if utility.is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

device = "cuda"
def set_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device == "mps":
        torch.mps.manual_seed(seed)
    else:
        raise NotImplementedError

    np.random.seed(seed)
    random.seed(seed)

set_seed(42)


configs = config.configs
parser = argparse.ArgumentParser(description='Setting configs')
utility.add_arguments(configs, parser)

args, unknown = parser.parse_known_args()
for arg in vars(args):
    value = getattr(args, arg)
    if value is not None:
        keys = arg.split(".")
        utility.nested_dict_update(value, configs, keys, indent=0)

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
name_changed = {}
remove = stables | cexs | not_listed
setting_list = []
start_date = "2024-07-01"
end_date = "2024-12-31"

for k, v in weekly_crypto_top20.items():
    valid_tickers = set(v) - remove
    t = pd.to_datetime(k) + pd.Timedelta("1min")
    listed = list(binance_swaps_return.loc[t].dropna().index) # Check whether a swap is listed on Binance at the start of the week.
    if t <= pd.to_datetime(start_date) or t >= pd.to_datetime(end_date):
        continue
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

# Create folder for saving
utility.create_dir(configs["setting"]["save_dir"])

# TCN
# Create folder for saving model training results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_out_dir_root = os.path.join(configs["setting"]["save_dir"], configs["portfolio_config"]["model"] + f"_{timestamp}")
utility.create_dir(model_out_dir_root)

input_length = configs["portfolio_config"]["input_length"]
model_type = configs["portfolio_config"]["model"]
input_td = pd.Timedelta(f"{input_length}min")
for setting in setting_list:
    #Prepare data for each week
    training_period, validation_period, testing_period, tickers = setting
    training_return = binance_swaps_return[tickers].loc[training_period[0]-input_td:training_period[1]].fillna(0)
    validation_return = binance_swaps_return[tickers].loc[validation_period[0]-input_td:validation_period[1]].fillna(0)
    testing_return = binance_swaps_return[tickers].loc[testing_period[0]-input_td:testing_period[1]].fillna(0)
    data_list = [training_return, validation_return]

    model_out_dir = os.path.join(model_out_dir_root, testing_period[0].isoformat().replace(":", "") + "_" + testing_period[1].isoformat().replace(":", ""))
    utility.create_dir(model_out_dir)
    nn_model = model.NNModel(data_list, configs, model_type, model_out_dir, device)
    nn_model.training()


# Modern Portfolio Theory (MPT)
# Create folder for saving model optimization results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_out_dir_root = os.path.join(configs["setting"]["save_dir"], "MPT" + f"_{timestamp}")
utility.create_dir(model_out_dir_root)

input_length = configs["portfolio_config"]["input_length"]
input_td = pd.Timedelta(f"{input_length}min")
for setting in setting_list:
    #Prepare data for each week
    training_period, validation_period, testing_period, tickers = setting
    training_return = binance_swaps_return[tickers].loc[training_period[0]-input_td:validation_period[1]].fillna(0)
    validation_return = None #Validation is optional for traditional approaches.
    data_list = [training_return, validation_return]

    model_out_dir = os.path.join(model_out_dir_root, testing_period[0].isoformat().replace(":", "") + "_" + testing_period[1].isoformat().replace(":", ""))
    utility.create_dir(model_out_dir)
    mpt_model = model.Markowitz(data_list, configs, model_out_dir)
    mpt_model.training()