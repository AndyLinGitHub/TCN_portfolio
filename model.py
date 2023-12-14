import random
import os
import itertools
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utility
from tcn import TCN

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
    
class BaseModel: 
    def __init__(self, returns, configs, save_dir=""):
        self.training_return, self.validation_return = returns
        self.configs = configs
        self.save_dir = save_dir

        self.input_period = self.configs["portfolio_config"]["input_period"]

        self.assets_num = self.training_return.shape[1]

    def predict_weight(self, input): # equal_weight
        weight = np.ones((input.shape[0] - self.input_period, self.assets_num)) / self.assets_num
        weight = pd.DataFrame(weight)
        weight.columns = input.columns
        weight.index = input.index[self.input_period:]
        
        return weight

    def training(self):
        pass

class Markowitz(BaseModel):
    def __init__(self, returns, configs, save_dir=""):
        super().__init__(returns, configs, save_dir)
        self.training_result = None

    def objective(self, weight, mean, cov):
        return -weight.dot(mean) / np.sqrt(weight.dot(cov).dot(weight))

    def training(self):
        # Setting for minimization.
        constraints = [{"type": "eq", "fun": lambda weights: np.sum(np.abs(weights)) - 1}]
        bounds = [(-1, 1)] * self.assets_num

        #With positive initial weight
        initial_weight = np.ones(self.assets_num) / self.assets_num
        mean = self.training_return.iloc[self.input_period:].mean().values
        cov = self.training_return.iloc[self.input_period:].cov().values
        result_p = minimize(self.objective, initial_weight, args=(mean, cov), constraints=constraints, bounds=bounds, method="SLSQP")

        #With negative initial weight
        initial_weight = -np.ones(self.assets_num) / self.assets_num
        mean = self.training_return.iloc[self.input_period:].mean().values
        cov = self.training_return.iloc[self.input_period:].cov().values
        result_n = minimize(self.objective, initial_weight, args=(mean, cov), constraints=constraints, bounds=bounds, method="SLSQP")

        #With mix initial weight
        initial_weight = (result_p.x + result_n.x)
        initial_weight = initial_weight/np.sum(np.abs(initial_weight))
        mean = self.training_return.iloc[self.input_period:].mean().values
        cov = self.training_return.iloc[self.input_period:].cov().values
        result = minimize(self.objective, initial_weight, args=(mean, cov), constraints=constraints, bounds=bounds, method="SLSQP")

        
        self.training_result = result.x
        with open('training_result.pkl', 'wb') as handle:
            pickle.dump(self.training_result, handle)
        
    def predict_weight(self, input):
        weight = np.tile(self.training_result, (input.shape[0] - self.input_period, 1))
        weight = pd.DataFrame(weight)
        weight.columns = input.columns
        weight.index = input.index[self.input_period:]

        return weight
    
class RiskParity(Markowitz):
    def __init__(self, returns, configs, save_dir=""):
        super().__init__(returns, configs, save_dir)
        self.training_result = None

    def objective(self, weight, mean, cov):
        sigma = np.sqrt(weight.dot(cov).dot(weight))
        rc = np.multiply(weight, cov.dot(weight)) / sigma
        rc = rc / sigma

        risk_target = 1 / weight.shape[0]
        error = np.square(rc - risk_target).sum()

        return error

class NNModel(BaseModel):
    def __init__(self, returns, configs, model_name, save_dir="", device="cuda"):
        super().__init__(returns, configs, save_dir)
        self.model_name = model_name
        self.configs = configs
        self.device = device

        self.hyperparameters_config = self.configs["hyperparameters_config"]
        self.model_structure_config = self.configs["model_structure_configs"][model_name]

        # Get all possible combinations from the given hyperparameter tuning range.
        self.hyperparameters_all_combinations =  self.get_all_combinations(self.hyperparameters_config)
        self.model_structure_all_combinations =  self.get_all_combinations(self.model_structure_config)

        # Prepare datasets
        data_tensor, target_tensor = self.get_tensor(self.training_return)
        data_tensor_valid, target_tensor_valid = self.get_tensor(self.validation_return)
        
        self.dataset = ReturnDataset(data_tensor, target_tensor)
        self.dataset_valid = ReturnDataset(data_tensor_valid, target_tensor_valid)
        

    def get_tensor(self, input):
        feature_list = []
        for i in range(self.input_period, input.shape[0]):
            feature = input.iloc[i - self.input_period:i]
            feature_list.append(feature.values)

        feature_list = np.array(feature_list, dtype = float)
        feature_tensor = torch.tensor(feature_list).to(self.device).float()
        target_tensor = torch.tensor(input.iloc[self.input_period:].values).to(self.device).float()

        return feature_tensor, target_tensor
        
    def training(self):
        print("TRAINING: ", self.training_return.index[self.input_period], "~", self.training_return.index[-1])
        print("VALIDATION: ", self.validation_return.index[self.input_period], "~", self.validation_return.index[-1])
        
        # Iterate over all possible training hyperparameter sets and model structure sets.
        # Select the set pair with the highest validation performance.
        all_combinations = itertools.product(self.hyperparameters_all_combinations, self.model_structure_all_combinations)
        min_loss_ac = np.inf
        for index, (hp_set, ms_set) in enumerate(all_combinations):
            dataloader =  DataLoader(self.dataset, batch_size=hp_set["batch_size"], shuffle=False, num_workers=0)
            dataloader_valid =  DataLoader(self.dataset_valid, batch_size=hp_set["batch_size"], shuffle=False, num_workers=0)

            
            nn_model = TimeSeriesModel(self.assets_num, self.input_period, self.model_name, ms_set, self.device)
            nn_model = nn_model.to(self.device)
            optimizer = torch.optim.Adam(nn_model.parameters(), lr=hp_set["lr"], weight_decay=hp_set["weight_decay"])

            progress_bar = tqdm(range(hp_set["epoch"]))
            min_loss = np.inf
            min_loss_info = None
            train_loss_hist = []
            valid_loss_hist = []
            for epoch in range(hp_set["epoch"]):
                train_loss, valid_loss = self.train_epoch(nn_model, optimizer, dataloader, dataloader_valid)
                progress_bar.set_description("Epoch: {}, Training Loss: {}, Validation Loss: {}".format(epoch, train_loss, valid_loss))
                progress_bar.update()
                train_loss_hist.append(train_loss)
                valid_loss_hist.append(valid_loss)

                if valid_loss < min_loss:
                    min_loss = valid_loss
                    save_info = {'epoch': epoch, 'state_dict': nn_model.state_dict(), 'valid_loss': valid_loss, 'hp_set': hp_set, 'ms_set': ms_set}
                    min_loss_info = save_info
                    torch.save(save_info, os.path.join(self.save_dir, self.model_name + f'_set_{index}_params.pth'))

            if min_loss < min_loss_ac:
                min_loss_ac = min_loss
                save_info = min_loss_info
                save_info["train_loss_hist"] = train_loss_hist
                save_info["valid_loss_hist"] = valid_loss_hist
                self.best_dir = os.path.join(self.save_dir, self.model_name + f'_best_set_params.pth')
                
                torch.save(save_info, self.best_dir)

    def train_epoch(self, nn_model, optimizer, dataloader, dataloader_valid):
        return_tensor = torch.tensor([]).to(self.device)
        for data_tensor, target_tensor in dataloader:
            outputs = nn_model(data_tensor)
            future_return = self.future_return(outputs, target_tensor)
            return_tensor = torch.cat((return_tensor, future_return), dim=0)

        optimizer.zero_grad()
        mean = return_tensor.mean()
        std = return_tensor.std()
        sharpe = -mean / std
        
        sharpe.backward()
        optimizer.step()
        train_loss = sharpe.item()

        return_list = []
        with torch.no_grad():
            for data_tensor, target_tensor in dataloader_valid:
                outputs = nn_model(data_tensor)
                future_return = self.future_return(outputs, target_tensor)
                return_list.extend(future_return.detach().cpu().numpy())

            mean = np.mean(return_list)
            std = np.std(return_list)
            sharpe = -mean / std
                
        valid_loss = sharpe

        return train_loss, valid_loss

    def predict_weight(self, input, ckpt_dir):
        checkpoint = torch.load(ckpt_dir)
        nn_model = TimeSeriesModel(self.assets_num, self.input_period, self.model_name, checkpoint["ms_set"], self.device)
        nn_model = nn_model.to(self.device)
        nn_model.load_state_dict(checkpoint["state_dict"])
        tensor_list, target_tensor = self.get_tensor(input)
        with torch.no_grad():
            weight = nn_model(tensor_list).cpu().numpy()

        weight = pd.DataFrame(weight)
        weight.columns = self.data_columns
        weight.index = self.data_index[start:end+1]

        return weight
        
    

    def future_return(self, outputs, target_batch):
        future_return = (outputs * target_batch.squeeze()).sum(axis=1)
    
        return future_return

    def get_all_combinations(self, d):
        all_combinations = list(itertools.product(*d.values()))
        all_combinations = [dict(zip(d.keys(), combo)) for combo in all_combinations]
    
        return all_combinations

class ReturnDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return len(self.target_tensor)

class TimeSeriesModel(nn.Module):
    def __init__(self, assets_num, input_period, model_name, ms_set, device):
        super().__init__()
        self.model_name = model_name

        if model_name == "TCN":
            self.ts_model = TCN(num_inputs=assets_num, **ms_set).to(device)
            out_size = input_period - (ms_set["kernel_size"] - 1) * ms_set["num_layers"]

        else:
            raise NotImplementedError
        
        self.sq_model = nn.Sequential(nn.Linear(out_size, out_size//2), nn.ReLU(), nn.Linear(out_size//2, assets_num), nn.Softmax(dim=1))
        
    def forward(self, x):
        if self.model_name == "TCN":
            out = self.ts_model(torch.transpose(x, 1, 2))
            out = torch.transpose(out, 1, 2)
            out = torch.squeeze(out)


        else:
            raise NotImplementedError
        
        out = self.sq_model(out)
        out = (out - out.mean(axis=1).unsqueeze(1))
        out = out/out.abs().sum(axis=1).unsqueeze(1)

        return out

