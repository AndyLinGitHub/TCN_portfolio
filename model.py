import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
#from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tcn import TemporalConvNet

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#TODO: Add fee, rebalance threshold, rebalance freq
class BaseModel: # equal_weight
    def __init__(self, asset_return, configs):
        self.asset_return = asset_return
        self.data_index = asset_return.index
        self.data_columns = asset_return.columns
        self.total_length = len(asset_return)
        self.assets_num = len(asset_return.columns)
        self.configs = configs

    def predict_weight(self, start, end):
        weight = self.asset_return.copy()[self.data_index[start]:self.data_index[end]]
        weight.iloc[:, :] = 1/self.assets_num

        return weight

    # Update model's parameters
    # Validation set for early stop and record weight
    def training(self, training_start, training_end, validation_start, validation_end):
        """
        training_start_date = self.data_index[training_start]
        training_end_date = self.data_index[training_end]
        validation_start_date = self.data_index[validation_start]
        validation_end_date = self.data_index[validation_end]
        print("TRAINING: ", training_start_date, training_end_date)
        print("VALIDATION: ", validation_start_date, validation_end_date)
        """
        pass

    # Calculate performance
    def cal_performance(self, weight):
        return_data = self.asset_return.loc[weight.index]
        return_data = (return_data*weight).sum(axis=1)
        return return_data

    def walk_forward_backtesting(self):
        input_period = self.configs["portfolio_config"]["input_period"]
        training_period = self.configs["portfolio_config"]["walk_forward"]["training_period"]
        validation_period = self.configs["portfolio_config"]["walk_forward"]["validation_period"]
        testing_period = self.configs["portfolio_config"]["walk_forward"]["testing_period"]

        return_series = []
        max_period = self.total_length - 1
        for i in range(input_period,  max_period, testing_period):
            training_start = i
            training_end = training_start + training_period - 1
            validation_start = training_end + 1
            validation_end = validation_start + validation_period - 1
            testing_start = validation_end + 1
            testing_end = testing_start + testing_period - 1
            if testing_end > max_period - 1:
                testing_end = max_period - 1

            self.training(training_start, training_end, validation_start, validation_end)

            testing_start_date, testing_end_date = self.data_index[testing_start], self.data_index[testing_end]
            print("TESTING: ", testing_start_date, testing_end_date)
            predicted_weight = self.predict_weight(testing_start, testing_end)
            performance = self.cal_performance(predicted_weight)
            return_series.append(performance)

            if testing_end == max_period - 1:
                break

        return_series = pd.concat(return_series)
        return return_series
    
    def vanilla_backtesting(self):
        input_period = self.configs["portfolio_config"]["input_period"]
        training_pct = self.configs["portfolio_config"]["vanilla"]["training_pct"]
        validation_pct = self.configs["portfolio_config"]["vanilla"]["validation_pct"]
        testing_pct = self.configs["portfolio_config"]["vanilla"]["testing_pct"]

        training_start = input_period
        training_end = training_start + int((self.total_length - input_period - 1)*training_pct)
        validation_start = training_end + 1
        validation_end = validation_start + int((self.total_length - input_period - 1)*validation_pct)
        testing_start = validation_end + 1
        testing_end = self.total_length - 1

        self.training(training_start, training_end, validation_start, validation_end)
        predicted_weight = self.predict_weight(training_start, training_end)
        training_performance = self.cal_performance(predicted_weight)
        predicted_weight = self.predict_weight(validation_start, validation_end)
        validation_performance = self.cal_performance(predicted_weight)

        testing_start_date, testing_end_date = self.data_index[testing_start], self.data_index[testing_end]
        print("TESTING: ", testing_start_date, testing_end_date)
        predicted_weight = self.predict_weight(testing_start, testing_end)
        testing_performance = self.cal_performance(predicted_weight)
        
        return training_performance, validation_performance, testing_performance
    


class Markowitz(BaseModel):
    def __init__(self, asset_return, configs):
        super().__init__(asset_return, configs)

    def predict_weight(self, start, end):
        def objective(weight, mean, cov):
            return -weight.dot(mean) / np.sqrt(weight.dot(cov).dot(weight))
        
        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1)] * self.assets_num
        initial_weight = np.ones(self.assets_num) / self.assets_num

        weight_list = []
        date_list = []
        input_period = self.configs["portfolio_config"]["input_period"]
        for i in range(start, end + 1):
            return_data = self.asset_return[self.data_index[i-input_period]:self.data_index[i-1]]
            mean = return_data.mean().values
            cov = return_data.cov().values
            result = minimize(objective, initial_weight, args=(mean, cov), constraints=constraints, bounds=bounds, method="SLSQP")
            weight_list.append(result.x)
            date_list.append(self.data_index[i])

        weight = pd.DataFrame(weight_list)
        weight.columns = self.data_columns
        weight.index = date_list

        return weight
    
class RiskParity(BaseModel):
    def __init__(self, asset_return, configs):
        super().__init__(asset_return, configs)

    def predict_weight(self, start, end):
        def objective(weight, cov):
            sigma = np.sqrt(weight.dot(cov).dot(weight))
            rc = np.multiply(weight, cov.dot(weight)) / sigma
            rc = rc / sigma

            risk_target = 1 / weight.shape[0]
            error = np.square(rc - risk_target).sum()

            return error
        
        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1)] * self.assets_num
        initial_weight = np.ones(self.assets_num) / self.assets_num

        weight_list = []
        date_list = []
        input_period = self.configs["portfolio_config"]["input_period"]
        for i in range(start, end + 1):
            return_data = self.asset_return[self.data_index[i-input_period]:self.data_index[i-1]]
            cov = return_data.cov().values
            result = minimize(objective, initial_weight, args=(cov), constraints=constraints, bounds=bounds, method="SLSQP")
            weight_list.append(result.x)
            date_list.append(self.data_index[i])

        weight = pd.DataFrame(weight_list)
        weight.columns = self.data_columns
        weight.index = date_list

        return weight

class NNModel(BaseModel):
    def __init__(self, asset_return, benchmark_return, init_weight, configs, model, device="cuda"):
        super().__init__(asset_return, configs)
        self.benchmark_return = benchmark_return
        self.init_weight = init_weight
        self.model_name = model
        self.configs = configs

        if self.model_name == "RNN":
            self.model = RNN(self.configs, self.assets_num)
        elif self.model_name == "LSTM":
            self.model = LSTM(self.configs, self.assets_num)
        elif self.model_name == "GRU":
            self.model = GRU(self.configs, self.assets_num)
        elif self.model_name == "TCN":
            self.model = TCN(self.configs, self.assets_num)
        else:
            raise NotImplementedError
        
        self.device = device
        self.model = self.model.to(self.device)
        self.hyperparameters_config = self.configs["hyperparameters_config"]
        self.optimizer = None

        if not self.hyperparameters_config["init_weight"]:
            self.loss_function = sharpe_loss_function
        else:
            self.loss_function = sharpe_loss_function_init_weight

    def get_data_tensor(self, start, end, data_only=False):
        input_period = self.configs["portfolio_config"]["input_period"]

        data_list = []
        target_list = []
        benchmark_list = []
        init_weight_list = []
        for i in range(start, end + 1):
            return_data = self.asset_return[self.data_index[i-input_period]:self.data_index[i-1]].values
            data_list.append(return_data)

            if not data_only:
                target = self.asset_return[self.data_index[i]:self.data_index[i]].values
                target_benchmark = self.benchmark_return[self.data_index[i]:self.data_index[i]].values
                init_weight = self.init_weight[self.data_index[i-1]:self.data_index[i-1]].values
                target_list.append(target)
                benchmark_list.append(target_benchmark)
                init_weight_list.append(init_weight)


        data_list = np.array(data_list, dtype = float)
        if not data_only:
            target_list = np.array(target_list, dtype = float)
            benchmark_list = np.array(benchmark_list, dtype = float)
            init_weight_list = np.array(init_weight_list, dtype = float)

        data_tensor = torch.tensor(data_list).to(self.device).float()
        if not data_only:
            target_tensor = torch.tensor(target_list).to(self.device).float()
            benchmark_tensor = torch.tensor(benchmark_list).to(self.device).float()
            init_weight_tensor = torch.tensor(init_weight_list).to(self.device).float()
        
        if not data_only:
            return data_tensor, target_tensor, benchmark_tensor, init_weight_tensor
        else:
            return data_tensor
        
    def training(self, training_start, training_end, validation_start, validation_end):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters_config["lr"])

        training_start_date = self.data_index[training_start]
        training_end_date = self.data_index[training_end]
        validation_start_date = self.data_index[validation_start]
        validation_end_date = self.data_index[validation_end]
        input_period = self.configs["portfolio_config"]["input_period"]

        print("TRAINING: ", training_start_date, training_end_date)
        print("VALIDATION: ", validation_start_date, validation_end_date)
        data_tensor, target_tensor, benchmark_tensor, init_weight_tensor = self.get_data_tensor(training_start, training_end)
        data_tensor_valid, target_tensor_valid, benchmark_tensor_valid, init_weight_tensor_valid = self.get_data_tensor(validation_start, validation_end)

        dataset= ReturnDataset(data_tensor, target_tensor, benchmark_tensor, init_weight_tensor)
        dataloader =  DataLoader(dataset, batch_size=self.hyperparameters_config["batch_size"], shuffle=False)
        dataset_valid= ReturnDataset(data_tensor_valid, target_tensor_valid, benchmark_tensor_valid, init_weight_tensor_valid)
        dataloader_valid =  DataLoader(dataset_valid, batch_size=self.hyperparameters_config["batch_size"], shuffle=False)
        
        min_loss = np.inf
        early_stop_counter = 0
        loss_list = []
        loss_valid_list = []
        for epoch in tqdm(range(self.hyperparameters_config["epoch"])):
            return_tensor = torch.tensor([]).to(self.device)
            for batch_idx, (data_batch, target_batch, benchmark_batch, init_weight_benchmark_batch) in enumerate(dataloader):
                outputs = self.model(data_batch)
                loss, future_return = self.loss_function(outputs, target_batch, benchmark_batch, init_weight_benchmark_batch)
                return_tensor = torch.cat((return_tensor, future_return), dim=0)

                #self.optimizer.zero_grad()
                #loss.backward()
                #self.optimizer.step()

            self.optimizer.zero_grad()
            if self.hyperparameters_config["optimization_target"] == "sharpe":
                sharpe = -return_tensor.mean() / return_tensor.std()
            elif self.hyperparameters_config["optimization_target"] == "std":
                sharpe = return_tensor.std()
            else:
                raise NotImplementedError

            #sharpe = -return_tensor.mean()
            sharpe.backward()
            self.optimizer.step()
            loss_list.append(sharpe.item())

            return_list = []
            with torch.no_grad():
                for batch_idx, (data_batch, target_batch, benchmark_batch, init_weight_benchmark_batch) in enumerate(dataloader_valid):
                    outputs = self.model(data_batch)
                    loss, future_return = self.loss_function(outputs, target_batch, benchmark_batch, init_weight_benchmark_batch)
                    return_list.extend(future_return.detach().cpu().numpy())

            if self.hyperparameters_config["optimization_target"] == "sharpe":
                sharpe = -np.mean(return_list) / np.std(return_list)
            elif self.hyperparameters_config["optimization_target"] == "std":
                sharpe = np.std(return_list)
            else:
                raise NotImplementedError
            
            #sharpe = -np.mean(return_list)
            loss_valid_list.append(sharpe)
            if loss_valid_list[-1] < min_loss:
                min_loss = loss_valid_list[-1]
                torch.save(self.model.state_dict(), self.model_name + '_best_params.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.hyperparameters_config["early_stop"]:
                break

            #tqdm.write("Epoch: {}, Training Loss: {}, Validation Loss: {}".format(epoch, loss_list[-1], loss_list_valid[-1]))

        if self.configs["setting"]["plot"]:
            plt.plot(loss_list)
            plt.plot(loss_valid_list)
            plt.show()

    def predict_weight(self, start, end):
        self.model.load_state_dict(torch.load(self.model_name + '_best_params.pth'))
        data_tensor = self.get_data_tensor(start, end, True)
        with torch.no_grad():
            weight = self.model(data_tensor).cpu().numpy()

        weight = pd.DataFrame(weight)
        weight.columns = self.data_columns
        weight.index = self.data_index[start:end+1]

        if self.hyperparameters_config["init_weight"]:
            init_weight = self.init_weight.loc[self.data_index[start-1:end]]
            init_weight.index = self.data_index[start:end+1]
            weight = weight - 1/self.assets_num + init_weight

        return weight
        
    

def sharpe_loss_function(outputs, target_batch, benchmark_batch, init_weight_benchmark_batch):
    future_return = (outputs * target_batch.squeeze()).sum(axis=1)# - benchmark_batch.squeeze()
    sharpe = future_return.mean() / future_return.std()
    #benchmark_return = benchmark_batch.mean(axis=1)
    #alpha = (future_return - benchmark_return).mean()
    #alpha = future_return.mean()

    return -sharpe, future_return

def sharpe_loss_function_init_weight(outputs, target_batch, benchmark_batch, init_weight_batch):
    outputs = outputs - 1/outputs.shape[-1] + init_weight_batch.squeeze()
    future_return = (outputs * target_batch.squeeze()).sum(axis=1)# - benchmark_batch.squeeze()
    sharpe = future_return.mean() / future_return.std()
    #benchmark_return = benchmark_batch.mean(axis=1)
    #alpha = (future_return - benchmark_return).mean()
    #alpha = future_return.mean()

    return -sharpe, future_return

class ReturnDataset(Dataset):
    def __init__(self, data, targets, benchmarks, init_weights):
        self.data = data
        self.targets = targets
        self.benchmarks = benchmarks
        self.init_weights= init_weights

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        z = self.benchmarks[index]
        w = self.init_weights[index]

        return x, y, z, w
    
    def __len__(self):
        return len(self.data)

class RNN(nn.Module):
    def __init__(self, configs, assets_num):
        super().__init__()
        model_structure_config = configs["model_structure_config"]["RNN"]
        self.rnn = nn.RNN(input_size=assets_num, **model_structure_config)

        out_size = model_structure_config["hidden_size"]
        if model_structure_config["bidirectional"]:
            out_size *= 2

        self.fc = nn.Linear(out_size, assets_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)

        return out


class LSTM(nn.Module):
    def __init__(self, configs, assets_num):
        super().__init__()
        model_structure_config = configs["model_structure_config"]["LSTM"]
        self.lstm = nn.LSTM(input_size=assets_num, **model_structure_config)

        out_size = model_structure_config["hidden_size"]
        if model_structure_config["bidirectional"]:
            out_size *= 2

        self.fc = nn.Linear(out_size, assets_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)

        return out

class GRU(nn.Module):
    def __init__(self, configs, assets_num):
        super().__init__()
        model_structure_config = configs["model_structure_config"]["GRU"]
        self.gru = nn.GRU(input_size=assets_num, **model_structure_config)

        out_size = model_structure_config["hidden_size"]
        if model_structure_config["bidirectional"]:
            out_size *= 2

        self.fc = nn.Linear(out_size, assets_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)

        return out

class TCN(nn.Module):
    def __init__(self, configs, assets_num):
        super().__init__()
        model_structure_config = configs["model_structure_config"]["TCN"]
        self.tcn = TemporalConvNet(num_inputs=assets_num, **model_structure_config)

        out_size = model_structure_config["hidden_size"]

        self.fc = nn.Linear(out_size, assets_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = torch.transpose(x, 1, 2)
        out =  self.tcn(out)
        out = torch.transpose(out, 1, 2)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)

        return out