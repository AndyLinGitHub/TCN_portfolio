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

from tcn import TemporalConvNet, TemporalConvNet2D

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

set_seed(42)

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
    def __init__(self, asset_return, feature_list, normalize_list, configs, model_name, device="cuda"):
        super().__init__(asset_return, configs)
        self.feature_list = feature_list
        self.normalize_list = normalize_list
        self.model_name = model_name
        self.configs = configs
        self.device = device

        self.model = TimeSeriesModel(self.assets_num, self.model_name, self.configs, self.feature_list, self.device)
        self.model = self.model.to(self.device)

        self.hyperparameters_config = self.configs["hyperparameters_config"]
        self.loss_function = loss_function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters_config["lr"], weight_decay=self.hyperparameters_config["weight_decay"])

        

    def get_tensor(self, start, end):
        input_period = self.configs["portfolio_config"]["input_period"]

        tensor_list = []
        for feature, normalize in zip(self.feature_list, self.normalize_list):
            data_list = []
            for i in range(start, end + 1):
                feature_data = feature[self.data_index[i-input_period]:self.data_index[i-1]]
                if normalize[0] == "rank":
                    feature_data = feature_data.rank(axis=normalize[1], pct=True)# - 0.5
                elif normalize[0] == "minmax":
                    min = feature_data.min(axis=normalize[1])
                    max = feature_data.max(axis=normalize[1])
                    feature_data = (feature_data.sub(min, axis=normalize[1]^1)).div(max - min, axis=normalize[1]^1)# - 0.5
                elif normalize[0] == "None":
                    pass
                else:
                    raise NotImplementedError
                
                feature_data = feature_data.values
                data_list.append(feature_data)

            data_list = np.array(data_list, dtype = float)
            data_tensor = torch.tensor(data_list).to(self.device).float()
            tensor_list.append(data_tensor)

        if self.model_name in ["RNN", "LSTM", "GRU", "TCN"]:
            data_tensor = torch.cat(tensor_list, dim=2)
        elif self.model_name == "TCN2D":
            data_tensor = torch.stack(tensor_list, dim=1)

        target_list = []
        for i in range(start, end + 1):
            target = self.asset_return[self.data_index[i]:self.data_index[i]].values
            target_list.append(target)
        target_list = np.array(target_list, dtype = float)
        target_tensor = torch.tensor(target_list).to(self.device).float()

        return data_tensor, target_tensor
        
    def training(self, training_start, training_end, validation_start, validation_end):
        training_start_date = self.data_index[training_start]
        training_end_date = self.data_index[training_end]
        validation_start_date = self.data_index[validation_start]
        validation_end_date = self.data_index[validation_end]

        print("TRAINING: ", training_start_date, training_end_date)
        print("VALIDATION: ", validation_start_date, validation_end_date)
        data_tensor, target_tensor = self.get_tensor(training_start, training_end)
        data_tensor_valid, target_tensor_valid = self.get_tensor(validation_start, validation_end)

        dataset= ReturnDataset(data_tensor, target_tensor)
        dataloader =  DataLoader(dataset, batch_size=self.hyperparameters_config["batch_size"], shuffle=False, num_workers=0)
        dataset_valid= ReturnDataset(data_tensor_valid, target_tensor_valid)
        dataloader_valid =  DataLoader(dataset_valid, batch_size=self.hyperparameters_config["batch_size"], shuffle=False, num_workers=0)
        
        min_loss = np.inf
        early_stop_counter = 0
        loss_list = []
        loss_valid_list = []
        progress_bar = tqdm(range(self.hyperparameters_config["epoch"]))
        for epoch in range(self.hyperparameters_config["epoch"]):
            return_tensor = torch.tensor([]).to(self.device)
            for data_tensor, target_tensor in dataloader:
                outputs = self.model(data_tensor)
                future_return = self.loss_function(outputs, target_tensor)
                return_tensor = torch.cat((return_tensor, future_return), dim=0)

            self.optimizer.zero_grad()
            if self.hyperparameters_config["optimization_target"] == "sharpe":
                mean = return_tensor.mean()
                std = return_tensor.std()
                #skew = torch.mean(torch.pow((return_tensor - mean) / std, 3))
                #sharpe = skew
                sharpe = - mean / std
            elif self.hyperparameters_config["optimization_target"] == "std":
                sharpe = return_tensor.std() * np.sqrt(252)
            elif self.hyperparameters_config["optimization_target"] == "calmar":
                sharpe = torch.cumprod(torch.add(return_tensor, 1), dim=0)
                sharpe = -sharpe[-1] / (1 - sharpe/torch.cummax(sharpe, 0)[0]).max()
            else:
                raise NotImplementedError
            sharpe.backward()
            self.optimizer.step()

            loss_list.append(sharpe.item())

            return_list = []
            with torch.no_grad():
                for data_tensor, target_tensor in dataloader_valid:
                    outputs = self.model(data_tensor)
                    future_return = self.loss_function(outputs, target_tensor)
                    return_list.extend(future_return.detach().cpu().numpy())

            if self.hyperparameters_config["optimization_target"] == "sharpe":
                mean = np.mean(return_list)
                std = np.std(return_list)
                #skew = np.mean(((np.array(return_list) - mean) / std)**3)
                #sharpe = skew
                sharpe = - mean / std

            elif self.hyperparameters_config["optimization_target"] == "std":
                sharpe = np.std(return_list) * np.sqrt(252)
            elif self.hyperparameters_config["optimization_target"] == "calmar":
                series = pd.Series(return_list)
                sharpe = series.add(1).cumprod()
                sharpe = -sharpe.iloc[-1] / (1 - sharpe/sharpe.cummax()).max()
            else:
                raise NotImplementedError
            
            loss_valid_list.append(sharpe)

            if loss_valid_list[-1] < min_loss:
                min_loss = loss_valid_list[-1]
                torch.save(self.model.state_dict(), self.model_name + '_best_params.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.hyperparameters_config["early_stop"]:
                break

            progress_bar.set_description("Epoch: {}, Training Loss: {}, Validation Loss: {}".format(epoch, loss_list[-1], loss_valid_list[-1]))
            progress_bar.update()

        if self.configs["setting"]["plot"]:
            plt.plot(loss_list)
            plt.plot(loss_valid_list)
            plt.show()

    def predict_weight(self, start, end):
        self.model.load_state_dict(torch.load(self.model_name + '_best_params.pth'))
        tensor_list, target_tensor = self.get_tensor(start, end)
        with torch.no_grad():
            weight = self.model(tensor_list).cpu().numpy()

        weight = pd.DataFrame(weight)
        weight.columns = self.data_columns
        weight.index = self.data_index[start:end+1]

        return weight
        
    

def loss_function(outputs, target_batch):
    future_return = (outputs * target_batch.squeeze()).sum(axis=1)

    return future_return

class ReturnDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        """
        items = []
        for feature in self.tensor_list:
            items.append(feature[index])

        items.append(self.target_tensor[index])
        """
        return self.data_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return len(self.target_tensor)

class TimeSeriesModel(nn.Module):
    def __init__(self, assets_num, model_name, configs, feature_list, device):
        super().__init__()
        self.model_name = model_name
        #self.model_list = nn.ModuleList()
        input_size = 0
        for feature in feature_list:
            input_size += len(feature.columns)

        model_structure_config = configs["model_structure_config"][model_name]
        if model_name in ["RNN", "LSTM", "GRU"]:
            model_class = getattr(nn, model_name)
            self.model = model_class(input_size=input_size, **model_structure_config).to(device)
            """
            for feature in feature_list:
                model = model_class(input_size=len(feature.columns), **model_structure_config).to(device)
                self.model_list.append(model)
            """

            out_size = model_structure_config["hidden_size"]
            if "bidirectional" in model_structure_config.keys() and model_structure_config["bidirectional"]:
                out_size *= 2

        elif model_name == "TCN":
            self.model = TemporalConvNet(num_inputs=input_size, **model_structure_config).to(device)
            """
            for feature in feature_list:
                model = TemporalConvNet(num_inputs=len(feature.columns), **model_structure_config).to(device)
                self.model_list.append(model)
            """
            #out_size = model_structure_config["hidden_size"] * configs["portfolio_config"]["input_period"]
            out_size = model_structure_config["hidden_size"]

        elif model_name == "TCN2D":
            self.model = TemporalConvNet2D(num_inputs=len(feature_list), **model_structure_config).to(device)
            out_size = model_structure_config["hidden_size"]*assets_num
        else:
            raise NotImplementedError
        
        

        #self.fc = nn.Linear(out_size*len(feature_list), assets_num)
        self.fc = nn.Linear(out_size, out_size//2)
        self.fc2 = nn.Linear(out_size//2, assets_num)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #outs = []

        if self.model_name in ["RNN", "LSTM", "GRU"]:
            out, _ = self.model(x)
            """
            for x, model in zip(xs, self.model_list):
                out, _ = model(x)
                outs.append(out)
            """
            #out = torch.cat([out[:, -1, :] for out in outs], dim=1)
            out = out[:, -1, :]

        elif self.model_name == "TCN":
            out = self.model(torch.transpose(x, 1, 2))
            out = torch.transpose(out, 1, 2)
            """
            for x, model in zip(xs, self.model_list):
                out = model(torch.transpose(x, 1, 2))
                out = torch.transpose(out, 1, 2)
                outs.append(out)
            """
            #out = torch.cat([out.reshape(out.shape[0], out.shape[-1]*out.shape[-2]) for out in outs], dim=1)
            #out = torch.cat([out.mean(axis=1) for out in outs], dim=1)
            #out = torch.cat([out[:, -1, :] for out in outs], dim=1)
            out = out[:, -1, :]

        elif self.model_name == "TCN2D":
            out = self.model(torch.transpose(x, 2, 3))
            out = torch.transpose(out, 2, 3)
            out = out[:, :, -1, :]
            out = out.flatten(start_dim=1)


        else:
            raise NotImplementedError
        
        out = self.fc(out)
        out = self.tanh(out)
        out = self.fc2(out)
        #out = self.sigmoid(out)
        out = self.tanh(out)
        #out1 = torch.clamp(out, 0, 1)
        #out2 = torch.clamp(out, -1, 0)
        #out1 = self.softmax(out1)
        #out2 = self.softmax(out2)
        out = out / out.abs().sum(axis=1).unsqueeze(1)
        #out = torch.sign(out) * self.softmax(out.abs())
        #out = (out1 - out2) /2

        return out


class RNN(nn.Module):
    def __init__(self, configs, feature_list):
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
    def __init__(self, configs, feature_list):
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
    def __init__(self, configs, feature_list):
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
    def __init__(self, configs, feature_list):
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