#reference: https://github.com/locuslab/TCN
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_layers, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        out_channels = []
        for i in range(num_layers):
            out_channels.append(2**(i))
        out_channels.reverse()
        
        layers = []
        for i in range(num_layers):
            in_channel = num_inputs if i == 0 else out_channels[i-1]
            out_channel = out_channels[i]
            conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1)
            conv.weight.data.normal_(0, 0.1)
            layers += [conv, nn.ReLU(), nn.Dropout(dropout)]
        
        self.tcn = nn.Sequential(*layers)
         
    def forward(self, x):
        return self.tcn(x)