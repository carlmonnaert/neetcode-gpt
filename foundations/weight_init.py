import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt( 2 / (fan_in + fan_out) )
        w = std * torch.randn(fan_out,fan_in)
        return [[round(x.item(),4) for x in l] for l in w]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / fan_in)
        w = std * torch.randn(fan_out,fan_in)
        return [[round(x.item(),4) for x in l] for l in w]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights = []
        for i in range(num_layers):
            f_in = input_dim if i == 0 else hidden_dim
            f_out = hidden_dim
            if init_type == 'xavier':
                std = math.sqrt(2 / (f_in + f_out))
                w = torch.randn(f_out, f_in) * std
            elif init_type == 'kaiming':
                std = math.sqrt(2 / f_in)
                w = torch.randn(f_out, f_in) * std
            else:
                w = torch.randn(f_out, f_in)
            weights.append(w)
        
        x = torch.randn(input_dim)
        std_l = []
        for w in weights:
            x = torch.relu(x @ w.T)
            std_l.append(round(x.std().item(), 2))
        
        return std_l