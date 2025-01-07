# This script defines a class `EMA` (Exponential Moving Average) for maintaining a moving average of model parameters.
# The `EMA` class is used to stabilize training by smoothing the updates to the model's parameters.
# It includes methods to register model parameters, update the moving averages, and retrieve the state dictionary of the moving averages.

import torch

class EMA:
    def __init__(self, module: torch.nn.Module, mu: float=0.999):
        self.mu = mu
        self.shadow = {}
        self.register(module)

    def register(self, module: torch.nn.Module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.shadow["pe"] = module.pe # add positional encodings to EMA
        
    def update(self, module: torch.nn.Module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1-self.mu)*param.data + self.mu*self.shadow[name].data
    
    def state_dict(self):
        return self.shadow


    