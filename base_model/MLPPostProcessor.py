import torch.nn as nn
import torch


class MLPPostProcessor(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dims=[128, 64], 
                 output_dim=1, 
                 normalization="batchnorm", 
                 dropout_rate=0.1):
        super(MLPPostProcessor, self).__init__()

        self.mlp = self._build_mlp(input_dim, hidden_dims, output_dim, normalization, dropout_rate)

    def _build_mlp(self, input_dim, hidden_dims, output_dim, normalization, dropout_rate):
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.mlp(x)  # [batch, output_dim]
        
        return output
    
