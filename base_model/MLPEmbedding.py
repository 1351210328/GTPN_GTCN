import torch
import torch.nn as nn

from base_model.PosEncode import PositionalEncoding


class MLPEmbedding(nn.Module):
    def __init__(self,
                 num_inputs=6,
                 input_dim=1,
                 hidden_dims=[128, 128],
                 output_dim=64,
                 normalization="batchnorm",
                 dropout_rate=0.2,
                 use_positional_encoding=False,
                 positional_encoding_kwargs=None):
        super(MLPEmbedding, self).__init__()
        self.mlp = self._build_mlp(input_dim, hidden_dims, output_dim, normalization, dropout_rate)

        self.num_inputs = num_inputs
        self.input_dim = input_dim

        if use_positional_encoding:
            if positional_encoding_kwargs is None:
                positional_encoding_kwargs = {}
            self.positional_encoding = PositionalEncoding(d_model=output_dim, **positional_encoding_kwargs)
        else:
            self.positional_encoding = None

    def _build_mlp(self, input_dim, hidden_dims, output_dim, normalization, dropout_rate):
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))

            if normalization == "batchnorm" and hidden_dim > 1:
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
        batch_size, n, dim = x.size()
        if n != self.num_inputs or dim != self.input_dim:
            raise ValueError(f"Input shape mismatch: expected [batch, {self.num_inputs}, {self.input_dim}], but got [batch, {n}, {dim}].")
        x_reshaped = x.reshape(batch_size * n, dim)  # [batch * n, dim]
        embedded_reshaped = self.mlp(x_reshaped)  # [batch * n, output_dim]
        embedded = embedded_reshaped.view(batch_size, n, -1)  # [batch, n, output_dim]
        if self.positional_encoding is not None:
            embedded = self.positional_encoding(embedded)
        
        return embedded

