import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, 
                 d_model, 
                 max_len=5000, 
                 dropout=0.1, 
                 init_method="normal", 
                 scale=0.02, 
                 use_cls=True,
                 use_dropout=True):
        super(PositionalEncoding, self).__init__()
        
        self.use_cls = use_cls
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=dropout)
        if self.use_cls:
            self.positional_encoding = nn.Parameter(torch.zeros(1, max_len + 1, d_model), requires_grad=True)  # +1 for CLS
        else:
            self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=True)

        if init_method == "normal":
            nn.init.normal_(self.positional_encoding, mean=0, std=scale)
        elif init_method == "uniform":
            nn.init.uniform_(self.positional_encoding, a=-scale, b=scale)
        else:
            raise ValueError(f"Unsupported initialization method: {init_method}")

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        if self.use_cls:

            cls_encoding = self.positional_encoding[:, 0:1, :]  # [1, 1, d_model]
            positional_encodings = self.positional_encoding[:, 1:seq_len + 1, :]  # [1, seq_len, d_model]
            x_with_cls = torch.cat([cls_encoding.expand(batch_size, -1, -1), x], dim=1)  # [batch_size, seq_len + 1, d_model]
            x_with_cls = x_with_cls + torch.cat([cls_encoding, positional_encodings], dim=1)[:, :seq_len + 1, :]
            return self.dropout(x_with_cls) if self.use_dropout else x_with_cls
        else:
            positional_encodings = self.positional_encoding[:, :seq_len, :]  # [1, seq_len, d_model]
            x_with_pos = x + positional_encodings
            return self.dropout(x_with_pos) if self.use_dropout else x_with_pos


if __name__ == '__main__':
    d_model = 64
    max_len = 10
    batch_size = 2
    seq_len = 5
    model_with_cls = PositionalEncoding(d_model=d_model, max_len=max_len, use_cls=True)
    input_data = torch.randn(batch_size, seq_len, d_model)
    output_with_cls = model_with_cls(input_data)
    print("Output with CLS shape:", output_with_cls.shape)  # [2, 6, 64] (seq_len + 1)

    model_without_cls = PositionalEncoding(d_model=d_model, max_len=max_len, use_cls=False)
    input_data = torch.randn(batch_size, seq_len, d_model)
    output_without_cls = model_without_cls(input_data)
    print("Output without CLS shape:", output_without_cls.shape)  # [2, 5, 64] (seq_len)