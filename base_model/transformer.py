import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self,
                 dim1,
                 dim2,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim1,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=dim2,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer,
                                             num_layers=num_decoder_layers)
        
    def forward(self, src, tgt):
        # src: [batch_size, seq1, dim1]
        # tgt: [batch_size, seq2, dim2]
        # No need to transpose when batch_first=True
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output


if __name__ == '__main__':
    batch_size = 32
    seq1 = 10
    seq2 = 20
    dim1 = 256
    dim2 = 256
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 128
    
    
    model = TransformerModel(dim1, dim2, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
    
    src = torch.rand(batch_size, seq1, dim1)
    tgt = torch.rand(batch_size, seq2, dim2)
    
    output = model(src, tgt)
    print(output.shape)  # Expected output: [batch_size, seq2, dim1]
