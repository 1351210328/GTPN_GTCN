import pytorch_lightning as pl

from base_model.transformer import TransformerModel
from base_model.MLPEmbedding import MLPEmbedding
from base_model.MLPPostProcessor import MLPPostProcessor
from scripts.mechanism_expand import expand_mechanism
import torch.nn as nn
import torch

import numpy as np
import os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class GridTransformer(pl.LightningModule):
    def __init__(self,
                 mechanism_num_inputs=6 * 4 + 1,
                 para_num_inputs=6,
                 mechanism_input_dim=3,
                 para_input_dim=1,
                 mechanism_hidden_dims=[128, 128],
                 para_hidden_dims=[128, 128],
                 embed_out_dim=128,
                 normalization="batchnorm",
                 nhead=4,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dropout_rate=0.1,
                 encoder_dim=512,
                 m_pos_enc_max_len=6 * 4 + 1,
                 p_pos_enc_max_len=6,
                 pos_enc_init_method="normal",
                 pos_enc_scale=1.0,
                 use_mechanism_pos_enc=True,
                 use_para_pos_enc=True,
                 dh_params=None,
                 tool_offset=None,
                 mechanism_offsets=[-2.0, -1.0, 1.0, 2.0]
                 ):
        super(GridTransformer, self).__init__()
        self.dh_params = dh_params
        self.tool_offset = tool_offset
        self.mechanism_offsets = mechanism_offsets
        
        # [batch, mechanism_num_inputs, mechanism_input_dim]
        self.mechanism_embed = MLPEmbedding(
            num_inputs=mechanism_num_inputs,
            input_dim=mechanism_input_dim,
            hidden_dims=mechanism_hidden_dims,
            output_dim=embed_out_dim,
            normalization=normalization,
            dropout_rate=dropout_rate,
            use_positional_encoding=use_mechanism_pos_enc,  # if use position encoding
            positional_encoding_kwargs={
                "max_len": m_pos_enc_max_len,
                "dropout": dropout_rate,
                "init_method": pos_enc_init_method,
                "scale": pos_enc_scale,
                "use_cls": False  # not use CLS defaultly
            }
        )
        
        # [batch, para_num_inputs, para_input_dim]
        self.para_embed = MLPEmbedding(
            num_inputs=para_num_inputs,
            input_dim=para_input_dim,
            hidden_dims=para_hidden_dims,
            output_dim=embed_out_dim,
            normalization=normalization,
            dropout_rate=dropout_rate,
            use_positional_encoding=use_para_pos_enc,
            positional_encoding_kwargs={
                "max_len": p_pos_enc_max_len,
                "dropout": dropout_rate,
                "init_method": pos_enc_init_method,
                "scale": pos_enc_scale,
                "use_cls": True
            }
        )
        
        # Transformer
        self.model = TransformerModel(
            dim1=embed_out_dim,
            dim2=embed_out_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=encoder_dim,
            dropout=dropout_rate
        )
        
        self.linear_back = MLPPostProcessor(
            input_dim=embed_out_dim,
            hidden_dims=[512, 256, 32],
            output_dim=3
        )
        
        # loss function
        self.criterion = nn.MSELoss()


    def forward(self, mechanism_data, para_data):
        mechanism_embedded = self.mechanism_embed(mechanism_data)  # [batch, mechanism_num_inputs, embed_out_dim]
        para_embedded = self.para_embed(para_data)  # [batch, para_num_inputs + 1, embed_out_dim]
        transformer_output = self.model(mechanism_embedded, para_embedded)  # [batch, para_num_inputs, embed_out_dim]
        transformer_output = transformer_output[:, 0, :]  # [batch, embed_out_dim]
        output = self.linear_back(transformer_output)  # [batch, para_num_inputs]
        return output  # [batch, para_num_inputs]


    def training_step(self, batch, batch_idx):
        para_data, _, error, _ = batch
        if (self.dh_params is not None) and (self.tool_offset is not None):
            self.dh_params = self.dh_params.to(para_data.device)
            self.tool_offset = self.tool_offset.to(para_data.device)
        expand_mechanism_data, _ = expand_mechanism(para_data,
                                                    self.dh_params,
                                                    self.tool_offset,
                                                    self.mechanism_offsets)
        para_data = para_data.unsqueeze(-1)

        predictions = self(expand_mechanism_data, para_data)

        loss = self.criterion(predictions, error)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        para_data, _, error, _ = batch
        if (self.dh_params is not None) and (self.tool_offset is not None):
            self.dh_params = self.dh_params.to(para_data.device)
            self.tool_offset = self.tool_offset.to(para_data.device)
        expand_mechanism_data, _ = expand_mechanism(para_data,
                                                    self.dh_params,
                                                    self.tool_offset,
                                                    self.mechanism_offsets)
        para_data = para_data.unsqueeze(-1)

        predictions = self(expand_mechanism_data, para_data)

        loss = self.criterion(predictions, error)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def test_step(self, batch, batch_idx):
        para_data, _, error, _ = batch
        if (self.dh_params is not None) and (self.tool_offset is not None):
            self.dh_params = self.dh_params.to(para_data.device)
            self.tool_offset = self.tool_offset.to(para_data.device)
        expand_mechanism_data, _ = expand_mechanism(para_data,
                                                    self.dh_params,
                                                    self.tool_offset,
                                                    self.mechanism_offsets)
        para_data = para_data.unsqueeze(-1)

        predictions = self(expand_mechanism_data, para_data)

        loss = self.criterion(predictions, error)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def predict_step(self, batch, batch_idx):
        para_data, mechanism, error, _ = batch
        if (self.dh_params is not None) and (self.tool_offset is not None):
            self.dh_params = self.dh_params.to(para_data.device)
            self.tool_offset = self.tool_offset.to(para_data.device)
        expand_mechanism_data, _ = expand_mechanism(para_data,
                                                    self.dh_params,
                                                    self.tool_offset,
                                                    self.mechanism_offsets)
        para_data = para_data.unsqueeze(-1)
        predictions = self(expand_mechanism_data, para_data)  # [batch, n]
        
        return predictions, error


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=300,
            eta_min=1e-6
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }

        return [optimizer], [scheduler]


