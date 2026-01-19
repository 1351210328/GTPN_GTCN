import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from base_model.MLPEmbedding import MLPEmbedding
from base_model.MLPPostProcessor import MLPPostProcessor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from scripts.mechanism_expand import expand_mechanism
from loss.GTNLoss import GTNLoss
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class GTCNet(pl.LightningModule):
    def __init__(self,
                 grid_transformer_params,
                 model_loss_path,
                 input_dim_theta=6,
                 input_dim_mechanism=4*6+1,
                 input_dim_real=3,
                 
                 theta_input_dim=1,
                 theta_hidden_dims=[128, 128],
                 mechanism_input_dim=3,
                 mechanism_hidden_dims=[128, 128],
                 real_input_dim=1,
                 real_hidden_dims=[128, 128, 256],
                 embed_out_dim=128,
                 normalization="batchnorm",
                 dropout_rate=0.3,
                
                 use_theta_pos_enc=True,
                 use_mechanism_pos_enc=True,
                 use_real_pos_enc=True,
                 t_pos_enc_max_len=6,
                 m_pos_enc_max_len=6 * 4 + 1,
                 r_pos_enc_max_len=3,
                 pos_enc_init_method="normal",
                 pos_enc_scale=1.0,
                 
                 en_nhead=6,
                 dim_forward_en=256,
                 num_encoder_layers=6,
                 de_nhead=6,
                 dim_forward_de=256,
                 num_decoder_layers=6,
                 
                 output_dim=6,
                 
                 lambda_l=0.05,

                 dh_params=None,
                 tool_offset=None,
                 mechanism_offsets=[-2.0, -1.0, 1.0, 2.0]
                 ):
        super(GTCNet, self).__init__()
        self.save_hyperparameters()
        self.dh_params = dh_params
        self.tool_offset = tool_offset
        self.mechanism_offsets = mechanism_offsets
        self.embedding_theta = MLPEmbedding(
            num_inputs=input_dim_theta,
            input_dim=theta_input_dim,
            hidden_dims=theta_hidden_dims,
            output_dim=embed_out_dim,
            normalization=normalization,
            dropout_rate=dropout_rate,
            use_positional_encoding=use_theta_pos_enc,
            positional_encoding_kwargs={
                "max_len": t_pos_enc_max_len,
                "dropout": dropout_rate,
                "init_method": pos_enc_init_method,
                "scale": pos_enc_scale,
                "use_cls": True
            }
        )
        self.embedding_mechanism = MLPEmbedding(
            num_inputs=input_dim_mechanism,
            input_dim=mechanism_input_dim,
            hidden_dims=mechanism_hidden_dims,
            output_dim=embed_out_dim,
            normalization=normalization,
            dropout_rate=dropout_rate,
            use_positional_encoding=use_mechanism_pos_enc,
            positional_encoding_kwargs={
                "max_len": m_pos_enc_max_len,
                "dropout": dropout_rate,
                "init_method": pos_enc_init_method,
                "scale": pos_enc_scale,
                "use_cls": False
            }
        )
        self.embedding_real = MLPEmbedding(
            num_inputs=input_dim_real,
            input_dim=real_input_dim,
            hidden_dims=real_hidden_dims,
            output_dim=embed_out_dim,
            normalization=normalization,
            dropout_rate=dropout_rate,
            use_positional_encoding=use_real_pos_enc,
            positional_encoding_kwargs={
                "max_len": r_pos_enc_max_len,
                "dropout": dropout_rate,
                "init_method": pos_enc_init_method,
                "scale": pos_enc_scale,
                "use_cls": False
            }
        )

        encoder_layer = TransformerEncoderLayer(d_model=embed_out_dim,
                                                nhead=en_nhead,
                                                dim_feedforward=dim_forward_en,
                                                batch_first=True)
        self.encoder_mechanism = TransformerEncoder(encoder_layer,
                                                    num_layers=num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model=embed_out_dim,
                                                nhead=de_nhead,
                                                dim_feedforward=dim_forward_de,
                                                batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_layers=num_decoder_layers)
        self.output_processor = MLPPostProcessor(input_dim=embed_out_dim,
                                                 hidden_dims=[128, 128],
                                                 output_dim=output_dim,
                                                 normalization=normalization,
                                                 dropout_rate=dropout_rate)
        self.gtn_loss = GTNLoss(
            grid_transformer_params,
            model_loss_path, dh_params,
            tool_offset,
            mechanism_offsets)
        self.gtn_loss.requires_grad_(False) 
        self.lambda_l = lambda_l
        


    def forward(self, theta, target, real_position):
        embedded_theta = self.embedding_theta(theta)
        embedded_mechanism = self.embedding_mechanism(target)
        embedded_real = self.embedding_real(real_position)

        mechanism_output = self.encoder_mechanism(embedded_mechanism)

        encoder_output = torch.cat([embedded_real, mechanism_output], dim=1)  # [batch_size, seq_len, hidden_dim]

        decoder_output = self.decoder(embedded_theta, encoder_output)

        delta_theta = self.output_processor(decoder_output[:, 0, :])  # [batch_size, output_dim]
        return delta_theta


    def training_step(self, batch, batch_idx):
        theta, target, _, real_position = batch
        expand_mechanism_data, _ = expand_mechanism(theta,
                                            self.dh_params.to(theta.device),
                                            self.tool_offset.to(theta.device),
                                            self.mechanism_offsets)
        theta = theta.unsqueeze(-1)
        real_position = real_position.unsqueeze(-1)

        delta_theta = self.forward(theta, expand_mechanism_data, real_position)
        loss_model = self.gtn_loss(delta_theta, theta, target)

        loss_theta = (delta_theta.pow(2)).mean()
        loss = (1.0 - self.lambda_l) * loss_model + self.lambda_l * loss_theta

        self.log('train_model', loss_model, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_theta', loss_theta, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_l', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        theta, target, _, real_position = batch
        expand_mechanism_data, _ = expand_mechanism(theta,
                                            self.dh_params.to(theta.device),
                                            self.tool_offset.to(theta.device),
                                            self.mechanism_offsets)
        theta = theta.unsqueeze(-1)
        real_position = real_position.unsqueeze(-1)
        delta_theta = self.forward(theta, expand_mechanism_data, real_position)
        loss_model = self.gtn_loss(delta_theta, theta, target)
        loss_theta = (delta_theta.pow(2)).mean()
        loss = (1.0 - self.lambda_l) * loss_model + self.lambda_l * loss_theta
        self.log('val_model', loss_model, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_theta', loss_theta, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_l', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def test_step(self, batch, batch_idx):
        theta, target, _, real_position = batch
        expand_mechanism_data, _ = expand_mechanism(theta,
                                            self.dh_params.to(theta.device),
                                            self.tool_offset.to(theta.device),
                                            self.mechanism_offsets)
        theta = theta.unsqueeze(-1)
        real_position = real_position.unsqueeze(-1)

        delta_theta = self.forward(theta, expand_mechanism_data, real_position)
        loss_model = self.gtn_loss(delta_theta, theta, target)
        loss_theta = (delta_theta.pow(2)).mean()
        loss = (1.0 - self.lambda_l) * loss_model + self.lambda_l * loss_theta
        self.log('test_model', loss_model, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_theta', loss_theta, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_l', loss, on_epoch=True, prog_bar=True, logger=True)
    
    def predict_step(self, batch, batch_idx):
        theta, target, error, real_position = batch
        expand_mechanism_data, _ = expand_mechanism(theta,
                                            self.dh_params.to(theta.device),
                                            self.tool_offset.to(theta.device),
                                            self.mechanism_offsets)   
        theta = theta.unsqueeze(-1)
        real_position = real_position.unsqueeze(-1)

        delta_theta = self.forward(theta, expand_mechanism_data, real_position)
        loss_model = self.gtn_loss.forward_loss(delta_theta, theta, target)  # [batch_size, 3]
        loss_theta = (delta_theta.pow(2))  # [batch_size, 6]
        return loss_model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,            # 初始学习率
            weight_decay=1e-4   # 权重衰减（L2 正则化）
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=5e-6)
        return [optimizer], [scheduler]
