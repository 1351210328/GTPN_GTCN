import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from main_model.GTCNet import GTCNet
from dataset.data_module import NetDataModule


def main(data_params, model_params, training_params, dh_params, tool_offset):
    logger = TensorBoardLogger("tb_logs", name="GTCNet", version="version_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_l',
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        filename='{epoch}-{step}',
        save_top_k=1,
        mode='min',
        every_n_train_steps=None,
        verbose=False
    )

    trainer = pl.Trainer(
        max_epochs=training_params["max_epochs"],
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=20,
        callbacks=[lr_monitor, checkpoint_callback],
        enable_progress_bar=True
    )

    net = GTCNet(**model_params)

    data_module = NetDataModule(file_path=data_params["data_path"],
                                        dh_params=dh_params,
                                        tool_offset=tool_offset,
                                        batch_size=data_params["batch_size"],
                                        splits=(0.8, 0.1, 0.1))

    trainer.fit(net, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
    trainer.predict(dataloaders=data_module, ckpt_path=checkpoint_callback.best_model_path)

data_params = {
    "batch_size": 32,
    "data_path": "data/merge2copy.csv"
}

training_params = {
    "max_epochs": 500,
    "gpus": 1 if torch.cuda.is_available() else 0,
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints"
}

def generate_mechanism_offsets(delta):
    return [-2.0 * delta, -1.0 * delta, 1.0 * delta, 2.0 * delta]

delta = 1.0
mechanism_offsets = generate_mechanism_offsets(delta)

dh_params = torch.tensor([[89.159, 0.0,    0.0, torch.pi/2],
                    [   0.0, 0.0, -425.0, 0.0],
                    [   0.0, 0.0,-392.25, 0.0],
                    [109.15, 0.0,    0.0, torch.pi/2],
                    [ 94.65, 0.0,    0.0, -torch.pi/2],
                    [  82.3, 0.0,    0.0, 0.0]])
tool_offset = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

GridTransformer_model_params = {
    "mechanism_num_inputs": 6 * 4 + 1,
    "para_num_inputs": 6,
    "mechanism_input_dim": 3,
    "para_input_dim": 1,
    "mechanism_hidden_dims": [128, 128],
    "para_hidden_dims": [128, 128],
    "embed_out_dim": 128,
    "normalization": "batchnorm",
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dropout_rate": 0.3,
    "encoder_dim": 256,
    "m_pos_enc_max_len": 6 * 4 + 1,
    "p_pos_enc_max_len": 6,
    "pos_enc_init_method": "normal",
    "pos_enc_scale": 1.0,
    "use_mechanism_pos_enc": True,
    "use_para_pos_enc": True,
    "dh_params": dh_params,
    "tool_offset": tool_offset,
    "mechanism_offsets": mechanism_offsets
}

# train GTPN first,then load the pre-trained GTPN weights to train GTCN
model_loss_path = "/checkpoints/GTPN.ckpt"
model_params = {
    "grid_transformer_params": GridTransformer_model_params,
    "model_loss_path": model_loss_path,
    "input_dim_theta": 6,
    "input_dim_mechanism": 4 * 6 + 1,
    "input_dim_real": 3,
    "theta_input_dim": 1,
    "theta_hidden_dims": [128, 128],
    "mechanism_input_dim": 3,
    "mechanism_hidden_dims": [128, 128],
    "real_input_dim": 1,
    "real_hidden_dims": [128, 128, 256],
    "embed_out_dim": 256,
    "normalization": "batchnorm",
    "dropout_rate": 0.1,
    "use_theta_pos_enc": True,
    "use_mechanism_pos_enc": True,
    "use_real_pos_enc": True,
    "t_pos_enc_max_len": 6,
    "m_pos_enc_max_len": 6 * 4 + 1,
    "r_pos_enc_max_len": 3,
    "pos_enc_init_method": "normal",
    "pos_enc_scale": 1.0,
    "en_nhead": 4,
    "dim_forward_en": 256,
    "num_encoder_layers": 4,
    "de_nhead": 4,
    "dim_forward_de": 256,
    "num_decoder_layers": 4,
    "output_dim": 6,
    "lambda_l": 0.05,
    "dh_params": dh_params,
    "tool_offset": tool_offset,
    "mechanism_offsets": mechanism_offsets
}

if __name__ == "__main__":
    pl.seed_everything(721+42)
    main(data_params, model_params, training_params, dh_params, tool_offset)
    