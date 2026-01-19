import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from main_model.GridTransformer import GridTransformer
from dataset.data_module import NetDataModule

torch.set_default_dtype(torch.float32)


def train(data_params, training_params, dh_params, tool_offset, model_params):
    logger = TensorBoardLogger("tb_logs", name="GTNet", version="version_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
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
        log_every_n_steps=10,
        callbacks=[lr_monitor, checkpoint_callback]
    )

    net = GridTransformer(**model_params)

    data_module = NetDataModule(file_path=data_params["data_path"],
                                        dh_params=dh_params,
                                        tool_offset=tool_offset,
                                        batch_size=data_params["batch_size"],
                                        splits=(0.8, 0.1, 0.1))

    trainer.fit(net, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
    trainer.predict(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)

data_params = {
    "batch_size": 32,
    "data_path": "data/merge2copy.csv"
}

training_params = {
    "max_epochs": 300,
    "gpus": 1 if torch.cuda.is_available() else 0,
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints"
}

# 配置参数
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

model_params = {
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
    "mechanism_offsets": [-2.0, -1.0, 1.0, 2.0]
}

if __name__ == "__main__":
    pl.seed_everything(721+111)  # 0d000721
    train(data_params, training_params, dh_params, tool_offset, model_params)
    