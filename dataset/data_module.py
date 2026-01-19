import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset.dataset import NetDataset


class NetDataModule(pl.LightningDataModule):
    def __init__(self, file_path, dh_params, tool_offset, batch_size=16, splits=(0.8, 0.1, 0.1)):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.splits = splits
        self.dh_params = dh_params
        self.tool_offset = tool_offset

    def setup(self, stage=None):
        dataset = NetDataset(self.file_path,
                                  self.dh_params,
                                  self.tool_offset)
        n = len(dataset)

        train_size = int(n * self.splits[0])
        val_size = int(n * self.splits[1])
        test_size = n - train_size - val_size

        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
    
    def get_train_size(self):
        return len(self.train_ds) if hasattr(self, 'train_ds') else 0
    
    def get_val_size(self):
        return len(self.val_ds) if hasattr(self, 'val_ds') else 0
    
    def get_test_size(self):
        return len(self.test_ds) if hasattr(self, 'test_ds') else 0
    
    def get_predict_size(self):
        return len(self.test_ds) if hasattr(self, 'test_ds') else 0
