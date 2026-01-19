import pandas as pd
import torch
from torch.utils.data import Dataset
from scripts.robot_forward_kinematics import forward_kinematics_torch


class NetDataset(Dataset):
    def __init__(self, file_path, dh_params, tool_offset):
        df = pd.read_csv(file_path, index_col=False)
        # theta (N, 6), mechanism (N, 3), real_position (N, 3)
        self.theta = df[['1', '2', '3', '4', '5', '6']].values
        self.real_position = df[['x-real', 'y-real', 'z-real']].values
        self.theta = torch.tensor(self.theta, dtype=torch.float32)
        
        rad_theta = torch.deg2rad(self.theta)
        cal_mechanism, _ = forward_kinematics_torch(rad_theta, dh_params, tool_offset)

        self.mechanism = torch.tensor(cal_mechanism, dtype=torch.float32)

        self.real_position = torch.tensor(self.real_position, dtype=torch.float32)
        self.error = self.real_position - self.mechanism

    def __len__(self):
        return len(self.real_position)

    def __getitem__(self, idx):
        return self.theta[idx], self.mechanism[idx], self.error[idx], self.real_position[idx]

