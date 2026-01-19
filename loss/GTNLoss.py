import torch
import torch.nn as nn

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset.data_module import DeepONetDataModule

from main_model.GridTransformer import GridTransformer
from scripts.mechanism_expand import expand_mechanism


class GTNLoss(nn.Module):
    def __init__(self,
                 GT_params,
                 model_loss_path,
                 dh_params,
                 tool_offset,
                 mechanism_offsets):
        super(GTNLoss, self).__init__()
        self.GridTransformer = GridTransformer(**GT_params)
        self.GridTransformer.load_state_dict(torch.load(model_loss_path)['state_dict'])
        self.GridTransformer.requires_grad_(False)
        
        self.dh_params = dh_params

        self.tool_offset = tool_offset
        self.mechanism_offsets = mechanism_offsets
        self.loss = nn.MSELoss()
        self.loss2 = nn.L1Loss()

    def forward(self, delta_theta, theta, target):
        new_theta = theta.squeeze(-1) + delta_theta
        if (self.dh_params is not None) and (self.tool_offset is not None):
            self.dh_params = self.dh_params.to(new_theta.device)
            self.tool_offset = self.tool_offset.to(new_theta.device)

        expand_mechanism_data, _ = expand_mechanism(new_theta,
                                                    self.dh_params,
                                                    self.tool_offset,
                                                    self.mechanism_offsets)

        new_theta = new_theta.unsqueeze(-1)

        new_error = self.GridTransformer(expand_mechanism_data, new_theta)
        new_pos = new_error + expand_mechanism_data[:, 0, :]
        loss_model = self.loss(target, new_pos)
        return loss_model
    

    def forward_loss(self, delta_theta, theta, target):
        new_theta = theta.squeeze(-1) + delta_theta
        if (self.dh_params is not None) and (self.tool_offset is not None):
            self.dh_params = self.dh_params.to(new_theta.device)
            self.tool_offset = self.tool_offset.to(new_theta.device)

        expand_mechanism_data, _ = expand_mechanism(new_theta,
                                                    self.dh_params,
                                                    self.tool_offset,
                                                    self.mechanism_offsets)

        new_theta = new_theta.unsqueeze(-1)

        new_error = self.GridTransformer(expand_mechanism_data, new_theta)
        new_pos = new_error + expand_mechanism_data[:, 0, :]
        loss_model = (new_pos - target)
        return loss_model
