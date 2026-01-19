import torch
if __name__ == '__main__':
    from robot_forward_kinematics import forward_kinematics_torch
else:
    from .robot_forward_kinematics import forward_kinematics_torch
import torch

def expand_mechanism(para_data: torch.Tensor,
                     dh_params: torch.Tensor,
                     tool_offset: torch.Tensor,
                     offsets=[-2.0, -1.0, 1.0, 2.0],
                     robot_type=''):
    batch_size, n = para_data.shape
    num_offsets = len(offsets)
    offsets_tensor = torch.tensor(offsets, dtype=para_data.dtype, device=para_data.device)  # [len(offsets)]

    expanded_data = para_data.unsqueeze(1).expand(batch_size, n * num_offsets, n).clone()  # [batch, n * len(offsets), n]

    joint_indices = torch.arange(n, device=para_data.device).repeat_interleave(num_offsets)  # [n * len(offsets)]

    offset_values = offsets_tensor.repeat(n)  # [n * len(offsets)]

    joint_indices = joint_indices.unsqueeze(0).expand(batch_size, -1)  # [batch, n * len(offsets)]
    offset_values = offset_values.unsqueeze(0).expand(batch_size, -1)  # [batch, n * len(offsets)]

    original_values = para_data.gather(1, joint_indices)  # [batch, n * len(offsets)]
    new_values = original_values + offset_values  # [batch, n * len(offsets)]

    expanded_data.scatter_(2, joint_indices.unsqueeze(-1), new_values.unsqueeze(-1))

    original_data = para_data.unsqueeze(1)  # [batch, 1, n]
    expanded_data_with_original = torch.cat([original_data, expanded_data], dim=1)  # [batch, len(offset) * n + 1, n]

    expanded_data_with_original = expanded_data_with_original.view(batch_size * (num_offsets * n + 1), n)  # [batch * (len(offset) * n + 1), n]

    expanded_data_rad = torch.deg2rad(expanded_data_with_original)
    positions, rpys = forward_kinematics_torch(expanded_data_rad, dh_params, tool_offset)

    expanded_positions = positions.view(batch_size, num_offsets * n + 1, 3)
    expanded_rpys = rpys.view(batch_size, num_offsets * n + 1, 3)

    return expanded_positions.to(para_data.device), expanded_rpys.to(para_data.device)
