import torch


def dh_transform_torch(d, theta, a, alpha):
    d = torch.as_tensor(d,device=d.device)
    theta = torch.as_tensor(theta, device=theta.device)
    a = torch.as_tensor(a, device=a.device)
    alpha = torch.as_tensor(alpha, device=alpha.device)

    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    T = torch.zeros((*theta.shape, 4, 4), device=theta.device)
    T[..., 0, 0] = ct
    T[..., 0, 1] = -st * ca
    T[..., 0, 2] = st * sa
    T[..., 0, 3] = a * ct
    T[..., 1, 0] = st
    T[..., 1, 1] = ct * ca
    T[..., 1, 2] = -ct * sa
    T[..., 1, 3] = a * st
    T[..., 2, 1] = sa
    T[..., 2, 2] = ca
    T[..., 2, 3] = d
    T[..., 3, 3] = 1.0

    return T


def rotation_matrix_to_rpy_torch(R):
    roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    pitch = torch.atan2(-R[:, 2, 0], torch.sqrt(R[:, 2, 1]**2 + R[:, 2, 2]**2))
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    return torch.stack([roll, pitch, yaw], dim=1)


def forward_kinematics_torch(q, dh_params: torch.Tensor, tool_offset: torch.Tensor):
    batch_size, num_joints = q.shape
    T = torch.eye(4, device=q.device).expand(batch_size, 4, 4)  # (batch_size, 4, 4)
    
    for i, (d, offset, a, alpha) in enumerate(dh_params):
        theta = q[:, i] + offset
        T_joint = dh_transform_torch(d, theta, a, alpha)  # (batch_size, 4, 4)
        T = torch.matmul(T, T_joint)

    tool_offset = tool_offset.to(q.device)
    T_end = torch.matmul(T, tool_offset.unsqueeze(0))  # (batch_size, 4, 4)

    position = T_end[:, :3, 3]  # (batch_size, 3)
    R = T_end[:, :3, :3]  # (batch_size, 3, 3)

    rpy = rotation_matrix_to_rpy_torch(R)  # (batch_size, 3)

    return position, rpy

