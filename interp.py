import torch
import torch.nn as nn


class _Volume4dImpl(nn.Module):
    def __init__(self, data, domain_half_size):
        super().__init__()

        self.data = torch.as_tensor(data)
        self.domain_half_size = domain_half_size

        if self.data.ndim == 4:
            self.data.unsqueeze_(-1)


    def forward(self, xs):
        shape = self.data.shape
        xs = xs.view(-1, 4)

        xs = 0.5 * xs / self.domain_half_size + 0.5
        xs = xs * torch.tensor([shape[0], shape[1], shape[2], shape[3]], device=xs.device).float()
        indices = xs.long()
        lerpw = xs - indices.float()

        x0 = indices[:, 0].clamp(min=0, max=shape[0]-1)
        y0 = indices[:, 1].clamp(min=0, max=shape[1]-1)
        z0 = indices[:, 2].clamp(min=0, max=shape[2]-1)
        t0 = indices[:, 3].clamp(min=0, max=shape[3]-1)
        x1 = (x0 + 1).clamp(max=shape[0]-1)
        y1 = (y0 + 1).clamp(max=shape[1]-1)
        z1 = (z0 + 1).clamp(max=shape[2]-1)
        t1 = (t0 + 1).clamp(max=shape[3]-1)

        return (
            self.data[x0, y0, z0, t0] * (1.0 - lerpw[:, 0:1]) * (1.0 - lerpw[:, 1:2]) * (1.0 - lerpw[:, 2:3]) * (1 - lerpw[:, 3:4]) +
            self.data[x0, y0, z0, t1] * (1.0 - lerpw[:, 0:1]) * (1.0 - lerpw[:, 1:2]) * (1.0 - lerpw[:, 2:3]) * lerpw[:, 3:4] +
            self.data[x0, y0, z1, t0] * (1.0 - lerpw[:, 0:1]) * (1.0 - lerpw[:, 1:2]) * lerpw[:, 2:3] * (1 - lerpw[:, 3:4]) +
            self.data[x0, y0, z1, t1] * (1.0 - lerpw[:, 0:1]) * (1.0 - lerpw[:, 1:2]) * lerpw[:, 2:3] * lerpw[:, 3:4] +
            self.data[x0, y1, z0, t0] * (1.0 - lerpw[:, 0:1]) * lerpw[:, 1:2] * (1.0 - lerpw[:, 2:3]) * (1 - lerpw[:, 3:4]) +
            self.data[x0, y1, z0, t1] * (1.0 - lerpw[:, 0:1]) * lerpw[:, 1:2] * (1.0 - lerpw[:, 2:3]) * lerpw[:, 3:4] +
            self.data[x0, y1, z1, t0] * (1.0 - lerpw[:, 0:1]) * lerpw[:, 1:2] * lerpw[:, 2:3] * (1 - lerpw[:, 3:4]) +
            self.data[x0, y1, z1, t1] * (1.0 - lerpw[:, 0:1]) * lerpw[:, 1:2] * lerpw[:, 2:3] * lerpw[:, 3:4] +
            self.data[x1, y0, z0, t0] * lerpw[:, 0:1] * (1.0 - lerpw[:, 1:2]) * (1.0 - lerpw[:, 2:3]) * (1 - lerpw[:, 3:4]) +
            self.data[x1, y0, z0, t1] * lerpw[:, 0:1] * (1.0 - lerpw[:, 1:2]) * (1.0 - lerpw[:, 2:3]) * lerpw[:, 3:4] +
            self.data[x1, y0, z1, t0] * lerpw[:, 0:1] * (1.0 - lerpw[:, 1:2]) * lerpw[:, 2:3] * (1 - lerpw[:, 3:4]) +
            self.data[x1, y0, z1, t1] * lerpw[:, 0:1] * (1.0 - lerpw[:, 1:2]) * lerpw[:, 2:3] * lerpw[:, 3:4] +
            self.data[x1, y1, z0, t0] * lerpw[:, 0:1] * lerpw[:, 1:2] * (1.0 - lerpw[:, 2:3]) * (1 - lerpw[:, 3:4]) +
            self.data[x1, y1, z0, t1] * lerpw[:, 0:1] * lerpw[:, 1:2] * (1.0 - lerpw[:, 2:3]) * lerpw[:, 3:4] +
            self.data[x1, y1, z1, t0] * lerpw[:, 0:1] * lerpw[:, 1:2] * lerpw[:, 2:3] * (1 - lerpw[:, 3:4]) +
            self.data[x1, y1, z1, t1] * lerpw[:, 0:1] * lerpw[:, 1:2] * lerpw[:, 2:3] * lerpw[:, 3:4]
        ).squeeze()


class Volume4d(nn.Module):
    def __init__(self, data, domain_half_size=None):
        super().__init__()
        data = torch.as_tensor(data)
        if domain_half_size is None:
            domain_half_size = torch.ones(4)
        else:
            domain_half_size = torch.as_tensor(domain_half_size)
        self.interp_mod = torch.jit.script(_Volume4dImpl(
            data,
            domain_half_size,
        ))


    def forward(self, xs):
        return self.interp_mod(xs)
