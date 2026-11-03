import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, base=1e4):
        super().__init__()
        self.base = base
    def forward(self, X):
        s, d = X.shape
        if d % 2 != 0:
            raise ValueError('hidden size must be even')
        i = torch.arange(0, d, 2, dtype=torch.float32) # (d/2, )
        theta = 1 / torch.pow(self.base, i / d) # (d/2, ) -> theta
        pos = torch.arange(0, s, dtype=torch.float32).unsqueeze(1) # (s, 1)
        angles = pos * theta # (s, d/2)

        res = torch.zeros(size=(s, d), dtype=torch.float32)
        res[:, ::2] = X[:, ::2] * torch.cos(angles) - X[:, 1::2] * torch.sin(angles)
        res[:, 1::2] = X[:, 1::2] * torch.cos(angles) + X[:, ::2] * torch.sin(angles)
        return res

if __name__ == '__main__':
    s, d = 10, 64
    X = torch.randn((s, d))
    rope = RoPE()
    res = rope(X)
