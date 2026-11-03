import torch
from torch import nn 

class SinusoidalPE(nn.Module):
    def __init__(self, base=1e4):
        super().__init__()
        self.base = base
    def forward(self, X):
        s, d = X.shape
        if d % 2 != 0:
            raise ValueError('hidden_size must be even')
        pos = torch.arange(0, s, 1, dtype=torch.float32).unsqueeze(1) # (slen, 1)
        i = torch.arange(0, d, 2, dtype=torch.float32) # (d/2,)
        div = torch.pow(self.base, i / d)
        angles = pos / div # (slen, d/2)
        pe = torch.empty(size=(s, d),dtype=torch.float32)
        pe[:, ::2] = torch.sin(angles) 
        pe[:, 1::2] = torch.cos(angles)
        return X + pe, pe

if __name__ == '__main__':
    s, d = 10, 64
    X = torch.randn((s, d))
    pe = SinusoidalPE()
    X, p = pe(X)

