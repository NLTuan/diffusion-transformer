import torch
from torch import nn


class LayerNorm(nn.Module):
    
    def __init__(self, input_shape, eps=0.00001):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(input_shape))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm
    
    
if __name__ == "__main__":
    a = torch.rand(1, 23, 128) * 213 + 23
    
    ln = LayerNorm(a.shape)
    
    print('statistics pre norm')
    print(a.mean(dim=-1))
    print(a.var(dim=-1))
    
    b = ln(a)
    print('statistics post norm')
    print(b.mean(dim=-1))
    print(b.var(dim=-1))