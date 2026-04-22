import torch
from torch import nn

class TimestepEmbedder(nn.Module):
    """
        Embeds the timestep with sin-cosine positional embeddings
    """
    def __init__(self, dim=1152, theta=100000):
        pass
    
    def forward(self, x):
        pass
    
class LabelEmbedder(nn.Module):
    """
        Embeds the label with sin-cosine positional embeddings
    """
    def __init__(self, dim=1152):
        pass
    
    def forward(self, x):
        pass