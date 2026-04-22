import torch
from torch import nn
import math

class TimestepEmbedder(nn.Module):
    """
        Embeds the timestep with sin-cosine positional embeddings
    """
    def __init__(self, hidden_dim, freq_emb_dim=256, theta=100000):
        super().__init__()
        self.freq_emb_dim = freq_emb_dim
        
        self.proj = nn.Sequential(
            nn.Linear(freq_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    
    @staticmethod
    def timestep_emb(t, dim, theta=10000):
        """
            Sine-Cosine + learned positional embeddings that are concatenated one after another (No weaving)
        """
        
        # t is a 1 dimensional vector with the timesteps that aren't discrete
        half = dim // 2
        
        freqs = torch.exp(
            -math.log(theta) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device) # Shape = (dim/2)
        
        args = t[:, None] * freqs[None] # (t, 1) * (1, dim/2) = (t, dim/2)
        
        out = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # 0 pad if the dimension is odd
        if (dim % 2 != 0):
            out = torch.cat([out, torch.zeros_like(out[:, 0])], dim=-1)
        
        return out
    
    def forward(self, x):
        embs = self.timestep_emb(x, self.freq_emb_dim)
        return self.proj(embs)
        
    
class LabelEmbedder(nn.Module):
    """
        Embeds the label with learned positional embeddings
    """
    def __init__(self, num_labels, dim, dropout):
        super().__init__()
        cfg_emb = dropout > 0
        self.embs = nn.Embedding(num_labels + cfg_emb)
        self.dim = dim
        self.dropout = dropout
        
        
    def cfg_filter(self, labels, force_drop_ids=None):
        """
            Drop labels and turn them into the classifier free labels
        """
        
        if force_drop_ids == None:
            drop = torch.rand(labels.shape[0], device = labels.device) < self.dropout
        else:
            drop = force_drop_ids == 1
        
        labels = torch.where(drop, self.num_classes, labels)
    
    def forward(self, labels, train, force_drop_ids=None):
        drop = self.dropout > 0
        if (train and drop) or force_drop_ids != None:
            labels = force_drop_ids(labels, force_drop_ids)
        
        return self.embs(labels)
    
    
    
if __name__ == '__main__':
    time_embs = TimestepEmbedder(512)
    
    timesteps = torch.randn(100)
    
    print(f"Timestep embedding dimension: {time_embs(timesteps).shape}")