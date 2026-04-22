import torch
from einops import rearrange

from torch import nn

class Patchify(nn.Module):
    def __init__(self, 
                 dim=1152, 
                 patch_size=2,
                 in_channels=4):
        
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x) # (B, C, W, H) -> (B, hidden_dim, w // patch_size, h // patch_size)
        return rearrange(x, 'b d h w -> b (h w) d') # (B, "seq_len", hidden_dim)
    
    
if __name__ == "__main__":
    noised_latent = torch.randn(3, 4, 32, 32)
    
    patchify = Patchify()
    print(patchify(noised_latent).shape)