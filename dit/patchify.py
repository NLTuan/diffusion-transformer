import torch
from einops import rearrange

from torch import nn

class Patchify(nn.Module):
    """
        Turns an image into a sequence of embedded tokens
    """
    
    def __init__(
        self, 
        dim=1152, 
        patch_size=2,
        in_channels=4
    ):
        
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x) # (B, C, W, H) -> (B, hidden_dim, w // patch_size, h // patch_size)
        return rearrange(x, 'b d h w -> b (h w) d') # (B, "seq_len", hidden_dim)
    

class Unpatchify(nn.Module):
    """
        Turns a sequence of tokens-like into an image
    """
    def __init__(
        self,
        patch_size
    ):
        super().__init__()
        self.patch_size = patch_size


    def forward(self, x):
        h = w = int(x.shape[1] ** 0.5)

        x = rearrange(
            x, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
            h=h, 
            w=w,
            p1=self.patch_size,
            p2=self.patch_size
            )
        return x
    
if __name__ == "__main__":
    noised_latent = torch.randn(3, 4, 32, 32)
    
    patchify = Patchify()
    print('Original shape:', noised_latent.shape)
    print('Patched shape:', patchify(noised_latent).shape)

    sample_final_mlp = nn.Linear(1152, 2 * 2 * 4)
    print('Final mlp output shape:', sample_final_mlp(patchify(noised_latent)).shape)

    unpatchify = Unpatchify(2)
    print('Unpatched shape:', unpatchify(sample_final_mlp(patchify(noised_latent))).shape)

