import torch
from torch import nn
from torch.nn import LayerNorm, MultiheadAttention


def modulate(x, scale, shift):
    """
        helper function for applying the AdaLN shift
    """
    return x * (1 + scale.unsqueeze(dim=1)) + shift.unsqueeze(dim=1)



class DiTBlock(nn.Module):
    def __init__(self, input_shape, num_heads, dropout=0.1):
        super().__init__()
        dim = input_shape[-1]
        self.ln1 = LayerNorm(input_shape, elementwise_affine=False)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.mha = MultiheadAttention(dim, num_heads, dropout)
        self.conditionning_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )
        
        self.ln2 = LayerNorm(input_shape)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
        
    def forward(self, x, conditionning):
        g1, b1, a1, g2, b2, a2 = self.conditionning_mlp(conditionning).chunk(6, dim=1)
        res = x
        x = modulate(self.ln1(x), g1, b1)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        x = res + a1.unsqueeze(1) * self.mha(q, k, v)[0]
        x = x + a2.unsqueeze(1) * self.ffn(modulate(self.ln2(x), g2, b2))
        return x
    

class FinalLayer(nn.Module):
    def __init__(self, input_dim, patch_size, out_channels):
        super().__init__()
        self.ln = nn.LayernNorm(input_dim, elementwise_affine=True)
        dim = input_dim[-1]
        self.conditionning_mlp = nn.Linear(
            nn.SiLU(),
            nn.Linear(dim, dim * 2)
        )
        
        self.mlp = nn.Linear(dim, dim * patch_size * out_channels)
    
    def forward(self, x, cond):
        shift, scale = self.conditionning_mlp(cond).chunk(2, dim=-1)
        x = modulate(x, shift, scale)
        x = self.mlp(x)
        
if __name__ == "__main__":
    cond = torch.randn(3, 128)
    
    latent = torch.randn(3, 23, 128)
    
    block = DiTBlock(latent.shape, 4)
    
    print(f'Block output shape: {block(latent, cond).shape}')