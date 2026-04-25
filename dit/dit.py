import torch
from torch import nn
import numpy as np
from patchify import Patchify, Unpatchify
from embeddings import TimestepEmbedder, LabelEmbedder
from dit_block import DiTBlock, FinalLayer

class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_dim=1152,
        head_size=16,
        num_labels=10,
        n_blocks=28,
        time_emb_dim=256,
        cfg_dropout=0.0
        ):
        super().__init__()    
        self.patchify = Patchify(hidden_dim, patch_size, in_channels)
        self.unpatchify = Unpatchify(patch_size)

        self.time_embedder = TimestepEmbedder(hidden_dim, time_emb_dim)
        self.label_embedder = LabelEmbedder(num_labels, hidden_dim, cfg_dropout)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, head_size, cfg_dropout)
            for _ in range(n_blocks)
        ])

        self.to_noise = FinalLayer(hidden_dim, patch_size, in_channels)
        
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim), requires_grad=False)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        grid_size = int(self.pos_embed.shape[1] ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patchify.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patchify.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.label_embedder.embs.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.proj[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.proj[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.conditionning_mlp[-1].weight, 0)
            nn.init.constant_(block.conditionning_mlp[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.to_noise.conditionning_mlp[-1].weight, 0)
        nn.init.constant_(self.to_noise.conditionning_mlp[-1].bias, 0)
        nn.init.constant_(self.to_noise.mlp.weight, 0)
        nn.init.constant_(self.to_noise.mlp.bias, 0)

    def forward(self, x, t, y=None):
        x = self.patchify(x) + self.pos_embed
        time_embs = self.time_embedder(t)
        label_embs = self.label_embedder(y, self.training)
        cond = time_embs + label_embs
        for block in self.blocks:
            x = block(x, cond)
        x = self.to_noise(x, cond)
        x = self.unpatchify(x)
        return x


def DiT_tiny():
    return DiT(
        hidden_dim=192,
        n_blocks=6,
        patch_size=2,
        time_emb_dim=256,
        cfg_dropout=0.1
    )

def DiT_small():
    return DiT(
        hidden_dim=384,
        n_blocks=12,
        patch_size=2,
        time_emb_dim=256,
        cfg_dropout=0.1
    )

def DiT_medium():
    return DiT(
        hidden_dim=768,
        n_blocks=24,
        patch_size=2,
        time_emb_dim=256,
        cfg_dropout=0.1
    )

def DiT_large():
    return DiT(
        hidden_dim=1152,
        n_blocks=28,
        patch_size=2,
        time_emb_dim=256,
        cfg_dropout=0.1
    )



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# Shamelessly stolen from MAE
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



if __name__ == "__main__":
    model = DiT_tiny()

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    model = DiT_small()

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    latent = torch.randn(3, 4, 32, 32)
    t = torch.randint(0, 1000, (1,))
    y = torch.randint(0, 10, (1,))
    print(model(latent, t, y).shape)