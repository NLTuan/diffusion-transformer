import torch
from torch import nn
from patchify import Patchify
from embeddings import TimestepEmbedder, LabelEmbedder
from dit_block import DiTBlock

class DiT(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_blocks,
        patch_size=2,
        time_emb_dim=256,
        cfg_dropout=0.0
        ):

        self.patchify = Patchify(hidden_dim, patch_size, in_channels=3)
        self.unpatchify = Unpatchify(hidden_dim, patch_size)

        self.time_embedder = TimestepEmbedder(time_emb_dim)
        self.label_embedder = LabelEmbedder(time_emb_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, time_emb_dim, cfg_dropout)
            for _ in range(n_blocks)
        ])

        self.to_noise = nn.Linear(hidden_dim, hidden_dim)

        


        
    
    
    def forward(self, x):
        pass