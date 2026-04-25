import torch
from torch import nn
from patchify import Patchify, Unpatchify
from embeddings import TimestepEmbedder, LabelEmbedder
from dit_block import DiTBlock

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

        self.time_embedder = TimestepEmbedder(time_emb_dim)
        self.label_embedder = LabelEmbedder(num_labels, hidden_dim, cfg_dropout)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, head_size, cfg_dropout)
            for _ in range(n_blocks)
        ])

        self.to_noise = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, t, y=None):
        x = self.patchify(x)
        x = self.time_embedder(x)
        x = self.label_embedder(x)
        for block in self.blocks:
            x = block(x)
        x = self.to_noise(x)
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



if __name__ == "__main__":
    model = DiT_tiny()

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    model = DiT_small()

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    model = DiT_medium()

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
