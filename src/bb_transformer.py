import torch
import torch.nn as nn
from open_clip.transformer import LayerNorm

from typing import Optional
from collections import OrderedDict


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class BBResidualBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            short_bb: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.short_bb = short_bb
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = nn.Identity()

        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = nn.Identity()

    def attention(self, q_x: torch.Tensor):
        return self.attn(q_x, q_x, q_x, need_weights=False, attn_mask=None)[0]

    def gamma(self,
              x1: torch.Tensor, # (S, B, E)
              g1: torch.Tensor, # (S, B, E)
              x0: Optional[torch.Tensor] = None, # (S, B, E)
              g0: Optional[torch.Tensor] = None, # (S, B, E)
              ):

        if (x0 is None) or (g0 is None):
            return torch.ones((x1.shape[1], x1.shape[2]), device=x1.device)
            # return torch.ones((x1.shape[1]), device=x1.device)

        delta_x = x1 - x0 # (S, B, E)
        delta_g = g0 - g1 # (S, B, E)

        if self.short_bb:
            BB_step = (delta_x * delta_g).sum(dim=(0, 2), keepdim=True) / ((delta_g * delta_g).sum(dim=(0, 2), keepdim=True) + 1e-5) # (B)
        else:
            BB_step = (delta_x * delta_x).sum(dim=(0, 2), keepdim=True) / ((delta_x * delta_g).sum(dim=(0, 2), keepdim=True) + 1e-5) # (B)

        return BB_step.squeeze() # (B)

    def forward(self,
                x1: torch.Tensor, # x1 shape: (S, B, E)
                x0: Optional[torch.Tensor] = None, # (S, B, E)
                g0: Optional[torch.Tensor] = None, # (S, B, E)
    ):
        x1_ = x1.detach() # (S, B, E)
        g1 = self.ls_1(self.attention(q_x=self.ln_1(x1))) # (S, B, E)
        g1_ = g1.detach() # (S, B, E)

        gamma1 = self.gamma(x1_, g1_, x0, g0) # (B)
        x2 = x1 + g1 * gamma1 # (S, B, E)

        x2_ = x2.detach() # (S, B, E)
        g2 = self.ls_2(self.mlp(self.ln_2(x2))) # (S, B, E)
        g2_ = g2.detach() # (S, B, E)

        gamma2 = self.gamma(x2_, g2_, x1_, g1_) # (B)
        x3 = x2 + g2 * gamma2 # (S, B, E)

        return x3, x2_, g2_ # (S, B, E), (S, B, E), (S, B, E)
        

class BBTransformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            short_bb: bool = False
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            BBResidualBlock(width, heads, mlp_ratio, short_bb)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor):
        x1, x0, g0 = self.resblocks[0](x)

        for r in range(1, len(self.resblocks)):
            x1, x0, g0 = self.resblocks[r](x1, x0, g0)

        return x1

