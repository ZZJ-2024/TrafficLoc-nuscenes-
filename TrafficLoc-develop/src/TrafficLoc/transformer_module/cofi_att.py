import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_attention import LinearAttention, FullAttention
import math

# D_MODEL = 128
# D_FFN = 256
# NHEAD = 4
# LAYER_NAMES = ['self', 'cross'] * 4
# ATTENTION = 'full'  # options: ['linear', 'full']


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='full'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
        """
        bs = x.size(0)  # batch size
        query, key, value = x, source, source

        # multi-head attention
        query = F.normalize(self.q_proj(query).view(bs, -1, self.nhead, self.dim))  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message, raw_att = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message, raw_att


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, D_MODEL, NHEAD, LAYER_NAMES, ATTENTION):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = D_MODEL
        self.nhead = NHEAD
        self.layer_names = LAYER_NAMES
        self.num_layers = len(self.layer_names)
        encoder_layer = LoFTREncoderLayer(D_MODEL, NHEAD, ATTENTION)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, on_layer='last'):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C] img feat
            feat1 (torch.Tensor): [N, S, C] point feat
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        raw_att_i2p_list = []
        raw_att_p2i_list = []
        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if name == 'self':
                feat0, _ = layer(feat0, feat0)
                feat1, _ = layer(feat1, feat1)
            elif name == 'cross':
                feat0, raw_att_i2p = layer(feat0, feat1)
                feat1, raw_att_p2i = layer(feat1, feat0)
                raw_att_i2p_list.append(raw_att_i2p)
                raw_att_p2i_list.append(raw_att_p2i)
            else:
                raise KeyError

        if on_layer == 'last':
            return feat0, feat1, raw_att_i2p_list[-1], raw_att_p2i_list[-1]
        elif on_layer == 'first':
            return feat0, feat1, raw_att_i2p_list[0], raw_att_p2i_list[0]
        elif on_layer == 'all':
            return feat0, feat1, raw_att_i2p_list, raw_att_p2i_list
        else:
            raise ValueError
        
        # return feat0, feat1, raw_att_i2p, raw_att_p2i
    

class PositionEmbeddingCoordsSineCoFiI2P(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """
    def __init__(self, n_dim: int = 1, d_model: int = 64, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb
    
if __name__ == "__main__":
    d_model = 256
    
    transformer = LocalFeatureTransformer(D_MODEL=d_model, NHEAD=4, LAYER_NAMES=['self', 'cross'] * 4, ATTENTION = 'full')
    transformer = transformer.cuda()
    
    img_feature = torch.rand((2,100,d_model)).cuda()
    pcd_feature = torch.rand((2,100,d_model)).cuda()
    
    f0, f1 = transformer(img_feature, pcd_feature)
    
    print(f0.shape)
    print(f1.shape)
    
    None