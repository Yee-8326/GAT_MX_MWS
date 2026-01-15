import torch.nn as nn
import torch.nn.functional as F
from ginconv_attn import GINConv_attn

__all__ = ["GIN_attn"]


class GIN_attn_Layer(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            feat_drop=0.0,
            negative_slope=0.2,
            residual=False,
            activation=None,
            attn_type=None,
            linear_in=74,
            linear_out=74

    ):
        super(GIN_attn_Layer, self).__init__()
        self.apply_func = nn.Linear(linear_in, linear_out)
        self.gin_attn_conv = GINConv_attn(
            in_feats=linear_in,
            out_feats=linear_out,
            apply_func=self.apply_func,
            attn_type=attn_type,
            activation=nn.functional.elu,
            # activation=nn.functional.leaky_relu
        )

        # self.gin_attn_conv = GINConv_attn(
        #     in_feats=74,
        #     out_feats=74,
        #     apply_func=None,
        #     attn_type=attn_type,
        #     activation=nn.functional.elu,
        #     # activation=nn.functional.leaky_relu
        # )

    def forward(self, bg, feats):
        bg = bg.to("cuda:0")
        feats = feats.to("cuda:0")
        out_feats = self.gin_attn_conv(bg, feats)
        return out_feats


class GIN_attn(nn.Module):
    def __init__(
            self,

            gin_attn_types,
            in_feats=74,
            out_feats=74
    ):
        super(GIN_attn, self).__init__()

        n_layers = len(gin_attn_types)
        self.gin_layers = nn.ModuleList()
        # for i in range(n_layers):
        #     self.gin_layers.append(
        #         GIN_attn_Layer(
        #             in_feats=in_feats,
        #             attn_type=gin_attn_types[i],
        #             out_feats=in_feats,
        #         )
        #     )
        self.gin_layers.append(
            GIN_attn_Layer(
                in_feats=in_feats,
                linear_in=in_feats,
                attn_type="NONE",
                out_feats=out_feats,
                linear_out=out_feats
            )
        )
        #




    def forward(self, bg, feats):
        # bg = bg.to("cuda:0")
        # feats = feats.to("cuda:0")

        for gin in self.gin_layers:
            feats = gin(bg, feats)
        return feats
