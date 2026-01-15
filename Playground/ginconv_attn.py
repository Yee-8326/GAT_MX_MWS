"""Torch Module for Graph Isomorphism Network layer"""
import math

import torch
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn, Tensor
import torch.nn.functional

from dgl import function as fn
from dgl.utils import expand_as_pair


class GINConv_attn(nn.Module):
    r"""Graph Isomorphism Network layer from `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    If a weight tensor on each edge is provided, the weighted graph convolution is defined as:

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{e_{ji} h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    where :math:`e_{ji}` is the weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that `e_{ji}` is broadcastable with `h_j^{l}`.

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula, default: None.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``), default: 'sum'.
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GINConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> lin = th.nn.Linear(10, 10)
    >>> conv = GINConv(lin, 'max')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.1804,  0.0758, -0.5159,  0.3569, -0.1408, -0.1395, -0.2387,  0.7773,
            0.5266, -0.4465]], grad_fn=<AddmmBackward>)

    >>> # With activation
    >>> from torch.nn.functional import relu
    >>> conv = GINConv(lin, 'max', activation=relu)
    >>> res = conv(g, feat)
    >>> res
    tensor([[5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [2.5011, 0.0000, 0.0089, 2.0541, 0.8262, 0.0000, 0.0000, 0.1371, 0.0000,
             0.0000]], grad_fn=<ReluBackward0>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,  # equal to linear apply_func


                 apply_func=None,
                 aggregator_type='mean',
                 init_eps=0,
                 learn_eps=True,
                 activation=None,
                 attn_type="None",
                 ):
        super(GINConv_attn, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        self.num_heads = 1
        self.attn_type = attn_type
        self.attn_tensor = None
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn = nn.Parameter(th.FloatTensor(size=(1, self.num_heads, out_feats)))
        self.sigmoid = nn.Sigmoid()
        self.feat_drop = nn.Dropout(0)

        if aggregator_type not in ('sum', 'max', 'mean'):
            raise KeyError(
                'Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * self.num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * self.num_heads, bias=False)
        else:
            # 全连接层
            self.fc_src = self.fc_dst = self.fc = nn.Linear(
                self._in_src_feats, out_feats * self.num_heads, bias=False)



    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        graph = graph.to("cuda:0")
        feat = feat.to("cuda:0")

        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')

            if (self.attn_type == "MX" or self.attn_type == "SD" or
                    self.attn_type == "ORIGINAL"):
                self.attn_tensor = self.get_attention_tensor(graph, feat)
                edge_weight = self.attn_tensor

            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, _reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation == nn.functional.leaky_relu:
                rst = self.activation(rst, negative_slope=0.2)
            elif self.activation is not None:
                rst = self.activation(rst)
            return rst

    def get_attention_tensor(self, graph, feat) -> Tensor:
        with graph.local_scope():
            # feat_src, feat_dst = expand_as_pair(feat, graph)
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self.num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self.num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                # Wh_i(src)、Wh_j(dst)在各head的特征组成的矩阵: (1, num_heads, out_feats)
                feat_src = self.fc_src(h_src).view(
                    -1, self.num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(
                    -1, self.num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_dst[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]

            if self.attn_type == "ORIGINAL":
                graph.srcdata.update(
                    {"el": feat_src}
                )  # (num_src_edge, num_heads, out_dim)
                graph.dstdata.update({"er": feat_dst})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
                e = self.leaky_relu(
                    graph.edata.pop("e")
                )  # (num_src_edge, num_heads, out_dim)
                e = (
                    (e * self.attn).sum(dim=-1).unsqueeze(dim=2)
                )  # (num_edge, num_heads, 1)
                # print(e.shape)

            elif self.attn_type == "MX":
                graph.srcdata.update(
                    {"el": feat_src}
                )  # (num_src_edge, num_heads, out_dim)
                graph.dstdata.update({"er": feat_dst})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
                e = self.leaky_relu(
                    graph.edata.pop("e")
                )  # (num_src_edge, num_heads, out_dim)
                e = (
                    (e * self.attn).sum(dim=-1).unsqueeze(dim=2)
                )  # (num_edge, num_heads, 1)
                # print(e.shape)
                # import SuperGAT DP Attn
                graph.apply_edges(fn.u_dot_v("el", "er", "logits"))
                logits = self.sigmoid(graph.edata.pop("logits"))
                # print(logits.shape)
                e = torch.mul(e, logits)

            elif self.attn_type == "SD":
                graph.srcdata.update(
                    {"el": feat_src}
                )  # (num_src_edge, num_heads, out_dim)
                graph.dstdata.update({"er": feat_dst})
                graph.apply_edges(fn.u_dot_v("el", "er", "logits"))
                logits = graph.edata.pop("logits")
                logits = torch.div(logits, math.sqrt(self._out_feats))
                e = logits

            else:
                return None
            func = nn.Sigmoid()
            return func(torch.flatten(e))