import dgl
import dgl.nn as dglnn
import torch
import torch.nn.functional as F

from ModelGCN import EwBaseN, GatNNet
from consts import SEED

# imports for DGL GraphConv
import tensorflow as tf
from dgl.base import DGLError
from dgl.ops import edge_softmax
from dgl import function as fn


class EwGatN(EwBaseN):
    """
    GAT GCN model with edge weightening.

    Be sure to modify this file if you modify GatN - bad practice here, but I'm not sure how to make it better.
    """
    def __init__(self, data: dgl.DGLGraph, alpha: float, hid_feats: int, num_heads: (int, int) = (1, 1),
                 feat_drop: (float, float) = (0.5, 0.), att_drop: (float, float) = (0.5, 0.),
                 negative_slope: float = .2, lr: float = 1e-3, weight_decay: float = 0., inductive: bool = False,
                 validation: bool = True, gpu: str = 'cpu', run_name: str = '', debug_mode: bool = False,
                 seed: int = SEED):
        """
        :param data: dgl.graph,
        :param alpha: float [0, 1] edge weightening strength parameter, 0 - no weights, 1 - complete counterweights
        :param hid_feats: int, size of each hidden layer
        :param num_heads: (int, int), number of attention heads for each Conv layer
        :param feat_drop: (float, float), probabilities of zeroeing feature value after each Conv layer
        :param att_drop: (float, float), attention dropout for each Conv layer
        :param negative_slope: float, LeakyReLU angle of negative slope, same for both Conv layers
        :param lr: float, learning rate of the optimizer
        :param weight_decay: float, weight decay (L2 penalty) of the optimizer
        :param inductive: bool, whether inductive setting should be used, transductive by default.
        :param validation: bool, whether validation set should be used, if False, add it to test set
        :param gpu: str, gpu to be run on in format, e.g., 'cuda:14'
        :param run_name: str, name of the run, for easy reading in wandb
        :param debug_mode: bool, if yes, then print instead of log to wandb
        :param seed: randomness seed
        """
        EwBaseN.__init__(self, data=data, alpha=alpha, inductive=inductive, validation=validation, gpu=gpu,
                         debug_mode=debug_mode, seed=seed)
        assert len(num_heads) == len(feat_drop) == len(att_drop) == 2
        self.model = EwGatNNet(data.ndata['feat'].shape[1], hid_feats, data.num_classes, num_heads, feat_drop, att_drop,
                             negative_slope).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        for data in [self.data, self.train_data, self.test_data] + [self.val_data] if self.validation else []:
            data.edata['w'] = data.edata['w'].reshape((data.edata['w'].shape[0], 1, 1))

        self.set_run_config(run_name=run_name, lr=lr, weight_decay=weight_decay)


class EwGatNNet(GatNNet):
    def __init__(self, in_feats, hid_feats, out_feats, num_heads, feat_drop, att_drop, negative_slope):
        super().__init__(in_feats, hid_feats, out_feats, num_heads, feat_drop, att_drop, negative_slope)
        self.conv0 = EwGATConv(in_feats=in_feats, out_feats=hid_feats, num_heads=num_heads[0], feat_drop=feat_drop[0],
                                   attn_drop=att_drop[0], negative_slope=negative_slope, activation=F.relu)
        self.conv1 = EwGATConv(in_feats=hid_feats * num_heads[0], out_feats=hid_feats, num_heads=num_heads[1],
                                   feat_drop=feat_drop[1], attn_drop=att_drop[1], negative_slope=negative_slope,
                                   activation=F.relu)


class EwGATConv(dglnn.GATConv):
    """
    I think it doesn't make much sense to apply weights on edges, because attention will just learn to negate them.
    Let's try it anyway and see what are the results.

    We've added multiplication of the edge attention weights by fair edge weights.
    """
    def forward(self, graph, feat, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            # multiply by "fair" edge weights
            graph.edata['e'] *= graph.edata['w']
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
