from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torchdrug.utils import sparse_coo_tensor
import functools
from torchdrug.layers import  MessagePassingBase
from torch_scatter import scatter_mean,scatter_max, scatter_add, scatter_softmax
from pdb import set_trace as st
from torchdrug.layers.readout import Readout

class GraphAttentionConv(MessagePassingBase):
    """
    Graph attentional convolution operator from `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    eps = 1e-10

    def __init__(self, input_dim, output_dim, edge_input_dim=None, num_head=1, negative_slope=0.2, concat=True,
                 batch_norm=False, activation="relu"):
        super(GraphAttentionConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.num_head = num_head
        self.concat = concat
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)
        # self.dropout = nn.Dropout(0.1)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if output_dim % num_head != 0:
            raise ValueError("Expect output_dim to be a multiplier of num_head, but found `%d` and `%d`"
                             % (output_dim, num_head))

        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, output_dim)
        else:
            self.edge_linear = None
        self.query = nn.Parameter(torch.zeros(num_head, output_dim * 2 // num_head))
        nn.init.kaiming_uniform_(self.query, negative_slope, mode="fan_in")

    def message(self, graph, input):
        # add self loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        hidden = self.linear(input)

        key = torch.stack([hidden[node_in], hidden[node_out]], dim=-1)
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.output_dim, device=graph.device)])
            key += edge_input.unsqueeze(-1)
        key = key.view(-1, *self.query.shape)
        # st()
        weight = torch.einsum("hd, nhd -> nh", self.query, key)
        # st()
        weight = self.leaky_relu(weight)
        weight = weight - scatter_max(weight.int(), node_out, dim=0, dim_size=graph.num_node)[0][node_out]
        attention = weight.exp() * edge_weight
        # why mean? because with mean we have normalized message scale across different node degrees
        normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]# ok
        attention = attention / (normalizer + self.eps)
        # attention = self.dropout(attention)
        # print(attention)
        value = hidden[node_in].view(-1, self.num_head, self.query.shape[-1] // 2)
        attention = attention.unsqueeze(-1).expand_as(value)
        message = (attention * value).flatten(1)
        # st()
        # message = torch.load("./message.tensor")
        return message


    def aggregate(self, graph, message):
        # add self loop
        # st()
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        update = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)# ok
        return update


    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

class AttentionReadout(Readout):

    def __init__(self, input_dim, type="node"):
        super(AttentionReadout, self).__init__(type)
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, graph, input):
        index2graph = self.get_index2graph(graph)
        weight = self.linear(input)
        
        attention = scatter_softmax((weight*100).int()*0.001, index2graph, dim=0)
        output = scatter_add(attention * input, index2graph, dim=0, dim_size=graph.batch_size)
        return output