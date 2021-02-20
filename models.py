from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv
import networkx as nx

from utils.common_tools import get_first_element, get_last_element


class GCN_gen(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aspect_embed_size):
        super(GCN_gen, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, aspect_embed_size))
        self.layers.append(GraphConv(aspect_embed_size, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i == self.n_layers - 1:
                aspect_embed = h
        return aspect_embed, h


class Generative_model(nn.Module):
    '''The generative model'''
    def __init__(self,
                 g,
                 nx_g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aspect_embed_size,
                 n_edge_type,
                 hidden_x,
                 neg_ratio):
        super(Generative_model, self).__init__()
        self.p_a_x = GCN_gen(g,
                             in_feats,
                             n_hidden,
                             n_classes,
                             n_layers,
                             activation,
                             dropout,
                             aspect_embed_size)
        self.x_enc = nn.Linear(in_feats, hidden_x)
        self.p_e_xa = nn.Linear(2 * (hidden_x + aspect_embed_size), 1)
        self.n_edge_type = n_edge_type
        self.p_t_e = nn.Linear(2 * aspect_embed_size, self.n_edge_type)

        self.dropout = dropout
        self.activation = activation
        self.neg_ratio = neg_ratio

        self.nx_g = nx_g
        self.edge_list = list(self.nx_g.edges)
        start_nodes    = list(map(get_first_element, self.edge_list))
        end_nodes      = list(map(get_last_element,  self.edge_list))
        start_nodes    = torch.LongTensor(start_nodes)
        end_nodes      = torch.LongTensor(end_nodes)
        assert start_nodes.size() == end_nodes.size()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_index_0 = start_nodes.to(self.device)
        self.edge_index_1 = end_nodes.to(self.device)
        self.edge_typer = nx.get_edge_attributes(self.nx_g, 'type')


    def forward(self, node_features):
        aspect_embed, logits = self.p_a_x(node_features)
        aspect = F.log_softmax(aspect_embed, dim=1)
        x = F.dropout(node_features, p=self.dropout, training=self.training)
        x = self.activation(self.x_enc(x))

        edge_index_0 = self.edge_index_0
        edge_index_1 = self.edge_index_1

        a_query = F.embedding(edge_index_0, aspect)
        a_key   = F.embedding(edge_index_1, aspect)
        x_query = F.embedding(edge_index_0, x)
        x_key   = F.embedding(edge_index_1, x)
        xa = torch.cat([x_query, x_key, a_query, a_key], dim=1)
        e_pred_pos = self.p_e_xa(xa)

        e_pred_neg = None
        if self.neg_ratio > 0:
            num_edges_pos  = edge_index_0.size(0)
            num_nodes      = node_features.size(0)
            num_edges_neg  = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = torch.randint(num_nodes, (2, num_edges_neg)).to(x.device)
            a_query = F.embedding(edge_index_neg[0], aspect)
            a_key   = F.embedding(edge_index_neg[1], aspect)
            x_query = F.embedding(edge_index_neg[0], x)
            x_key   = F.embedding(edge_index_neg[1], x)
            xa = torch.cat([x_query, x_key, a_query, a_key], dim=1)
            e_pred_neg = self.p_e_xa(xa)

        return e_pred_pos, e_pred_neg, aspect, logits


    def nll_generative(self, node_features, post_aspect, trn_node, trn_label):
        e_pred_pos, e_pred_neg, aspect_gen, logits = self.forward(node_features)
        y_log_prob = F.log_softmax(logits, dim=1)
        nll_y = F.nll_loss(y_log_prob[trn_node], trn_label)
        temp  = torch.exp(post_aspect) * (post_aspect - aspect_gen)
        KLD   = torch.sum(temp, dim=-1).mean()

        nll_p_g_xa = -torch.mean(F.logsigmoid(e_pred_pos))
        if e_pred_neg is not None:
            nll_p_g_xa += -torch.mean(F.logsigmoid(-e_pred_neg))

        return nll_y + nll_p_g_xa + KLD


    def self_supervised(self, node_features, aspect_embed, \
                        edge_type, mask_rate, node_types=None, \
                        masked_node_indices=None):
        '''
            node and edge type as self-supervision.
            based on message propagation among partially edge-masked sub-graphs.
        '''
        if masked_node_indices is None:
            num_nodes = node_features.size(0)
            sample_size = int(num_nodes * mask_rate + 1)
            masked_node_indices = random.sample(range(num_nodes), sample_size)

        if node_types:
            mask_node_types_list = []
            for node_idx in masked_node_indices:
                mask_node_types_list.append(node_types[node_idx].view(1, -1))
            mask_node_types = torch.cat(mask_node_types_list, dim=0).to(self.device)
        masked_node_indices = torch.tensor(masked_node_indices).to(self.device)

        masked_edges_candidate = []
        masked_edges_candidate_indices = []
        for e_idx, e in enumerate(self.edge_list):
            u, v = e
            if u not in masked_node_indices and v not in masked_node_indices \
                and e_idx not in masked_edges_candidate_indices:
                masked_edges_candidate.append(e)
                masked_edges_candidate_indices.append(e_idx)
        assert len(masked_edges_candidate_indices) == len(masked_edges_candidate)

        n_masked_edges = int(len(masked_edges_candidate) * mask_rate + 1)
        temp = list(zip(masked_edges_candidate, masked_edges_candidate_indices))
        random.shuffle(temp)
        ec, eci = zip(*temp)
        ec, eci = list(ec), list(eci)
        masked_edges = ec[:n_masked_edges]
        masked_edges_indices = eci[:n_masked_edges]
        masked_edges_indices = torch.tensor(masked_edges_indices).to(self.device)
        edge_type = torch.cat([edge_type, edge_type], dim=0)
        masked_edge_types = edge_type[masked_edges_indices]

        start_nodes = list(map(get_first_element, masked_edges))
        end_nodes   = list(map(get_last_element,  masked_edges))
        m_edge_index_0 = torch.LongTensor(start_nodes).to(self.device)
        m_edge_index_1 = torch.LongTensor(end_nodes).to(self.device)
        assert m_edge_index_0.size() == m_edge_index_1.size()

        aspect = F.log_softmax(aspect_embed, dim=1)
        a_s    = F.embedding(m_edge_index_0, aspect)
        a_e    = F.embedding(m_edge_index_1, aspect)
        e_embed = torch.cat([a_s, a_e], dim=1)
        edge_type_pred = self.p_t_e(e_embed)
        criterion = nn.CrossEntropyLoss()
        ss_loss = criterion(edge_type_pred, masked_edge_types)
        return ss_loss



class GCN_post(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aspect_embed_size):
        super(GCN_post, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, aspect_embed_size))
        self.layers.append(GraphConv(aspect_embed_size, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i == self.n_layers - 1:
                aspect_embed = h
        return aspect_embed, h


class GraphSAGE_post(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aspect_embed_size,
                 aggregator_type='pool'):
        super(GraphSAGE_post, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, \
                           activation=activation))
        for i in range(n_layers - 2):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, \
                               activation=activation))
        self.layers.append(SAGEConv(n_hidden, aspect_embed_size, aggregator_type))
        self.layers.append(SAGEConv(aspect_embed_size, n_classes, aggregator_type))
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i == self.n_layers - 1:
                aspect_embed = h
        return aspect_embed, h


class AdaptiveLinear(nn.Module):
    def __init__(self, n_input, n_output):
        super(AdaptiveLinear, self).__init__()
        w = np.zeros((n_input, n_output), dtype=np.float32)
        w = nn.Parameter(torch.from_numpy(w))
        b = np.zeros(n_output, dtype=np.float32)
        b = nn.Parameter(torch.from_numpy(b))
        self.n_input  = n_input
        self.n_output = n_output
        self.w = w
        self.b = b
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.w) + self.b


class A2Conv(nn.Module):
    def __init__(self, embed_size, K):
        super(A2Conv, self).__init__()
        assert embed_size % K == 0
        self.embed_size = embed_size
        self.K = K
        self._cache_zero_d = torch.zeros(1, self.embed_size)
        self._cache_zero_k = torch.zeros(1, self.K)

    def forward(self, node_features, neighbors, edgenode_iter):
        device = node_features.device
        if self._cache_zero_d.device != device:
            self._cache_zero_d = self._cache_zero_d.to(device)
            self._cache_zero_k = self._cache_zero_k.to(device)
        num_node = node_features.size(0)
        num_neib = neighbors.size(0) // num_node
        embed_size = self.embed_size
        K = self.K
        a_size = self.embed_size // self.K

        h = F.normalize(node_features.view(num_node, K, a_size), dim=2).view(num_node, embed_size)
        z = torch.cat([h, self._cache_zero_d], dim=0)
        z = z[neighbors].view(num_node, num_neib, K, a_size)
        u = None
        for it in range(edgenode_iter):
            if u is None:
                edge_weight = self._cache_zero_k.expand(num_node * num_neib, K).view(num_node, num_neib, K)
            else:
                edge_weight = torch.sum(z * u.view(num_node, 1, K, a_size), dim=3)
            edge_weight = F.softmax(edge_weight, dim=2)
            u = torch.sum(z * edge_weight.view(num_node, num_neib, K, 1), dim=1)
            u += h.view(num_node, K, a_size)
            if it < edgenode_iter - 1:
                u = F.normalize(u, dim=2)
        return u.view(num_node, embed_size)


class A2GNN_post(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 n_hidden,
                 aspect_embed_size,
                 a2gnn_num_layer,
                 args):
        super(A2GNN_post, self).__init__()
        n_aspect   = args.K
        embed_size = n_hidden * n_aspect
        self.ALP   = AdaptiveLinear(in_feats, embed_size)
        conv_ls    = []
        for i in range(a2gnn_num_layer - 1):
            conv = A2Conv(embed_size, n_aspect)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls     = conv_ls
        self.aspect_mlp  = nn.Linear(embed_size, aspect_embed_size)
        self.mlp         = nn.Linear(aspect_embed_size, n_classes)
        self.dropout     = args.dropout
        self.edgenode_it = args.edgenode_it

    def _dropout(self, node_features):
        return F.dropout(node_features, self.dropout, training=self.training)

    def forward(self, node_features, nb):
        nb = nb.view(-1)
        h = F.relu(self.ALP(node_features))
        for conv in self.conv_ls:
            h = self._dropout(F.relu(conv(h, nb, self.edgenode_it)))
        aspect_embed = self.aspect_mlp(h)
        h = self.mlp(aspect_embed)
        return aspect_embed, h



class GenGNN(nn.Module):
    '''
        SS-PGM model.
    '''
    def __init__(self, gen_config, post_config):
        super(GenGNN, self).__init__()

        self.gen_type = gen_config.pop("type")
        if self.gen_type == "gcn" or self.gen_type == "a2gnn":
            self.gen = Generative_model(**gen_config)
        else:
            raise NotImplementedError("Generative model type {} not supported.".format(self.gen_type))

        self.post_type = post_config.pop("type")
        if self.post_type == "gcn":
            self.post = GCN_post(**post_config)
        elif self.post_type == 'graphsage':
            self.post = GraphSAGE_post(**post_config)
        elif self.post_type == 'a2gnn':
            post_config.pop('g')
            post_config.pop('dropout')
            post_config.pop('activation')
            self.post = A2GNN_post(**post_config)
        else:
            raise NotImplementedError("Generative model type {} not supported.".format(self.post_type))

    def cal_post(self, node_features, neib_sampler):
        aspect_embed, logits = self.post(node_features, neib_sampler.sample()) 
        return aspect_embed, logits