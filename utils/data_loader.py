import os
import numpy as np
import pickle
from collections import defaultdict
import networkx as nx

import torch

def data_loader(data_dir, dataset_name, device):
    '''
        load Heterogeneous graph datasets.
    '''
    dataset_name = dataset_name.upper()
    print('loading {} from {} ...'.format(dataset_name, data_dir))
    node_feat_path = os.path.join(data_dir, dataset_name, 'node_features.pkl')
    with open(node_feat_path, 'rb') as f:
        node_features = pickle.load(f)
    edge_path = os.path.join(data_dir, dataset_name, 'edges.pkl')
    with open(edge_path, 'rb') as f:
        edges = pickle.load(f)
    label_path = os.path.join(data_dir, dataset_name, 'labels.pkl')
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)

    ## edge types
    edge_type_list = ['PA', 'AP', 'PS', 'SP']
    n_edge_type = len(edge_type_list)
    id_2_edgetype = dict(zip(list(range(len(edge_type_list))), edge_type_list))
    edgetype_2_id = dict(zip(edge_type_list, list(range(len(edge_type_list)))))

    P_PA = np.sort(edges[0].nonzero()[0])
    A_PA = np.sort(edges[0].nonzero()[1])
    A_AP = np.sort(edges[1].nonzero()[0])
    P_AP = np.sort(edges[1].nonzero()[1])
    assert sum(P_PA == P_AP) == sum(A_PA == A_AP) == \
           edges[0].nonzero()[0].shape == edges[1].nonzero()[0].shape
    assert set(P_PA) == set(P_AP)
    assert set(A_PA) == set(A_AP)

    P_PS = np.sort(edges[2].nonzero()[0])
    S_PS = np.sort(edges[2].nonzero()[1])
    S_SP = np.sort(edges[3].nonzero()[0])
    P_SP = np.sort(edges[3].nonzero()[1])
    assert sum(P_PS == P_SP) == sum(S_PS == S_SP) == \
           edges[2].nonzero()[0].shape == edges[3].nonzero()[0].shape
    assert set(P_PS) == set(P_SP)
    assert set(S_PS) == set(S_SP)

    num_nodes = edges[0].shape[0]
    adj_GTN = []
    adj_graphsage = defaultdict(set)
    nx_graph = nx.Graph()
    for idx, edge in enumerate(edges):
        edge_tmp  = np.vstack((edge.nonzero()[0], edge.nonzero()[1]))
        edge_tmp  = torch.from_numpy(edge_tmp).long().to(device)
        n_edge    = edge_tmp.shape[1]
        value_tmp = torch.ones(n_edge)
        value_tmp = value_tmp.long().to(device)
        adj_GTN.append((edge_tmp, value_tmp))
        cur_edge_type = id_2_edgetype[idx]
        for i in range(n_edge):
            source_node = edge_tmp[0][i].item()
            target_node = edge_tmp[1][i].item()
            adj_graphsage[source_node].add(target_node)
            nx_graph.add_edge(source_node, target_node, type=cur_edge_type)
    edge_tmp  = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).long().to(device)  # --> 此行以及下一行是添加单位矩阵(Identify matrix, 即对角线上元素都为1), 即self-connections
    value_tmp = torch.ones(num_nodes).long().to(device)
    adj_GTN.append((edge_tmp, value_tmp))
    assert len(adj_graphsage) == len(nx_graph.nodes()) == num_nodes

    ## node features
    node_features = torch.from_numpy(node_features).float().to(device)

    trn_node  = torch.from_numpy(np.array(labels[0])[:, 0]).long().to(device)
    trn_label = torch.from_numpy(np.array(labels[0])[:, 1]).long().to(device)
    val_node  = torch.from_numpy(np.array(labels[1])[:, 0]).long().to(device)
    val_label = torch.from_numpy(np.array(labels[1])[:, 1]).long().to(device)
    tst_node  = torch.from_numpy(np.array(labels[2])[:, 0]).long().to(device)
    tst_label = torch.from_numpy(np.array(labels[2])[:, 1]).long().to(device)

    num_classes = torch.max(trn_label).item() + 1

    data_split = (trn_node, trn_label, val_node, val_label, tst_node, tst_label)
    adjs = (adj_GTN, adj_graphsage, nx_graph)

    edge_list = list(nx_graph.edges)
    edge_typer  = nx.get_edge_attributes(nx_graph, 'type')
    edge_type   = []
    node_type_i = []
    node_type_j = []
    nodetype_2_id = {'P': 0, 'A': 1, 'S': 2}
    for edge in edge_list:
        edge_type_str = edge_typer[edge]
        edge_type_int = edgetype_2_id[edge_type_str]
        edge_type.append(edge_type_int)
        node_type_i.append(nodetype_2_id[edge_type_str[0]])
        node_type_j.append(nodetype_2_id[edge_type_str[1]])

    assert len(edge_type) == len(edge_list) == len(node_type_i) == len(node_type_j)
    edge_type   = torch.LongTensor(edge_type).to(device)
    node_type_i = torch.LongTensor(node_type_i).to(device)
    node_type_j = torch.LongTensor(node_type_j).to(device)

    print('{} is loaded!'.format(dataset_name))

    return num_nodes, num_classes, node_features, data_split, adjs, \
           edge_list, edge_type, node_type_i, node_type_j, n_edge_type


if __name__ == '__main__':
    data_dir = '../data'
    dataset_name = 'acm'
    assert dataset_name.lower() in ['acm', 'imdb', 'dblp']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes, num_classes, node_features, data_split, adjs = \
        data_loader(data_dir, dataset_name, device)
