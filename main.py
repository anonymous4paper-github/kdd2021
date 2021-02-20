from __future__ import division
from __future__ import print_function

import os
import argparse
import copy
import random
import numpy as np
from sklearn.metrics import f1_score
from termcolor import cprint

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import sys
from utils.data_loader import data_loader
from utils.common_tools import print_args, set_seed
from utils.graph_utils import NeibSampler

from dgl import DGLGraph
from dgl.data import register_data_args
import networkx as nx
from models import GenGNN

def args_parse():
    # Training settings
    parser = argparse.ArgumentParser()

    # General configs
    parser.add_argument("--dataset_name", default="acm",
                        help='Which dataset to use')
    parser.add_argument("--data_dir", type=str, default="./preprocessed_data",
                        help="The path of the preprocessed dataset")
    parser.add_argument("--gen_type", default="gcn",
                        help="Generative model settings")
    parser.add_argument("--post_type", default="a2gnn",
                        help="Posterior model settings")
    parser.add_argument('--gpuid', type=int, default=0,
                        help="Which gpu to use")
    parser.add_argument("--result_path", default="./results",
                        help="Dir to save results")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of epochs to train")
    parser.add_argument("--patience", type=int, default=60,
                        help="Early stopping patience")
    parser.add_argument("--self_loop", action='store_true',
                        help="Graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)

    # Common hyper-parameters
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Number of hidden units")
    parser.add_argument("--activation", default="relu",
                        help="Non-linear activation function")

    # GNN hyper-parameters
    parser.add_argument("--n_gnn_layers", type=int, default=4,
                        help="Number of (common) GNN layers")
    parser.add_argument('--a2gnn_num_layer', type=int, default=3,
                        help='Number of conv layers in a2gnn')
    parser.add_argument("--hidden_x", type=int, default=16,
                        help="Number of hidden units for x_enc")
    parser.add_argument("--K", type=int, default=7,
                        help="Number of aspects")
    parser.add_argument('--aspect_embed_size', type=int, default=16,
                        help='Size of the aspect embedding')
    parser.add_argument('--edgenode_it', type=int, default=6,
                        help='Number of iterations among aspect-aware calculating')
    parser.add_argument('--n_nb', type=int, default=20,
                        help='(Max) size of the sampled neighborhood')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--aggregator_type', type=str, default='gcn',
                        help='Aggregator type of the SAGE model. (gcn | pool | lstm | mean)')

    # Other hyper-parameters
    parser.add_argument("--lamda", type=float, default=0.1,
                        help="Lambda coefficient for nll_discriminative")
    parser.add_argument("--neg_ratio", type=float, default=1.0,
                        help="Negative sample ratio")
    parser.add_argument("--mask_rate", type=float, default=0.3,
                        help="Mask ratio for the self-supervised learning")
    parser.add_argument("--run_times", type=int, default=50,
                        help="How many times to run the experiments for average eval")
    
    args = parser.parse_args()
    return args



def main_hin(args):
    '''
        The main function to run. (train / val / test)
    '''
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    assert dataset_name.lower() in ['acm', 'imdb', 'dblp']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes, num_classes, node_features, data_split, adjs, \
    edge_list, edge_type, node_type_i, node_type_j, n_edge_type = \
        data_loader(data_dir, dataset_name, device)

    trn_node, trn_label, val_node, val_label, tst_node, tst_label = data_split
    adj_GTN, adj_graphsage, nx_graph = adjs

    in_feats = node_features.size(1)
    g = nx_graph.to_directed()

    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    dgl_g = DGLGraph(g)
    n_edges = dgl_g.number_of_edges()
    degs = dgl_g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.to(device)
    dgl_g.ndata['norm'] = norm.unsqueeze(1)

    model_args = {
        "g"         : dgl_g,
        "in_feats"  : in_feats,
        "n_classes" : num_classes,
        "n_hidden"  : args.hidden,
        "dropout"   : args.dropout,
        "activation": F.relu,
    }

    gen_type   = args.gen_type
    post_type  = args.post_type

    gen_config = copy.deepcopy(model_args)
    gen_config["type"]               = gen_type
    gen_config["neg_ratio"]          = args.neg_ratio
    gen_config['hidden_x']           = args.hidden_x
    gen_config['aspect_embed_size']  = args.aspect_embed_size
    gen_config['nx_g']               = g
    gen_config['n_edge_type']        = n_edge_type
    gen_config['n_layers']           = args.n_gnn_layers

    post_config = copy.deepcopy(model_args)
    post_config["type"]              = post_type
    post_config['aspect_embed_size'] = args.aspect_embed_size
    
    if post_type == 'graphsage':
        post_config['n_layers']        = args.n_gnn_layers
        post_config['aggregator_type'] = args.aggregator_type
    elif post_type == 'a2gnn':
        post_config['a2gnn_num_layer'] = args.a2gnn_num_layer
        post_config['args']            = args
    else:
        post_config['n_layers']        = args.n_gnn_layers

    model = GenGNN(gen_config, post_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [80, 120, 150], 0.6)

    neib_sampler = NeibSampler(nx_graph, args.n_nb).to(device)

    best_val_macro_f1 = -1
    best_tst_macro_f1 = -1
    best_val_micro_f1 = -1
    best_tst_micro_f1 = -1
    for epoch in range(args.n_epochs):
        is_new_best = False
        model.train()
        aspect_embed, logits = model.cal_post(node_features, neib_sampler)
        post_aspect = F.log_softmax(aspect_embed, dim=1)
        y_log_prob  = F.log_softmax(logits, dim=1)
        nll_generative = model.gen.nll_generative(node_features,
                                                  post_aspect,
                                                  trn_node,
                                                  trn_label)
        mask_rate = args.mask_rate
        ss_loss = model.gen.self_supervised(node_features,
                                            aspect_embed,
                                            edge_type,
                                            mask_rate)
        nll_discriminative = F.nll_loss(y_log_prob[trn_node], trn_label)
        trn_loss = args.lamda * (nll_generative + ss_loss) + nll_discriminative

        trn_logits   = logits[trn_node]
        val_logits   = logits[val_node]
        tst_logits   = logits[tst_node]
        trn_logits_  = trn_logits
        val_logits_  = val_logits
        tst_logits_  = tst_logits

        trn_label_   = trn_label.cpu().numpy()
        val_label_   = val_label.cpu().numpy()
        tst_label_   = tst_label.cpu().numpy()
        trn_pred     = trn_logits_.cpu().detach().numpy().argmax(axis=1)
        val_pred     = val_logits_.cpu().detach().numpy().argmax(axis=1)
        tst_pred     = tst_logits_.cpu().detach().numpy().argmax(axis=1)

        trn_macro_f1 = f1_score(trn_label_, trn_pred, average="macro")
        trn_micro_f1 = f1_score(trn_label_, trn_pred, average="micro")
        val_macro_f1 = f1_score(val_label_, val_pred, average='macro')
        val_micro_f1 = f1_score(val_label_, val_pred, average='micro')
        tst_macro_f1 = f1_score(tst_label_, tst_pred, average="macro")
        tst_micro_f1 = f1_score(tst_label_, tst_pred, average="micro")

        if val_macro_f1 > best_val_macro_f1:
            is_new_best = True
            best_val_macro_f1 = val_macro_f1
            best_tst_macro_f1 = tst_macro_f1
        if val_micro_f1 > best_val_micro_f1:
            is_new_best = True
            best_val_micro_f1 = val_micro_f1
            best_tst_micro_f1 = tst_micro_f1

        optimizer.zero_grad()
        trn_loss.backward()
        optimizer.step()
        scheduler.step()

        if is_new_best:
            cprint('epoch:{:>3d}/{}  trn_loss: {:.5f}  trn_macro_f1: {:.4f} | val_macro_f1: {:.4f} | tst_macro_f1: {:.4f}'.format(
                    epoch, args.n_epochs, trn_loss.item(), trn_macro_f1, val_macro_f1, tst_macro_f1), 'green')
        else:
            print('epoch:{:>3d}/{}  trn_loss: {:.5f}  trn_macro_f1: {:.4f} | val_macro_f1: {:.4f} | tst_macro_f1: {:.4f}'.format(
                   epoch, args.n_epochs, trn_loss.item(), trn_macro_f1, val_macro_f1, tst_macro_f1))

    return best_tst_macro_f1, best_tst_micro_f1



if __name__ == '__main__':
    args = args_parse()
    print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    set_seed(args.seed)

    torch.autograd.set_detect_anomaly(True)

    best_tst_macro_f1_list = []
    best_tst_micro_f1_list = []
    run_times = args.run_times
    for i in range(run_times):
        print('{}-th run'.format(i + 1))
        best_tst_macro_f1, best_tst_micro_f1 = main_hin(args)
        best_tst_macro_f1_list.append(best_tst_macro_f1)
        best_tst_micro_f1_list.append(best_tst_micro_f1)

    print('SS-PGM on {} [average test performance over {} runs] '
          'macro-f1 score: {:.5f}  micro-f1 score: {:.5f}'
          .format(args.dataset_name, run_times,
                  np.mean(best_tst_macro_f1_list),
                  np.mean(best_tst_micro_f1_list)))