#!/bin/bash

gpuid=0
data_dir='./preprocessed_data'
dataset_name='acm'
n_epochs=100
lr=0.01
hidden_x=16

gen_type='gcn'
post_type='a2gnn'
weight_decay=0.0036
lamda=0.05

K=7
n_gnn_layers=3
aggregator_type='mean'
a2gnn_num_layer=5


python main.py \
	--gpuid ${gpuid} \
	--data_dir ${data_dir} \
	--dataset_name ${dataset_name} \
	--n_epochs ${n_epochs} \
	--lr ${lr} \
	--hidden_x ${hidden_x} \
	--gen_type ${gen_type} \
	--post_type ${post_type} \
	--weight_decay ${weight_decay} \
    --lamda ${lamda} \
	--K ${K} \
    --n_gnn_layers ${n_gnn_layers} \
    --aggregator_type ${aggregator_type} \
    --a2gnn_num_layer ${a2gnn_num_layer} \
