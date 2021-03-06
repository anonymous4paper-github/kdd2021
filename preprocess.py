import os
import numpy as np
from scipy import io
from scipy.sparse import csr_matrix
import pickle

def preprocess(raw_data_dir, save_dir, dataset_name='ACM'):
    '''
    Data preprocessing.
    The raw data is from the NeurIPS 2019 paper "Graph Transformer Networks".
    Take acm as a demo.
    '''
    raw_file_name = dataset_name.upper() + '.mat'
    raw_data_path = os.path.join(raw_data_dir, raw_file_name)
    mat_file = io.loadmat(raw_data_path)
    paper_conf = mat_file['PvsC'].nonzero()[1]

    '''[0,1,9,10,13] : KDD, SIGMOD, SIGCOMM, MobiCOMM, VLDB'''
    # DataBase
    paper_db = np.isin(paper_conf, [1, 13])
    paper_db_idx = np.where(paper_db == True)[0]
    paper_db_idx = np.sort(np.random.choice(paper_db_idx, 994, replace=False))
    # Data Mining
    paper_dm = np.isin(paper_conf, [0])
    paper_dm_idx = np.where(paper_dm == True)[0]
    # Wireless Communication
    paper_wc = np.isin(paper_conf, [9, 10])
    paper_wc_idx = np.where(paper_wc == True)[0]

    paper_idx = np.sort(list(paper_db_idx) + list(paper_dm_idx) + list(paper_wc_idx))

    # 0 : database, 1: wireless communication, 2: data mining
    paper_target = []
    for idx in paper_idx:
        if idx in paper_db_idx:
            paper_target.append(0)
        elif idx in paper_wc_idx:
            paper_target.append(1)
        else:
            paper_target.append(2)
    paper_target = np.array(paper_target)

    ## Edges: PA, AP, PS, SP
    authors = mat_file['PvsA'][paper_idx].nonzero()[1]
    author_dic = {}
    re_authors = []
    for author in authors:
        if author not in author_dic:
            author_dic[author] = len(author_dic) + len(paper_idx)
        re_authors.append(author_dic[author])
    re_authors = np.array(re_authors)

    subjects = mat_file['PvsL'][paper_idx].nonzero()[1]
    subject_dic = {}
    re_subjects = []
    for subject in subjects:
        if subject not in subject_dic:
            subject_dic[subject] = len(subject_dic) + len(paper_idx) + len(author_dic)
        re_subjects.append(subject_dic[subject])
    re_subjects = np.array(re_subjects)

    node_num = len(paper_idx) + len(author_dic) + len(subject_dic)

    papers = mat_file['PvsA'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_pa = csr_matrix((data, (papers, re_authors)), shape=(node_num, node_num))

    papers = mat_file['PvsL'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_ps = csr_matrix((data, (papers, re_subjects)), shape=(node_num, node_num))

    A_ap = A_pa.transpose()
    A_sp = A_ps.transpose()
    edges = [A_pa, A_ap, A_ps, A_sp]
    edge_save_dir = os.path.join(save_dir, dataset_name.upper())
    if not os.path.exists(edge_save_dir):
        os.makedirs(edge_save_dir)
    edge_save_path = os.path.join(edge_save_dir, 'edges.pkl')
    with open(edge_save_path, 'wb') as f:
        pickle.dump(edges, f)
    print('Save {} done!'.format(edge_save_path))

    ## Node feature
    terms = mat_file['TvsP'].transpose()[paper_idx].nonzero()[1]
    term_dic = {}
    re_terms = []
    for term in terms:
        if term not in term_dic:
            term_dic[term] = len(term_dic) + len(paper_idx) + len(author_dic) + len(subject_dic)
        re_terms.append(term_dic[term])
    re_terms = np.array(re_terms)

    mat_file['TvsP'].transpose()


    tmp_num_node = node_num + len(term_dic)
    papers = mat_file['PvsA'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_pa_tmp = csr_matrix((data, (papers, re_authors)), shape=(tmp_num_node,tmp_num_node))
    papers = mat_file['PvsL'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_ps_tmp = csr_matrix((data, (papers, re_subjects)), shape=(tmp_num_node,tmp_num_node))
    papers = mat_file['PvsT'][paper_idx].nonzero()[0]
    data = np.ones_like(papers)
    A_pt_tmp = csr_matrix((data, (papers, re_terms)), shape=(tmp_num_node,tmp_num_node))

    paper_feat   = np.array(A_pt_tmp[:len(paper_idx),-len(term_dic):].toarray()>0, dtype=np.int)
    author_feat  = np.array(A_pa_tmp.transpose().dot(A_pt_tmp)[len(paper_idx):len(paper_idx)+len(author_dic),-len(term_dic):].toarray()>0, dtype=np.int)
    subject_feat = np.array(A_ps_tmp.transpose().dot(A_pt_tmp)[len(paper_idx)+len(author_dic):len(paper_idx)+len(author_dic)+len(subject_dic),-len(term_dic):].toarray()>0, dtype=np.int)

    node_feature = np.concatenate((paper_feat, author_feat, subject_feat))
    feature_save_dir = os.path.join(save_dir, dataset_name.upper())
    if not os.path.exists(feature_save_dir):
        os.makedirs(feature_save_dir)
    feature_save_path = os.path.join(feature_save_dir, 'node_features.pkl')
    with open(feature_save_path, 'wb') as f:
        pickle.dump(node_feature, f)
    print('Save {} done!'.format(feature_save_path))

    ## Label
    train_valid_DB = list(np.random.choice(np.where(paper_target == 0)[0], 300, replace=False))
    train_valid_WC = list(np.random.choice(np.where(paper_target == 1)[0], 300, replace=False))
    train_valid_DM = list(np.random.choice(np.where(paper_target == 2)[0], 300, replace=False))

    train_idx    = np.array(train_valid_DB[:200] + train_valid_WC[:200] + train_valid_DM[:200])
    train_target = paper_target[train_idx]
    train_label  = np.vstack((train_idx, train_target)).transpose()
    valid_idx    = np.array(train_valid_DB[200:] + train_valid_WC[200:] + train_valid_DM[200:])
    valid_target = paper_target[valid_idx]
    valid_label  = np.vstack((valid_idx, valid_target)).transpose()
    test_idx     = np.array(list((set(np.arange(paper_target.shape[0])) - set(train_idx)) - set(valid_idx)))
    test_target  = paper_target[test_idx]
    test_label   = np.vstack((test_idx, test_target)).transpose()

    labels = [train_label, valid_label, test_label]
    label_save_dir = os.path.join(save_dir, dataset_name.upper())
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    label_save_path = os.path.join(label_save_dir, 'labels.pkl')
    with open(label_save_path, 'wb') as f:
        pickle.dump(labels, f)
    print('Save {} done!'.format(label_save_path))


if __name__ == '__main__':
    raw_data_dir = './raw_data'
    dataset_name = 'ACM'
    save_dir = './preprocessed_data'
    preprocess(raw_data_dir, save_dir, dataset_name)
