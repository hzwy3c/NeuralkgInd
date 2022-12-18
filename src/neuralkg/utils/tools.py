import os
import dgl
import lmdb
import time
import yaml
import json
import queue
import torch
import pickle
import struct
import random
import logging
import datetime
import importlib
import numpy as np
import networkx as nx
import scipy.sparse as ssp
import multiprocessing as mp
from tqdm import tqdm
from scipy.special import softmax
from scipy.sparse import csc_matrix

def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'model.TransE'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def save_config(args):
    args.save_config = False  #防止和load_config冲突，导致把加载的config又保存了一遍
    if not os.path.exists("config"):
        os.mkdir("config")
    config_file_name = time.strftime(str(args.model_name)+"_"+str(args.dataset_name)) + ".yaml"
    day_name = time.strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join("config", day_name)):
        os.makedirs(os.path.join("config", day_name))
    config = vars(args)
    with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))

def load_config(args, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        args.__dict__.update(config)
    return args

def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    date_file = os.path.join(args.save_path, date)

    if not os.path.exists(date_file):
        os.makedirs(date_file)

    hour = str(int(dt.strftime("%H")) + 8)
    name = hour + dt.strftime("_%M_%S")
    if args.special_name != None:
        name = args.special_name
    log_file = os.path.join(date_file,  "_".join([args.model_name, args.dataset_name, name, 'train.log']))

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=logging.INFO,
        datefmt='%m-%d %H:%M',
        filename=log_file,
        filemode='a'
    )

def log_metrics(epoch, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s: %.4f at epoch %d' % (metric, metrics[metric], epoch))

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def reidx_withr_ande(tri, rel_reidx, ent_reidx):
    tri_reidx = []
    for h, r, t in tri:
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx

def reidx(tri):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0
    rel_reidx = dict()
    relidx = 0
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        if r not in rel_reidx.keys():
            rel_reidx[r] = relidx
            relidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx, dict(rel_reidx), dict(ent_reidx)

def reidx_withr(tri, rel_reidx):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx, dict(ent_reidx)

def data2pkl(dataset_name):
    train_tri = []
    file = open('./dataset/{}/train.txt'.format(dataset_name))
    train_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    valid_tri = []
    file = open('./dataset/{}/valid.txt'.format(dataset_name))
    valid_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    test_tri = []
    file = open('./dataset/{}/test.txt'.format(dataset_name))
    test_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    train_tri, fix_rel_reidx, ent_reidx = reidx(train_tri)
    valid_tri = reidx_withr_ande(valid_tri, fix_rel_reidx, ent_reidx)
    test_tri = reidx_withr_ande(test_tri, fix_rel_reidx, ent_reidx)

    file = open('./dataset/{}_ind/train.txt'.format(dataset_name))
    ind_train_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    file = open('./dataset/{}_ind/valid.txt'.format(dataset_name))
    ind_valid_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    file = open('./dataset/{}_ind/test.txt'.format(dataset_name))
    ind_test_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    test_train_tri, ent_reidx_ind = reidx_withr(ind_train_tri, fix_rel_reidx)
    test_valid_tri = reidx_withr_ande(ind_valid_tri, fix_rel_reidx, ent_reidx_ind)
    test_test_tri = reidx_withr_ande(ind_test_tri, fix_rel_reidx, ent_reidx_ind)

    save_data = {'train_graph': {'train': train_tri, 'valid': valid_tri, 'test': test_tri,
                                 'rel2idx': fix_rel_reidx, 'ent2idx': ent_reidx},
                 'ind_test_graph': {'train': test_train_tri, 'valid': test_valid_tri, 'test': test_test_tri,
                                    'rel2idx': fix_rel_reidx, 'ent2idx': ent_reidx_ind}}

    pickle.dump(save_data, open(f'./dataset/{dataset_name}.pkl', 'wb'))

def gen_subgraph_datasets(args, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):
    testing = 'test' in splits
    if testing:
        adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel = load_ind_data_grail(args)
    else:
        adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel, _, _, _, _ = load_data_grail(args)

    graphs = {}
    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': args.max_links}

    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], args.num_neg_samples_per_link,
                                                max_size=split['max_size'], constrained_neg_prob=args.constrained_neg_prob)

    links2subgraphs(adj_list, graphs, args, max_label_value, testing)

def load_ind_data_grail(args):
    data = pickle.load(open(args.pk_path, 'rb'))

    splits = ['train', 'test']

    triplets = {}
    for split_name in splits:
        triplets[split_name] = np.array(data['ind_test_graph'][split_name])[:, [0, 2, 1]]

    train_rel2idx = data['ind_test_graph']['rel2idx']
    train_ent2idx = data['ind_test_graph']['ent2idx']
    train_idx2rel = {i: r for r, i in train_rel2idx.items()}
    train_idx2ent = {i: e for e, i in train_ent2idx.items()}

    adj_list = []
    for i in range(len(train_rel2idx)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                shape=(len(train_ent2idx), len(train_ent2idx))))

    return adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel 

def load_data_grail(args, add_traspose_rels=False):
    data = pickle.load(open(args.pk_path, 'rb'))

    splits = ['train', 'valid']

    triplets = {}
    for split_name in splits:
        triplets[split_name] = np.array(data['train_graph'][split_name])[:, [0, 2, 1]]

    train_rel2idx = data['train_graph']['rel2idx']
    train_ent2idx = data['train_graph']['ent2idx']
    train_idx2rel = {i: r for r, i in train_rel2idx.items()}
    train_idx2ent = {i: e for e, i in train_ent2idx.items()}
    
    h2r = {}
    t2r = {}
    m_h2r = {}
    m_t2r = {}
    if args.model_name == 'SNRI':
        # Construct the the neighbor relations of each entity
        num_rels = len(train_idx2rel)
        num_ents = len(train_idx2ent)
        h2r = {}
        h2r_len = {}
        t2r = {}
        t2r_len = {}
        
        for triplet in triplets['train']:
            h, t, r = triplet
            if h not in h2r:
                h2r_len[h] = 1
                h2r[h] = [r]
            else:
                h2r_len[h] += 1
                h2r[h].append(r)
            
            if args.add_traspose_rels:
                # Consider the reverse relation, the id of reverse relation is (relation + #relations)
                if t not in t2r:
                    t2r[t] = [r + num_rels]
                else:
                    t2r[t].append(r + num_rels)
            if t not in t2r:
                t2r[t] = [r]
                t2r_len[t]  = 1
            else:
                t2r[t].append(r)
                t2r_len[t] += 1

        # Construct the matrix of ent2rels
        h_nei_rels_len = int(np.percentile(list(h2r_len.values()), 75))
        t_nei_rels_len = int(np.percentile(list(t2r_len.values()), 75))
        logging.info("Average number of relations each node: ", "head: ", h_nei_rels_len, 'tail: ', t_nei_rels_len)
        
        # The index "num_rels" of relation is considered as "padding" relation.
        # Use padding relation to initialize matrix of ent2rels.
        m_h2r = np.ones([num_ents, h_nei_rels_len]) * num_rels
        for ent, rels in h2r.items():
            if len(rels) > h_nei_rels_len:
                rels = np.array(rels)[np.random.choice(np.arange(len(rels)), h_nei_rels_len)]
                m_h2r[ent] = rels
            else:
                rels = np.array(rels)
                m_h2r[ent][: rels.shape[0]] = rels      
        
        m_t2r = np.ones([num_ents, t_nei_rels_len]) * num_rels
        for ent, rels in t2r.items():
            if len(rels) > t_nei_rels_len:
                rels = np.array(rels)[np.random.choice(np.arange(len(rels)), t_nei_rels_len)]
                m_t2r[ent] = rels
            else:
                rels = np.array(rels)
                m_t2r[ent][: rels.shape[0]] = rels

        # Sort the data according to relation id 
        if args.sort_data:
            triplets['train'] = triplets['train'][np.argsort(triplets['train'][:,2])]

    adj_list = []
    for i in range(len(train_rel2idx)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                shape=(len(train_ent2idx), len(train_ent2idx))))

    return adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel, h2r, m_h2r, t2r, m_t2r

def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size

def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)

def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across reelations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]
    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)

    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges

def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)

def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value

def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)

def links2subgraphs(A, graphs, params, max_label_value=None, testing=False):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM
    
    if testing:
        env = lmdb.open(params.test_db_path, map_size=map_size, max_dbs=6)
    else:
        env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)

        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))

def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes

def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes

def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]

def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)

def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)

def bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)

def get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors

def sp_row_vec_from_idx_list(idx_list, dim):

    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)

def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl

def gen_subgraph_datasets_directed(args):
    adj_list, triplets, train_ent2idx, train_rel2idx, train_idx2ent, train_idx2rel = load_data_grail(args)

    splits = ['train', 'valid']
    graphs = {}
    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': args.max_links}

    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], args.num_neg_samples_per_link,
                                                max_size=split['max_size'], constrained_neg_prob=args.constrained_neg_prob)

    links2subgraphs_directed(adj_list, graphs, args)

def links2subgraphs_directed(A, graphs, params, max_label_value=None):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM

    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_directed_subgraph, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing directed subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)

        logging.info(f"Extracting enclosing directed subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))

def extract_save_directed_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = directed_subgraph_extraction_labeling((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)

def directed_subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes

def parallel_worker3(x):
    return undirected_bfs3(*x)

def undirected_bfs3(graph, graph_inverse, source, nbd_size=2):
        visit = {}
        distance = {}
        visit[source] = 1
        distance[source] = 0
        q = queue.Queue()
        q.put((source, -1))
        all_neighbors = []
        all_neighbors.append(source)
        all_relations = []

        neighbors_inverse_1hop = []
        neighbors_inverse_1hop.append(source)

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys() and distance[top[0]] < nbd_size:
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                       # if np.array([top[0], target, graph[top[0]][target]]) not in all_relations:
                        if [top[0], target, graph[top[0]][target]] not in all_relations:
                              all_relations.append([top[0], target, graph[top[0]][target]])
                        
                    else:
                        visit[target] = 1
                        distance[target] = distance[top[0]] + 1
                        if distance[target] < 2:
                             neighbors_inverse_1hop.append(target)
                        all_neighbors.append(target)
                        all_relations.append([top[0], target, graph[top[0]][target]])
                        q.put((target, graph[top[0]][target]))

            if top[0] in graph_inverse.keys() and distance[top[0]] < nbd_size:
                for target in graph_inverse[top[0]].keys():
                    if(target in visit.keys()):
                        if [target, top[0], graph_inverse[top[0]][target]] not in all_relations:
                              all_relations.append([target, top[0], graph_inverse[top[0]][target]])
                        
                    else:
                        visit[target] = 1
                        distance[target] = distance[top[0]] + 1
                        if distance[target] < 2:
                             neighbors_inverse_1hop.append(target)
                        all_neighbors.append(target)
                        all_relations.append([target, top[0], graph_inverse[top[0]][target]])
                        q.put((target, graph_inverse[top[0]][target]))


        return np.array(all_neighbors), all_relations, distance, neighbors_inverse_1hop, source

def parallel_worker2(x):
    return extract_undirected_subgraph2(*x)

def extract_undirected_subgraph2(source, target, relation, neighbor_a, triple_a, distance_a, neighbor_b, triple_b, distance_b, inverse_1hop, hop=3):
        subgraph = []
        node = set()
        neighbor_a = set(neighbor_a); neighbor_b = set(neighbor_b); inverse = set(inverse_1hop)
        common = list(neighbor_a.intersection(inverse))
        graph = {}
        unfinished = True
        
        if len(common):
            common = list(neighbor_a.intersection(neighbor_b))
            if source not in common:
                common.append(source)
            if target not in common:
                common.append(target)      
            for j in  range(len(triple_a)):
                if (triple_a[j][0] in common) and (triple_a[j][1] in common) and (triple_a[j] != [source, target, relation]):
                      subgraph.append(triple_a[j])
                      node.add(triple_a[j][0])
                      node.add(triple_a[j][1])
                      if len(node)>100 and (source in node) and (target in node):
                           unfinished = False
                           break 
            if unfinished:
                for j in  range(len(triple_b)):
                    if (triple_b[j][1]  in common)  and (triple_b[j][0] in common) and (triple_b[j] not in subgraph) and (triple_b[j] != [source, target, relation]):
                         subgraph.append(triple_b[j])
                         node.add(triple_b[j][1])
                         node.add(triple_b[j][0])
                         if len(node)>100 and (source in node) and (target in node):
                              break

        '''
        if len(subgraph)==0:
            common = list(neighbor_a.intersection(neighbor_b))
            if len(common):
               # distance_a = undirected_distance[source]; distance_b = undirected_distance[target]
                common = list(neighbor_a.intersection(neighbor_b))
                if source not in common:
                    common.append(source)
                if target not in common:
                    common.append(target) 
                for j in  range(len(triple_a)):
                     if (triple_a[j][0] in common) and (triple_a[j][1] in common):
                         subgraph.append(triple_a[j])
                         node.add(triple_a[j][0])
                         node.add(triple_a[j][1])
                         if len(node)>50:
                              unfinished = False
                              break

            if common:
                for j in  range(len(triple_b)):
                    if (triple_b[j][1] in common)  and (triple_b[j][0] in common) and (triple_b[j] not in subgraph):
                         subgraph.append(triple_b[j])
                         node.add(triple_b[j][1])
                         node.add(triple_b[j][0])
                         if len(node)>100:
                              break
        '''
        node.add(source)
        node.add(target)
       # print(len(subgraph))
        node = list(node); distance_source = []; distance_target = []
       
        for nodes in node:
           
            if nodes == source:
                distance_source.append(0)
                distance_target.append(1)
                continue
            if nodes == target:
                distance_source.append(1)
                distance_target.append(0)
                continue
           
            if nodes not in distance_a.keys():
                distance_source.append(hop - distance_b[nodes])
            else:  
                distance_source.append(distance_a[nodes])
            if nodes not in distance_b.keys():
                distance_target.append(hop - distance_a[nodes])
            else:
                distance_target.append(distance_b[nodes])
        if len(node):
             distance_source = np.eye(hop+1)[distance_source]
             print(distance_source)
             distance_target = np.eye(hop+1)[distance_target]
             node = np.expand_dims(node, axis = 1)
             node_and_distance = np.concatenate([node, distance_source], axis = 1)
             node_and_distance = np.concatenate([node_and_distance, distance_target], axis = 1)
             graph['node'] = node_and_distance
        else:
             graph['node'] = np.array([])
        '''
        embedding_size = 32
        total_embed = []
        for nodes in node:
            if nodes not in distance_a.keys():
                u = hop - distance_b[nodes]
            else:  
                u = distance_a[nodes]
            if nodes not in distance_b.keys():
                s = hop - distance_a[nodes]
            else:
                s = distance_b[nodes]
        
            node_embed = np.random.normal(u, s, embedding_size)
            nodes = np.expand_dims(nodes, axis=0)
            node_and_embed = np.concatenate([nodes, node_embed], axis = 0)
            node_and_embed = np.expand_dims(node_and_embed, axis = 0)
            total_embed.append(node_and_embed)
        total_embed =  np.concatenate(total_embed, axis=0)
        graph['node'] = total_embed
        '''
        graph['edge'] = np.array(subgraph)
     #   graph['source'] = source; graph['target'] = target

        return graph, source, target

class Corpus:
    def __init__(self, args, train_data, validation_data, test_data, entity2id,
                 relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train, unique_entities_validation, unique_entities_test, get_2hop=False):
        self.train_triples = train_data[0]

        # Converting to sparse tensor
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix = (adj_indices, adj_values)#((row,col),data)
        self.unique_entities_train = unique_entities_train 
        self.graph = self.get_graph(self.train_adj_matrix)
        self.inverse_graph = self.get_inverse_graph(self.train_adj_matrix)

        adj_indices_val = torch.LongTensor(
            [validation_data[1][0], validation_data[1][1]])  # rows and columns
        adj_values_val = torch.LongTensor(validation_data[1][2])
        self.val_adj_matrix = (adj_indices_val, adj_values_val)
        self.val_graph = self.get_graph(self.val_adj_matrix)
        self.inverse_val_graph = self.get_inverse_graph(self.val_adj_matrix)
        self.unique_entities_validation = unique_entities_validation

        adj_indices_test = torch.LongTensor(
            [test_data[1][0], test_data[1][1]])  # rows and columns
        adj_values_test = torch.LongTensor(test_data[1][2])
        self.test_adj_matrix = (adj_indices_test, adj_values_test)
        self.test_graph = self.get_graph(self.test_adj_matrix)
        self.inverse_test_graph = self.get_inverse_graph(self.test_adj_matrix)
        self.unique_entities_test = unique_entities_test

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data
        self.validation_triples = validation_data[0]
        self.test_triples = test_data[0]

        self.headTailSelector = headTailSelector  # for selecting random entities
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        if(get_2hop):
           # self.graph = self.get_graph(self.train_adj_matrix)
            self.node_neighbors_2hop = self.get_further_neighbors()

        self.unique_entities_train = [self.entity2id[i]
                                      for i in unique_entities_train]

        self.unique_entities_validation = [self.entity2id[i]
                                      for i in unique_entities_validation]

        self.unique_entities_test = [self.entity2id[i]
                                      for i in unique_entities_test]



        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.validation_triples + self.test_triples)}
        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(len(self.valid_triples_dict), len(self.train_indices),
                                                                                                           len(self.validation_indices), len(self.test_indices)))
     #   self.total_train_data, self.train_label = self.get_total_train_data(8)
        # For training purpose
        self.batch_indices = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)


    def get_total_train_data(self, invalid_valid_ratio, neighbor_a, neighbor_b):
            batch_size = len(self.train_indices)
            self.batch_indices = np.empty((batch_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((batch_size * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            self.batch_indices[:batch_size, :] = self.train_indices
            self.batch_values[:batch_size, :] = self.train_values

            last_index = batch_size
            
            if invalid_valid_ratio > 0:

                self.batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_values[:last_index, :], (invalid_valid_ratio, 1))
                negative_sample = 0
                while negative_sample < last_index * invalid_valid_ratio:
                    if negative_sample%20 == 0:
                        print(negative_sample)
                    now_indice = random.choice([i for i in range(last_index)])
                    random_entity = random.choice(self.unique_entities_train)
                    if ((random_entity, self.batch_indices[now_indice, 1], self.batch_indices[now_indice, 2]) not in self.valid_triples_dict.keys()) and  self.common_neighbor(neighbor_a, neighbor_b, random_entity, self.batch_indices[now_indice, 2]) and (self.batch_indices[now_indice, 2]!=random_entity):
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    '''
                    elif np.random.uniform() < 0.:  ###################sample some negative zero graph
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    '''
                    if negative_sample>= last_index * invalid_valid_ratio:
                        break
                    random_entity = random.choice(self.unique_entities_train)
                    if ((self.batch_indices[now_indice, 0], self.batch_indices[now_indice, 1], random_entity) not in 
                      self.valid_triples_dict.keys())  and self.common_neighbor(neighbor_a, neighbor_b, self.batch_indices[now_indice, 0], random_entity) and (self.batch_indices[now_indice, 0]!=random_entity):
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    '''
                    elif np.random.uniform() < 0.:
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    '''
            return self.batch_indices, self.batch_values


    def get_total_train_data2(self, invalid_valid_ratio, neighbor_a, neighbor_b):
            batch_size = len(self.train_indices)
            self.batch_indices = np.empty((batch_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((batch_size * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            self.batch_indices[:batch_size, :] = self.train_indices
            self.batch_values[:batch_size, :] = self.train_values

            last_index = batch_size
            
            if invalid_valid_ratio > 0:
 
                self.batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_values[:last_index, :], (invalid_valid_ratio, 1))
                negative_sample = 0
                while negative_sample < last_index * invalid_valid_ratio:
                    if negative_sample%20 == 0:
                        print(negative_sample)
                    now_indice = random.choice([i for i in range(last_index)])
                    random_entity = random.choice(self.unique_entities_train)

                    if (random_entity, self.batch_indices[now_indice, 1], self.batch_indices[now_indice, 2]) not in self.valid_triples_dict.keys() and (not random_entity==self.batch_indices[now_indice, 2]):
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    if negative_sample>= last_index * invalid_valid_ratio:
                        break
                    random_entity = random.choice(self.unique_entities_train)

                    if (self.batch_indices[now_indice, 0], self.batch_indices[now_indice, 1], random_entity) not in  self.valid_triples_dict.keys() and (not random_entity==self.batch_indices[now_indice, 0]):
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
            return self.batch_indices, self.batch_values



    def get_total_val_data(self, invalid_valid_ratio, neighbor_a, neighbor_b):
            batch_size = len(self.validation_indices)
            self.batch_indices = np.empty((batch_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((batch_size * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            self.batch_indices[:batch_size, :] = self.validation_indices
            self.batch_values[:batch_size, :] = self.validation_values

            last_index = batch_size
            
            if invalid_valid_ratio > 0:

                self.batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_values[:last_index, :], (invalid_valid_ratio, 1))
                negative_sample = 0
                while negative_sample < last_index * invalid_valid_ratio:
                    if negative_sample%20 == 0:
                        print(negative_sample)
                    now_indice = random.choice([i for i in range(last_index)])
                    random_entity = random.choice(self.unique_entities_train)
                    if (random_entity, self.batch_indices[now_indice, 1], self.batch_indices[now_indice, 2]) not in self.valid_triples_dict.keys() and (self.batch_indices[now_indice, 2]!=random_entity)   and self.common_neighbor(neighbor_a, neighbor_b, random_entity, self.batch_indices[now_indice, 2]):
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]


                    if negative_sample>= last_index * invalid_valid_ratio:
                        break
                    random_entity = random.choice(self.unique_entities_train)
                    if (self.batch_indices[now_indice, 0], self.batch_indices[now_indice, 1], random_entity) not in self.valid_triples_dict.keys()   and (self.batch_indices[now_indice, 0]!=random_entity)   and self.common_neighbor(neighbor_a, neighbor_b, self.batch_indices[now_indice, 0], random_entity):
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1


            return self.batch_indices, self.batch_values




    def get_total_val_data2(self, invalid_valid_ratio, neighbor_a, neighbor_b):
            batch_size = len(self.validation_indices)
            self.batch_indices = np.empty((batch_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((batch_size * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            self.batch_indices[:batch_size, :] = self.validation_indices
            self.batch_values[:batch_size, :] = self.validation_values

            last_index = batch_size
            
            if invalid_valid_ratio > 0:

                self.batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_values[:last_index, :], (invalid_valid_ratio, 1))
                negative_sample = 0
                while negative_sample < last_index * invalid_valid_ratio:
                    if negative_sample%20 == 0:
                        print(negative_sample)
                    now_indice = random.choice([i for i in range(last_index)])
                    random_entity = random.choice(self.unique_entities_train)
                    if (random_entity, self.batch_indices[now_indice, 1], self.batch_indices[now_indice, 2]) not in self.valid_triples_dict.keys() and (self.batch_indices[now_indice, 2]!=random_entity):
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]


                    if negative_sample>= last_index * invalid_valid_ratio:
                        break
                    random_entity = random.choice(self.unique_entities_train)
                    if (self.batch_indices[now_indice, 0], self.batch_indices[now_indice, 1], random_entity) not in self.valid_triples_dict.keys()   and (self.batch_indices[now_indice, 0]!=random_entity):
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1


            return self.batch_indices, self.batch_values




    def get_test_all_data(self, unique_entities, neighbor_a, neighbor_b):

            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []
            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = np.tile(batch_indices[i, :], (len(entity_list), 1))    ############## to be determined
                new_x_batch_tail = np.tile(batch_indices[i, :], (len(entity_list), 1))

                if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1], new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys() or (not self.common_neighbor(neighbor_a, neighbor_b, new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][2])):
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],  new_x_batch_tail[tmp_index][2])
                    if temp_triple_tail in self.valid_triples_dict.keys() or (not self.common_neighbor(neighbor_a, neighbor_b, new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][2])):
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(new_x_batch_tail, last_index_tail, axis=0)

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(new_x_batch_head, 0, batch_indices[i], axis=0)
                new_x_batch_tail = np.insert(new_x_batch_tail, 0, batch_indices[i], axis=0)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail




    def get_test_all_data2(self, unique_entities, neighbor_a, neighbor_b, percent = 1, rank_number=100):

            start_time = time.time()
            indices = [i for i in range(len(self.test_indices))]
            indices = np.random.choice(indices, int(len(self.test_indices) * percent), replace=False)
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(batch_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []

            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = []    ############## to be determined
                new_x_batch_tail = []
                current_triple = batch_indices[i]
                new_x_batch_head.append(current_triple)
                new_x_batch_tail.append(current_triple)
                kk = 0
                for j in entity_list:
                    if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b, j,  current_triple[2]):
                       new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                       kk += 1
                       if kk>=rank_number:
                          break
                new_x_batch_head = np.array(new_x_batch_head)
                print(kk)
                kk = 0
                for j in entity_list:
                    if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b,  current_triple[0], j):
                       new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                       kk += 1
                       if kk>=rank_number:
                          break
                new_x_batch_tail = np.array(new_x_batch_tail)
                print(kk)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail



    def get_test_all_data3(self, unique_entities, neighbor_a, neighbor_b, percent = 1, rank_number=1000):

            start_time = time.time()
            indices = [i for i in range(len(self.test_indices))]
            indices = np.random.choice(indices, int(len(self.test_indices) * percent), replace=False)
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(batch_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []

            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = []    ############## to be determined
                new_x_batch_tail = []
                current_triple = batch_indices[i]
                new_x_batch_head.append(current_triple)
                new_x_batch_tail.append(current_triple)
                kk = 0; existed = []
               
                for j in entity_list:
                    if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b, j,  current_triple[2]) and (j != current_triple[2]):
                   # if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys():
                       new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                       kk += 1; existed.append(j)
                       if kk>=rank_number:
                          break
                '''
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[2]):
                         new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                         kk += 1; existed.append(j)
                '''
                if len(new_x_batch_head)>50:
                       new_x_batch_head2 = []
                       new_x_batch_head2.append(new_x_batch_head[0])
                       indices = [i for i in range(1, len(new_x_batch_head))]
                       index = list(np.random.choice(indices, 49, replace=False))
                       for kkk in index:
                           new_x_batch_head2.append(new_x_batch_head[kkk])
                      # new_x_batch_head2 = new_x_batch_head2 + new_x_batch_head[index]
                       print(len(new_x_batch_head2))
                       new_x_batch_head = new_x_batch_head2
                     
                new_x_batch_head = np.array(new_x_batch_head)
             #   print(kk)

                kk = 0; existed = []
                
                for j in entity_list:
                    if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b,  current_triple[0], j) and (j != current_triple[0]):
                   # if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys():
                       new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                       kk += 1; existed.append(j)
                       if kk>=rank_number:
                          break
 


                if len(new_x_batch_tail)>50:
                       new_x_batch_tail2 = []
                       new_x_batch_tail2.append(new_x_batch_tail[0])
                       indices = [i for i in range(1, len(new_x_batch_tail))]
                       index = list(np.random.choice(indices, 49, replace=False))
                       for kkk in index:
                           new_x_batch_tail2.append(new_x_batch_tail[kkk])
                       print(len(new_x_batch_tail2))
                       new_x_batch_tail = new_x_batch_tail2


                new_x_batch_tail = np.array(new_x_batch_tail)
               # print(kk)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail




    def get_test_all_data4(self, unique_entities, neighbor_a, neighbor_b, percent = 1, rank_number=50):

            start_time = time.time()
            indices = [i for i in range(len(self.test_indices))]
            indices = np.random.choice(indices, int(len(self.test_indices) * percent), replace=False)
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(batch_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []

            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = []    ############## to be determined
                new_x_batch_tail = []
                current_triple = batch_indices[i]
                new_x_batch_head.append(current_triple)
                new_x_batch_tail.append(current_triple)
                kk = 1; existed = []
               
  
               
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[2]):
                         new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                         kk += 1; existed.append(j)
               
                       
                new_x_batch_head = np.array(new_x_batch_head)
             #   print(kk)

                kk = 1; existed = []
                
                
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[0]):
                         new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                         kk += 1; existed.append(j)
               
                new_x_batch_tail = np.array(new_x_batch_tail)
               # print(kk)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail




    def get_test_all_data5(self, unique_entities, neighbor_a, neighbor_b, percent = 1, rank_number=1000):

            start_time = time.time()
            indices = [i for i in range(len(self.test_indices))]
            indices = np.random.choice(indices, int(len(self.test_indices) * percent), replace=False)
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(batch_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []
            all_test_triplet = []

            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = []    ############## to be determined
                new_x_batch_tail = []
                current_triple = batch_indices[i]
                new_x_batch_head.append(current_triple)
                new_x_batch_tail.append(current_triple)
                kk = 0; existed = []
               
                for j in entity_list:
                    if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b, j,  current_triple[2]) and (j != current_triple[2]):
                   # if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys():
                       new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                       kk += 1; existed.append(j)
                       if kk>=rank_number:
                          break
                '''
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[2]):
                         new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                         kk += 1; existed.append(j)
                '''
                if len(new_x_batch_head)>50:
                       new_x_batch_head2 = []
                       new_x_batch_head2.append(new_x_batch_head[0])
                       all_test_triplet.append(new_x_batch_head[0])
                       indices = [i for i in range(1, len(new_x_batch_head))]
                       index = list(np.random.choice(indices, 49, replace=False))
                       for kkk in index:
                           new_x_batch_head2.append(new_x_batch_head[kkk])
                           all_test_triplet.append(new_x_batch_head[kkk])
                      # new_x_batch_head2 = new_x_batch_head2 + new_x_batch_head[index]
                       print(len(new_x_batch_head2))
                       new_x_batch_head = new_x_batch_head2
                     
                new_x_batch_head = np.array(new_x_batch_head)
             #   print(kk)

                kk = 0; existed = []
                
                for j in entity_list:
                    if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b,  current_triple[0], j) and (j != current_triple[0]):
                   # if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys():
                       new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                       kk += 1; existed.append(j)
                       if kk>=rank_number:
                          break
                '''
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[0]):
                         new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                         kk += 1; existed.append(j)
                '''


                if len(new_x_batch_tail)>50:
                       new_x_batch_tail2 = []
                       new_x_batch_tail2.append(new_x_batch_tail[0])
                       all_test_triplet.append(new_x_batch_tail[0])
                       indices = [i for i in range(1, len(new_x_batch_tail))]
                       index = list(np.random.choice(indices, 49, replace=False))
                       for kkk in index:
                           new_x_batch_tail2.append(new_x_batch_tail[kkk])
                           all_test_triplet.append(new_x_batch_tail[kkk])

                       print(len(new_x_batch_tail2))
                       new_x_batch_tail = new_x_batch_tail2


                new_x_batch_tail = np.array(new_x_batch_tail)
               # print(kk)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail, np.array(all_test_triplet)






    def common_neighbor(self, neighbor_a, neighbor_b, a, b):
     #   common = [val for val in neighbor_a if val in neighbor_b]
        neighbor_a = neighbor_a[a]; neighbor_b = neighbor_b[b]
        common = False
        for x in neighbor_a:
            for y in neighbor_b:
                if x==y and x != a and x != b:
                   common = True
                   break
            if common:
                break
        return common
        '''
        if len(common):
           return True
        else:
           return False
        ''' 
 
    def get_batch_nhop_neighbors_undirected(self, batch_sources, graph, graph_inverse, nbd_size=2):
        batch_source_triples = {}
        batch_neighbor = {}
        batch_distance = {}; neighbor_inverse_1hop = {}
        print("length of unique_entities ", len(batch_sources))

        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker3, [(graph, graph_inverse, source, nbd_size) for source in batch_sources])
        remaining = results._number_left
       # pbar = tqdm(total=remaining)
        while True:
           # pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
      #  pbar.close()
        for neighbor, triples, distance, inverse_1hop, source in results:
             batch_source_triples[source] = triples
             batch_neighbor[source] = neighbor
             batch_distance[source] = distance
             neighbor_inverse_1hop[source] = inverse_1hop
        end = time.time()

        print('Time for extracting batch neighbor: ', end - start)

        return batch_neighbor, batch_source_triples, batch_distance,  neighbor_inverse_1hop


    def get_graph2(self, adj_matrix, unique_entities):
        graph = np.zeros([unique_entities, unique_entities])
        
        all_tiples = torch.cat([adj_matrix[0].transpose(
            0, 1), adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()
            graph[source][target] = value

        print("Graph created")
        return graph


    def get_graph(self, adj_matrix):
        graph = {}
        
        all_tiples = torch.cat([adj_matrix[0].transpose(
            0, 1), adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(source not in graph.keys()):
                graph[source] = {}
                graph[source][target] = value
            else:
                graph[source][target] = value
        print("Graph created")
        return graph




    def get_inverse_graph(self, adj_matrix):
        graph = {}
        
        all_tiples = torch.cat([adj_matrix[0].transpose(
            0, 1), adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(target not in graph.keys()):
                graph[target] = {}
                graph[target][source] = value
            else:
                graph[target][source] = value
        print("inverse graph created")
        return graph



    def get_batch_nhop_neighbors2(self, batch_sources, graph, graph_inverse, nbd_size=2):
        batch_source_triples = {}; inverse_source_triples = {}
        batch_neighbor = {}; inverse_neighbor = {}
        batch_distance = {}; inverse_distance = {}; neighbor_inverse_1hop = {}
        print("length of unique_entities ", len(batch_sources))

        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker2, [(graph, graph_inverse, source, nbd_size) for source in batch_sources])
        remaining = results._number_left
       # pbar = tqdm(total=remaining)
        while True:
           # pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get() #np.array(all_neighbors), all_relations, distance, np.array(all_neighbors_inverse), all_relations_inverse, distance_inverse, neighbors_inverse_1hop, source
        pool.close()
      #  pbar.close()
        for neighbor, triples, distance, neighbor_inverse, triples_inverse, distance_inverse, inverse_1hop, source in results:
             batch_source_triples[source] = triples
             batch_neighbor[source] = neighbor
             batch_distance[source] = distance
             inverse_source_triples[source] = triples_inverse
             inverse_neighbor[source] = neighbor_inverse
             inverse_distance[source] = distance_inverse
             neighbor_inverse_1hop[source] = inverse_1hop
        end = time.time()

        print('Time for extracting batch neighbor: ', end - start)

        return batch_neighbor, batch_source_triples, batch_distance, inverse_neighbor, inverse_source_triples, inverse_distance, neighbor_inverse_1hop



    def bfs(self, graph, source, nbd_size=2):
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > 2:
                            break
                        parent[target] = (top[0], graph[top[0]][target])

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if(distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):       ###### if not source
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]    #######([edge_value], [target])

        return neighbors

    def get_further_neighbors(self, nbd_size=2):
        neighbors = {}
        start_time = time.time()
        print("length of graph keys is ", len(self.graph.keys()))
        for source in self.graph.keys():
            # st_time = time.time()
            temp_neighbors = self.bfs(self.graph, source, nbd_size)
            for distance in temp_neighbors.keys():
                if(source in neighbors.keys()):
                    if(distance in neighbors[source].keys()):
                        neighbors[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[source][distance] = temp_neighbors[distance]
                else:
                    neighbors[source] = {}
                    neighbors[source][distance] = temp_neighbors[distance]

        print("time taken ", time.time() - start_time)

        print("length of neighbors dict is ", len(neighbors))
        return neighbors

    def get_batch_nhop_neighbors_all(self, args, batch_sources, node_neighbors, nbd_size=2):
        batch_source_triples = []
        print("length of unique_entities ", len(batch_sources))
        count = 0
        for source in batch_sources:
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():
                nhop_list = node_neighbors[source][nbd_size]

                for i, tup in enumerate(nhop_list):
                    if(args.partial_2hop and i >= 2):
                        break

                    count += 1
                    batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                                                 nhop_list[i][1][0]])     ######source, first_relation, last_relation, last_target 

        return np.array(batch_source_triples).astype(np.int32)
