import abc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import mean_nodes

class SNRI(nn.Module):
    def __init__(self, params):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        # self.relation2id = relation2id
        # self.ent2rels = ent2rels
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)

        # num_rels + 1 instead of nums_rels, in order to add a "padding" relation.
        self.rel_emb = nn.Embedding(self.params.num_rel + 1, self.params.inp_dim, sparse=False, padding_idx=self.params.num_rel)
        
        self.ent_padding = nn.Parameter(torch.FloatTensor(1, self.params.sem_dim).uniform_(-1, 1))
        if self.params.init_nei_rels == 'both':
            self.w_rel2ent = nn.Linear(2 * self.params.inp_dim, self.params.sem_dim)
        elif self.params.init_nei_rels == 'out' or 'in':
            self.w_rel2ent = nn.Linear(self.params.inp_dim, self.params.sem_dim)

        self.sigmoid = nn.Sigmoid()
        self.nei_rels_dropout = nn.Dropout(self.params.nei_rels_dropout)
        self.dropout = nn.Dropout(self.params.dropout)
        self.softmax = nn.Softmax(dim=1)

        if self.params.add_ht_emb:    
            # self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

        if self.params.comp_hrt:
            self.fc_layer = nn.Linear(2 * self.params.num_gcn_layers * self.params.emb_dim, 1)
        
        if self.params.nei_rel_path:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + 2 * self.params.emb_dim, 1)

        if self.params.comp_ht == 'mlp':
            self.fc_comp = nn.Linear(2 * self.params.emb_dim, self.params.emb_dim)

        if self.params.nei_rel_path:
            self.disc = Discriminator(self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim)
        else:
            self.disc = Discriminator(self.params.num_gcn_layers * self.params.emb_dim , self.params.num_gcn_layers * self.params.emb_dim)

        self.rnn = torch.nn.GRU(self.params.emb_dim, self.params.emb_dim, batch_first=True)

        self.batch_gru = BatchGRU(self.params.num_gcn_layers * self.params.emb_dim )

        self.W_o = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim * 2, self.params.num_gcn_layers * self.params.emb_dim)

    def init_ent_emb_matrix(self, g):
        """ Initialize feature of entities by matrix form """
        out_nei_rels = g.ndata['out_nei_rels']
        in_nei_rels = g.ndata['in_nei_rels']
        
        target_rels = g.ndata['r_label']
        out_nei_rels_emb = self.rel_emb(out_nei_rels)
        in_nei_rels_emb = self.rel_emb(in_nei_rels)
        target_rels_emb = self.rel_emb(target_rels).unsqueeze(2)

        out_atts = self.softmax(self.nei_rels_dropout(torch.matmul(out_nei_rels_emb, target_rels_emb).squeeze(2)))
        in_atts = self.softmax(self.nei_rels_dropout(torch.matmul(in_nei_rels_emb, target_rels_emb).squeeze(2)))
        out_sem_feats = torch.matmul(out_atts.unsqueeze(1), out_nei_rels_emb).squeeze(1)
        in_sem_feats = torch.matmul(in_atts.unsqueeze(1), in_nei_rels_emb).squeeze(1)
        
        if self.params.init_nei_rels == 'both':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(torch.cat([out_sem_feats, in_sem_feats], dim=1)))
        elif self.params.init_nei_rels == 'out':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(out_sem_feats))
        elif self.params.init_nei_rels == 'in':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(in_sem_feats))

        g.ndata['init'] = torch.cat([g.ndata['feat'], ent_sem_feats], dim=1)  # [B, self.inp_dim]

    def comp_ht_emb(self, head_embs, tail_embs):
        if self.params.comp_ht == 'mult':
            ht_embs = head_embs * tail_embs
        elif self.params.comp_ht == 'mlp':
            ht_embs = self.fc_comp(torch.cat([head_embs, tail_embs], dim=1))
        elif self.params.comp_ht == 'sum':
            ht_embs = head_embs + tail_embs
        else:
            raise KeyError(f'composition operator of head and relation embedding {self.comp_ht} not recognized.')

        return ht_embs

    def comp_hrt_emb(self, head_embs, tail_embs, rel_embs):
        rel_embs = rel_embs.repeat(1, self.params.num_gcn_layers)
        if self.params.comp_hrt == 'TransE':
            hrt_embs = head_embs + rel_embs - tail_embs
        elif self.params.comp_hrt == 'DistMult':
            hrt_embs = head_embs * rel_embs * tail_embs
        else: raise KeyError(f'composition operator of (h, r, t) embedding {self.comp_hrt} not recognized.')
        
        return hrt_embs

    def nei_rel_path(self, g, rel_labels, r_emb_out):
        """ Neighboring relational path module """
        # Only consider in-degree relations first.
        nei_rels = g.ndata['in_nei_rels']
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        heads_rels = nei_rels[head_ids]
        tails_rels = nei_rels[tail_ids]

        # Extract neighboring relational paths
        batch_paths = []
        for (head_rels, r_t, tail_rels) in zip(heads_rels, rel_labels, tails_rels):
            paths = []
            for h_r in head_rels:
                for t_r in tail_rels:
                    path = [h_r, r_t, t_r]
                    paths.append(path)
            batch_paths.append(paths)       # [B, n_paths, 3] , n_paths = n_head_rels * n_tail_rels
        
        batch_paths = torch.LongTensor(batch_paths).to(rel_labels.device)# [B, n_paths, 3], n_paths = n_head_rels * n_tail_rels
        batch_size = batch_paths.shape[0]
        batch_paths = batch_paths.view(batch_size * len(paths), -1) # [B * n_paths, 3]

        batch_paths_embs = F.embedding(batch_paths, r_emb_out, padding_idx=-1) # [B * n_paths, 3, inp_dim]

        # Input RNN 
        _, last_state = self.rnn(batch_paths_embs) # last_state: [1, B * n_paths, inp_dim]
        last_state = last_state.squeeze(0) # squeeze the dim 0 
        last_state = last_state.view(batch_size, len(paths), self.params.emb_dim) # [B, n_paths, inp_dim]
        # Aggregate paths by attention
        if self.params.path_agg == 'mean':
            output = torch.mean(last_state, 1) # [B, inp_dim]
        
        if self.params.path_agg == 'att':
            r_label_embs = F.embedding(rel_labels, r_emb_out, padding_idx=-1) .unsqueeze(2) # [B, inp_dim, 1]
            atts = torch.matmul(last_state, r_label_embs).squeeze(2) # [B, n_paths]
            atts = F.softmax(atts, dim=1).unsqueeze(1) # [B, 1, n_paths]
            output = torch.matmul(atts, last_state).squeeze(1) # [B, 1, n_paths] * [B, n_paths, inp_dim] -> [B, 1, inp_dim] -> [B, inp_dim]
        else:
            raise ValueError('unknown path_agg')
        
        return output # [B, inp_dim]

    def get_logits(self, s_G, s_g_pos, s_g_cor): 
        ret = self.disc(s_G, s_g_pos, s_g_cor)
        return ret
    
    def forward(self, data, is_return_emb=False, cor_graph=False):
        # Initialize the embedding of entities
        g, rel_labels = data
        
        # Neighboring Relational Feature Module
        ## Initialize the embedding of nodes by neighbor relations
        if self.params.init_nei_rels == 'no':
            g.ndata['init'] = g.ndata['feat'].clone()
        else:
            self.init_ent_emb_matrix(g)
        
        # Corrupt the node feature
        if cor_graph:
            g.ndata['init'] = g.ndata['init'][torch.randperm(g.ndata['feat'].shape[0])]  
        
        # r: Embedding of relation
        r = self.rel_emb.weight.clone()
        
        # Input graph into GNN to get embeddings.
        g.ndata['h'], r_emb_out = self.gnn(g, r)
        
        # GRU layer for nodes
        graph_sizes = g.batch_num_nodes
        out_dim = self.params.num_gcn_layers * self.params.emb_dim
        g.ndata['repr'] = F.relu(self.batch_gru(g.ndata['repr'].view(-1, out_dim), graph_sizes()))
        node_hiddens = F.relu(self.W_o(g.ndata['repr']))  # num_nodes x hidden 
        g.ndata['repr'] = self.dropout(node_hiddens)  # num_nodes x hidden
        g_out = mean_nodes(g, 'repr').view(-1, out_dim)

        # Get embedding of target nodes (i.e. head and tail nodes)
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        
        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out,
                               head_embs.view(-1, out_dim),
                               tail_embs.view(-1, out_dim),
                               F.embedding(rel_labels, r_emb_out, padding_idx=-1)], dim=1)
        else:
            g_rep = torch.cat([g_out, self.rel_emb(rel_labels)], dim=1)
        
        # Represent subgraph by composing (h,r,t) in some way. (Not use in paper)
        if self.params.comp_hrt:
            edge_embs = self.comp_hrt_emb(head_embs.view(-1, out_dim), tail_embs.view(-1, out_dim), F.embedding(rel_labels, r_emb_out, padding_idx=-1))
            g_rep = torch.cat([g_out, edge_embs], dim=1)

        # Model neighboring relational paths 
        if self.params.nei_rel_path:
            # Model neighboring relational path
            g_p = self.nei_rel_path(g, rel_labels, r_emb_out)
            g_rep = torch.cat([g_rep, g_p], dim=1)
            s_g = torch.cat([g_out, g_p], dim=1)
        else:
            s_g = g_out
        output = self.fc_layer(g_rep)

        self.r_emb_out = r_emb_out
        
        if not is_return_emb:
            return output
        else:
            # Get the subgraph-level embedding
            s_G = s_g.mean(0)
            return output, s_G, s_g

class RGCN(nn.Module):

    def __init__(self, params):
        super(RGCN, self).__init__()

        # self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rel
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.has_attn = params.has_attn
        
        self.is_comp = params.is_comp

        # self.device = params.device

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        # initialize aggregators for input and hidden layers
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # initialize basis weights for input and hidden layers
        # self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        # self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def create_features(self):
        # features = torch.arange(self.inp_dim).to(device=self.device)
        features = torch.arange(self.inp_dim)
        return features

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNBasisLayer(self.inp_dim,
                         self.emb_dim,
                         # self.input_basis_weights,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True,
                         has_attn=self.has_attn,
                         is_comp=self.is_comp)

    def build_hidden_layer(self, idx):
        return RGCNBasisLayer(self.emb_dim,
                         self.emb_dim,
                         # self.basis_weights,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         has_attn=self.has_attn,
                         is_comp=self.is_comp)

    def forward(self, g, r):
        for layer in self.layers:
            r = layer(g, r, self.attn_rel_emb)
        return g.ndata.pop('h'), r

class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x

class RGCNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, bias=None, activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        self.aggregator = aggregator

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, rel_emb, attn_rel_emb=None):
        raise NotImplementedError

class RGCNBasisLayer(RGCNLayer):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False, is_comp=''):
        super(
            RGCNBasisLayer,
            self).__init__(
            inp_dim,
            out_dim,
            aggregator,
            bias,
            activation,
            dropout=dropout,
            edge_dropout=edge_dropout,
            is_input_layer=is_input_layer)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn
        self.is_comp = is_comp

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        # self.weight = basis_weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        # Project relation embedding to current node input embedidng
        self.w_rel = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))
        if self.has_attn:
            self.A = nn.Linear(2 * self.inp_dim + 2 * self.attn_rel_emb_dim, inp_dim)
            self.B = nn.Linear(inp_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_rel, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, attn_rel_emb=None):

        # generate all weights from bases
        weight = self.weight.view(self.num_bases,
                                  self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(
            self.num_rels, self.inp_dim, self.out_dim)

        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))
        
        # input_ = 'feat' if self.is_input_layer else 'h'
        input_ = 'init' if self.is_input_layer else 'h'

        def comp(h, edge_data):
            """ Refer to CompGCN """
            if self.is_comp == 'mult':
                return h * edge_data
            elif self.is_comp == 'sub':
                return h - edge_data
            else:
                raise KeyError(f'composition operator {self.comp} not recognized.')

        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])
            
            # Similar to CompGCN to interact nodes and relations
            if self.is_comp:
                edge_data = comp(edges.src[input_], F.embedding(edges.data['type'], self.rel_emb, padding_idx=-1))
            else:
                edge_data = edges.src[input_]

            msg = edges.data['w'] * torch.bmm(edge_data.unsqueeze(1), w).squeeze(1)

            curr_emb = torch.mm(edges.dst[input_], self.self_loop_weight)  # (B, F)

            if self.has_attn:
                e = torch.cat([edges.src[input_], edges.dst[input_], attn_rel_emb(edges.data['type']), attn_rel_emb(edges.data['label'])], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                a = torch.ones((len(edges), 1)).to(device=w.device)

            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        g.update_all(msg_func, self.aggregator, None)

    def forward(self, g, rel_emb, attn_rel_emb=None):
        self.rel_emb = rel_emb
        self.propagate(g, attn_rel_emb)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr

        if self.is_input_layer:
            g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)

        rel_emb_out = torch.matmul(self.rel_emb, self.w_rel)
        rel_emb_out[-1, :].zero_()       # padding embedding as 0
        return rel_emb_out

class Aggregator(nn.Module):
    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    @abc.abstractmethod
    def update_embedding(curr_emb, nei_msg):
        raise NotImplementedError

class SumAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__(emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb

class MLPAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__(emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))

        return new_emb

class GRUAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__(emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)

        return new_emb

class Discriminator(nn.Module):
    r""" Discriminator module for calculating MI"""
    
    def __init__(self, n_e, n_g):
        """
        param: n_e: dimension of edge embedding
        param: n_g: dimension of graph embedding
        """
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_e, n_g, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0) # [1, F]
        c_x = c_x.expand_as(h_pl)   #[B, F]

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1) # [B];  self.f_k(h_pl, c_x): [B, 1]
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1) # [B]

        # print('Discriminator time:', time.time() - ts)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2))

        return logits

class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
      #  print(hidden.shape)
        message = F.relu(node + self.bias)
        MAX_node_len = max(a_scope)
        # padding
        message_lst = []
        hidden_lst = []
        a_start = 0
        for i in a_scope:
            i = int(i)
            if i == 0:
                assert 0
            cur_message = message.narrow(0, a_start, i)
            cur_hidden = hidden.narrow(0, a_start, i)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            a_start += i
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_node_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        kk = 0
        for a_size in a_scope:
            a_size = int(a_size)
            cur_message_unpadding.append(cur_message[kk, :a_size].view(-1, 2*self.hidden_size))
            kk += 1
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
    
        return cur_message_unpadding 