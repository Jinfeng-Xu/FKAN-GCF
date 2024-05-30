
from models.common.kanlayer import NaiveFourierKANLayer as KANLayer
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.abstract_recommender import GeneralRecommender
from models.common.init import xavier_normal_initialization
from models.common.layers import BiGNNLayer, SparseDropout
from models.common.loss import BPRLoss, EmbLoss


class FourierKAN_GCF(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FourierKAN_GCF, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.embedding_size = config['embedding_size']
        #self.hidden_size_list = config['hidden_size_list']
        #self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.n_layers = config['n_layer']
        self.hidden_size_list = [self.embedding_size] * self.n_layers
        self.node_dropout = config['dropout_node']
        self.message_dropout = config['dropout_message']
        self.reg_weight = config['reg_weight']
        self.grid_size = config['grid_size']

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])):
            self.GNNlayers.append(FourierGNNLayer(input_size, output_size, self.grid_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7  # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.
        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        A_hat = self.sparse_dropout(self.norm_adj_matrix) if self.node_dropout != 0 else self.norm_adj_matrix
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [all_embeddings]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(ngcf_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss

        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)  # L2 regularization of embeddings

        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        # get user embedding from storage variable
        u_embeddings = self.user_embedding.weight[user]
        item_embeddings = self.item_embedding.weight

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, item_embeddings.transpose(0, 1))

        return scores

class FourierGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim, grid_size):
        super(FourierGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        # self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = KANLayer(in_dim, out_dim, self.grid_size)

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        # inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return features + x + inter_part2
