import random
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp

class SequentialData(Dataset):
    def __init__(self, source, item_num, user_num=1, max_len=50, train_flag=0, neg_num=10, path=None, split=False):
        self.data = []
        self.n_user = user_num
        self.n_item = item_num
        self.neg_num = neg_num
        self.path = path
        self.trainUser, self.trainItem = [], []
        self.max_len = max_len
        self.split = split
        self.read(source)
        self.train_flag = train_flag

    def read(self, source):
        self.users = set()
        self.items = set()
        self.graph = dict()
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                time_stamp = int(conts[2])
                self.users.add(user_id)
                self.items.add(item_id)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((item_id, time_stamp))
        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[1])
            self.data.append([user_id+self.n_item, [x[0] for x in value]])
            
        self.users = list(self.users)
        self.items = list(self.items)
        
        if self.split:
            self.split_onebyone()
        else:
            self.length = len(self.users)
            self.data_aug = self.data


    def split_onebyone(self):
        self.data_aug = {}
        idx = 0
        for seq in self.data:
            user_id, item_list = seq[0], seq[1]
            if len(item_list) > self.max_len:
                for k in range(self.max_len, len(item_list)-1):
        
                    s = item_list[k-self.max_len:k]
                    self.data_aug[idx] = [user_id, s]
                    idx += 1
            else:
                self.data_aug[idx] = [user_id, item_list]
                idx += 1
                
        self.length = len(self.data_aug)

    def get_edge(self):
        edge = [[], []]
        for user_id, seq_list in self.data:
            if len(seq_list) > self.max_len:
                seq_list = seq_list[-self.max_len:]
            for item in seq_list:
                edge[0].append(item)
                edge[1].append(user_id)
                edge[0].append(user_id)
                edge[1].append(item)
        return torch.tensor(edge).long()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_train_user_item(self):
        for uid, items in self.data:
            self.trainUser.extend([uid] * len(items))
            self.trainItem.extend(items)

    def get_train_u_u_graph(self):
        user_adj_mat = sp.load_npz(self.path + '/u_u_graph.npz')
        user_adj_mat = user_adj_mat.tocoo()
        row = user_adj_mat.row
        col = user_adj_mat.col
        data = user_adj_mat.data
        u_u_graph = [row, col]
        return torch.tensor(u_u_graph).long(), torch.tensor(data).float(), user_adj_mat
    
    def get_train_u_i_graph(self):
        user_item_adj_mat = sp.load_npz(self.path + '/u_i_graph.npz')
        user_item_adj_mat = user_item_adj_mat.tocoo()
        row = user_item_adj_mat.row
        col = user_item_adj_mat.col
        data = user_item_adj_mat.data
        u_u_graph = [row, col]
        return torch.tensor(u_u_graph).long(), torch.tensor(data).float(), user_item_adj_mat
    
    def get_u_i_graph(self):
        return self.ui_adj

    def get_train_i_i_graph(self):
        item_adj_mat = sp.load_npz(self.path + '/i_i_graph.npz')
        item_adj_mat = item_adj_mat.tocoo()
        row = item_adj_mat.row
        col = item_adj_mat.col
        data = item_adj_mat.data
        i_i_graph = [row, col]
        return torch.tensor(i_i_graph).long(), torch.tensor(data).float(), item_adj_mat

    def __getitem__(self, index):
        user_id, item_list = self.data_aug[index][0], self.data_aug[index][1]

        if self.train_flag == 0:
            k = random.choice(range(4, len(item_list)))
            item_id_list = item_list[k]
        else:
            k = int(len(item_list) * 0.8)
            item_id_list = item_list[k:]

        if k >= self.max_len:
            hist_item_list = item_list[k - self.max_len: k]
            hist_mask_list = [1.0] * self.max_len
        else:
            hist_item_list = item_list[:k] + [0] * (self.max_len - k)
            hist_mask_list = [1.0] * k + [0.0] * (self.max_len - k)

        if self.train_flag != 0:
            item_id_list = item_id_list[:100] + [0] * max(0, 100 - len(item_id_list))

        return [torch.tensor(user_id), torch.tensor(item_id_list),
                torch.tensor(hist_item_list), torch.tensor(hist_mask_list)]

    def __len__(self):
        return self.length
    


class DiffusionData(Dataset):
    def __init__(self, matrix, max, min, perm, primal_matrix=None):
        self.matrix = matrix
        self.max = max
        self.min = min
        self.row_num, self.col_num = matrix.shape
        self.perm = perm
        self.train = True
        self.primal_matrix = primal_matrix
        self.edge_set = self.get_edge_set()
    
    def update_graph(self, graph):
        self.primal_matrix = graph
    
    def get_row(self, i):
        row, col = self.matrix._indices()
        value = self.matrix._values()
        row_index = i
        index = torch.where(row==row_index)
        col_index = col[index].reshape(-1)
        v = value[index].reshape(-1)
        zero = torch.zeros((self.col_num))
        zero[col_index] = v
        col_index = zero
        return col_index
    
    def __getitem__(self, index):
        return self.perm[index], self.get_row(index), self.max[index], self.min[index]

    def __len__(self):
        return self.row_num
        
    def get_edge_set(self):
        edge_set = set()
        indices = list(self.primal_matrix._indices().T.cpu().numpy())
        for edge in indices:
            edge_set.add(tuple(edge))
        return edge_set