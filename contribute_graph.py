import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import scipy.sparse as sp
import torch
from tqdm import trange
from tqdm import tqdm

dataset = 'gowalla'
neighbor_num = 50
path = f'datasets_large/{dataset}'
train_file = path + '/train.txt'
item_map_file = path + '/' + dataset + '_item_map.txt'
user_map_file = path + '/' + dataset + '_user_map.txt'
trainUniqueUsers, trainItem, trainUser = [], [], []
testUniqueUsers, testItem, testUser = [], [], []
max_len = 20

with open(item_map_file, 'r') as f:
    lines = f.readlines()
    last = lines[-1].strip()
    item_num = int(last.split(',')[-1]) + 1

with open(user_map_file, 'r') as f:
    lines = f.readlines()
    last = lines[-1]
    user_num = int(last.split(',')[-1]) + 1

print("Load data!")
u_i = sp.dok_matrix((user_num, item_num), dtype=np.int32)

interaction = dict()
all_train_data = []
with open(train_file, 'r') as f:
    for line in f:
        conts = line.strip().split(',')
        user_id = int(conts[0])
        item_id = int(conts[1])
        time_stamp = int(conts[2])
        if user_id not in interaction:
            interaction[user_id] = []
        interaction[user_id].append((item_id, time_stamp))
for user_id, value in tqdm(interaction.items()):
    value.sort(key=lambda x: x[1])
    all_train_data.append([user_id, [x[0] for x in value]])
    for v in value[-50:]:
        u_i[user_id, v[0]] = 1

print("Load data success!")

u_i_mat = u_i.tocsr()
sp.save_npz(path + f'/u_i_graph.npz', u_i_mat)

# construct user-user graph
print("Compute user-user matrix!")
u_u = u_i * u_i.transpose()

print("Construct user-user graph!")
u_u_mat = sp.dok_matrix((user_num, user_num), dtype=np.float32)
edge_user = [[], []]
for j in trange(user_num):
    user_neighbor = torch.tensor(u_u.getrow(j).toarray()).view(-1)
    user_neighbor = user_neighbor.topk(neighbor_num)
    score = user_neighbor[0].long().detach().numpy()
    neighbor = user_neighbor[1].long().detach().numpy()
    if score[0] == 0:
        continue
    for ne, s in zip(neighbor, score):
        u_u_mat[j, ne] = s
        u_u_mat[ne, j] = s

print("Construct user-user graph success!")
u_u_mat = u_u_mat.tocsr()
sp.save_npz(path + f'/u_u_graph.npz', u_u_mat)

# construct item-item graph
print("Compute item-item matrix!")
i_i = u_i.transpose() * u_i

print("Construct item-item graph!")
i_i_mat = sp.dok_matrix((item_num, item_num), dtype=np.float32)

for j in trange(item_num):
    item_neighbor = torch.tensor(i_i.getrow(j).toarray()).view(-1)
    item_neighbor = item_neighbor.topk(neighbor_num)
    score = item_neighbor[0].long().numpy()
    neighbor = item_neighbor[1].long().numpy()
    if score[0] == 0:
        continue
    for ne, s in zip(neighbor, score):
        i_i_mat[j, ne] = s
        i_i_mat[ne, j] = s

print("Construct item-item graph success!")
i_i_mat = i_i_mat.tocsr()
sp.save_npz(path + f'/i_i_graph.npz', i_i_mat)

print('end')
