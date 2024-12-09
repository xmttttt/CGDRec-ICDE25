import time
import torch
from model import *
from datasets import *

device = torch.device('cuda')

def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    """
    raw_samp = np.random.randint(low=0, high=n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds

def normalize(mat):
    degree = np.array(mat.sum(axis=-1))
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    dInvSqrtMat = sp.diags(dInvSqrt)
    return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

def makeBiAdj(opt, ui_mat):
    a = sp.csr_matrix((opt.user_num, opt.user_num))
    b = sp.csr_matrix((opt.item_num, opt.item_num))
    mat = sp.vstack([sp.hstack([a, ui_mat]), sp.hstack([ui_mat.transpose(), b])])
    mat = (mat != 0) * 1.0
    mat = (mat + sp.eye(mat.shape[0])) * 1.0
    mat = normalize(mat)
    dok_mat = ui_mat.todok()
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse.FloatTensor(idxs, vals, shape).to(device), dok_mat

def from_scipy(sparse_matrix):
    row = sparse_matrix.row
    col = sparse_matrix.col
    index = torch.from_numpy(np.array([row, col])).long()
    data = torch.from_numpy(sparse_matrix.data).float()
    shape = torch.Size(sparse_matrix.shape)
    return torch.sparse.FloatTensor(index, data, shape)

def makeTorchAdj(mat):
    mat_ = (mat + sp.eye(mat.shape[0])) * 1.0
    mat_ = (mat_ != 0) * 1.0
    mat_n = normalize(mat_)
    mat_n = from_scipy(mat_n)
    return mat_n.to(device).float()

def makeTorchAdj_noself(mat):
    # mat = (mat + sp.eye(mat.shape[0])) * 1.0
    mat = (mat != 0) * 1.0
    mat_n = normalize(mat)
    mat_n = from_scipy(mat_n)
    return mat_n.to(device).float()

def to_all_one(sparse_matrix):
    index = sparse_matrix._indices()
    data = torch.ones_like(sparse_matrix._values(), dtype=torch.float16)
    shape = sparse_matrix.shape
    return torch.sparse.FloatTensor(index, data, shape)

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

