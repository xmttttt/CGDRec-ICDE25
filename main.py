import time
import argparse
import datetime
import torch
from model import *
from datasets import *
from utils import *
import logging
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
init_seed(2020)

t = time.localtime()
filename = '-'.join([str(t.tm_mday), str(t.tm_hour), str(t.tm_min), '.log'])
fmt = "%(asctime)s : %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    filename=filename,
    filemode="w",
    format=fmt
)

def generate_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clothing', help='clothing/toys/gowalla')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_d', type=int, default=2048)
    parser.add_argument('--num_interest', type=int, default=4)
    parser.add_argument('--r', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--lr_step', type=float, default=10)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--mask_step', type=int, default=50)
    parser.add_argument('--max_len', type=int, default=20)

    parser.add_argument('--train_steps', type=int, default=5)
    parser.add_argument('--noise_steps', type=int, default=1)
    parser.add_argument('--denoise_steps', type=int, default=1)
    parser.add_argument('--rebuild_k', type=int, default=10)
    parser.add_argument('--noise_ratio', type=float, default=0.1)
    
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--sample_epoch', type=float, default=2)
    parser.add_argument('--time_size', type=int, default=64)
    parser.add_argument('--middle_size', type=int, default=512)
    parser.add_argument('--split_dataset', type=bool, default=True)
    
    parser.add_argument('--atten_mode', type=str, default='new')
    parser.add_argument('--test_dropout', type=float, default=0.5)
    parser.add_argument('--noise_test', type=float, default=0.2)

    # trade-off parameter

    parser.add_argument('--lambda_bpr', type=float, default=1)
    parser.add_argument('--lambda_cl', type=float, default=1)
    parser.add_argument('--lambda_diffcl', type=float, default=0)

    # number of layers within two levels
    parser.add_argument('--co_layers', type=int, default=1)
    parser.add_argument('--cl_layers', type=int, default=1)

    # autoencoder parameter
    parser.add_argument('--path_prob', type=int, default=0.5)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='split the portion')

    # temperature parameter
    parser.add_argument('--cl_tau', type=float, default=1, help='temperature parameter of co-info')
    parser.add_argument('--recon_tau', type=float, default=1, help='split the portion')
    
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--scheduler', type=bool, default=True)
    parser.add_argument('--graph_iter', type=bool, default=True)
    
    opt = parser.parse_args()
    return opt

def fix_opt(opt):
    if opt.dataset == 'clothing':
        opt.lr = 1e-3
        # opt.atten_mode = 'p'
        opt.r = 0.6
        opt.epoch = 50
        opt.scheduler = True
        opt.sample_epoch = 5
        opt.train_ratio = 0.05
        opt.train_steps = 5
        opt.noise_steps = 2
        opt.denoise_steps = 3
        opt.split_dataset = False
        opt.graph_iter = True
        opt.lambda_diffcl = 1e-5

    elif opt.dataset == 'toys':
        opt.lr = 1e-2
        opt.epoch = 30
        # opt.atten_mode = 'p'
        opt.r = 0.7
        opt.sample_epoch = 8
        opt.scheduler = True
        opt.train_ratio = 0.1 #0.1
        opt.middle_size = 128
        opt.train_steps = 6 #6
        opt.noise_steps = 1
        opt.denoise_steps = 3
        opt.split_dataset = False
        opt.graph_iter = True
        opt.lambda_diffcl = 1e-5
        
        # toys 压力测试

    elif opt.dataset == 'gowalla':
        opt.lr = 1e-2
        # opt.atten_mode = 'sp'
        opt.epoch = 50
        opt.batch_size_d = 64
        opt.r = 0.8
        opt.middle_size = 128
        opt.sample_epoch = 8
        opt.train_ratio = 0.02
        opt.train_steps = 7
        opt.denoise_steps = 4
        opt.noise_steps = 4
        # opt.train_ratio = 0.01
        opt.split_dataset = False
        opt.scheduler = True
        opt.graph_iter = True
        opt.lambda_diffcl = 1e-6
        
    # 复现完毕
    elif opt.dataset == 'yelp':
        opt.lr = 1e-3
        # opt.atten_mode = 'p'
        opt.r = 0.6
        opt.epoch = 50
        opt.sample_epoch = 5
        opt.train_ratio = 0.05 #0.02
        opt.train_steps = 5
        opt.noise_steps = 2
        opt.denoise_steps = 3
        opt.split_dataset = False
        opt.scheduler = False
        opt.graph_iter = True
        opt.lambda_diffcl = 1e-5

    # ？？？？
    elif opt.dataset == 'beauty':
        opt.lr = 1e-3
        # opt.atten_mode = 'p'
        opt.r = 0.8
        opt.epoch = 30
        opt.sample_epoch = 5
        opt.train_ratio = 0.05
        opt.train_steps = 5
        opt.noise_steps = 2
        opt.denoise_steps = 3
        opt.split_dataset = False
        opt.scheduler = False
        opt.graph_iter = True
        opt.lambda_diffcl = 1e-5
        
    if opt.denoise_steps > opt.train_steps:
        opt.denoise_steps = opt.train_steps
        
    opt.denoise_steps = opt.noise_steps + 1
    
    return opt

def evaluate_full(scores, pos_items, topN):
    score = scores[1][:, :, :topN].detach().cpu().numpy()
    value = scores[0][:, :, :topN].detach().cpu().numpy()

    pos_num = torch.sum(pos_items.gt(0), -1).numpy()
    pos_items = pos_items.numpy()

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    
    for i, iid_list in enumerate(pos_items):
        dcg = 0.0
        recall = 0
        item_list = score[i].reshape(-1)
        value_list = value[i].reshape(-1)
        item_list = list(zip(item_list, value_list))
        item_list.sort(key=lambda x: x[1], reverse=True)

        item_rank_set = set()
        item_rank_list = list()

        for j in range(len(item_list)):
            if item_list[j][0] not in item_rank_set and item_list[j][0] != 0:
                item_rank_set.add(item_list[j][0])
                item_rank_list.append(item_list[j][0])
                if len(item_rank_set) >= topN:
                    break

        for no, iid in enumerate(item_rank_list):
            if iid == 0:
                break
            if iid in iid_list:
                recall += 1
                dcg += 1.0 / math.log(no + 2, 2)

        idcg = 0.0

        # scale the NDCG score for better comparison
        expect_num = min(3, len(item_rank_list))

        for no in range(expect_num):
            idcg += 1.0 / math.log(no + 2, 2)

        total_recall += recall * 1.0 / pos_num[i]
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1

    total = pos_items.shape[0]

    recall = total_recall * 1.0 / total
    ndcg = total_ndcg * 1.0 / total
    hitrate = total_hitrate * 1.0 / total

    return recall, ndcg, hitrate


@torch.no_grad()
def evaluate_score(model, test_data, _type='test', embedding=None):
    result = [[], [], []]
    result20 = [[], [], []]

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=8, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    for data in tqdm(test_loader):
        user_id, pos_item, hist_item_list, hist_mask_list = data
        user_id, pos_item, hist_item_list, hist_mask_list = user_id.to(device), pos_item, \
                                                            hist_item_list.to(device), hist_mask_list.to(device)

        if embedding == None:
            interest_emb = model.sequence_modeling(hist_item_list, model.item_embedding.weight, hist_mask_list)
            item_emb = model.item_embedding.weight.data
        else:
            interest_emb = model.sequence_modeling(hist_item_list, embedding, hist_mask_list)
            item_emb = embedding.data
        
        interest_emb = interest_emb.data
        scores = torch.matmul(interest_emb.half(), item_emb.transpose(-2, -1).half())
        
        scores = scores.topk(50)
 
        recall, ndcg, hit = evaluate_full(scores, pos_item, 50)
        result[0].append(recall)
        result[1].append(ndcg)
        result[2].append(hit)

        recall2, ndcg2, hit2 = evaluate_full(scores, pos_item, 20)
        result20[0].append(recall2)
        result20[1].append(ndcg2)
        result20[2].append(hit2)

    return [np.mean(result[0]), np.mean(result[1]), np.mean(result[2])], \
           [np.mean(result20[0]), np.mean(result20[1]), np.mean(result20[2])]


def train_test(opt, model, train_data, valid_data, test_data, sparse_matrix, epoch, i_i_degree, u_u_degree, diffusion_loader_i, diffusion_loader_u, pos_item, pos_user):
    logging.info('start training: ' + str(datetime.datetime.now()))
    print('start training: ', datetime.datetime.now())
    model.train()

    batch_cl_loss = 0.0
    batch_gcl_loss = 0.0
    total_cl_loss = 0.0
    total_gcl_loss = 0.0

    u_u_matrixs, i_i_matrixs, u_i_matrixs = sparse_matrix

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=8, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)

    u_u_norm, u_u_dok, u_u_one, u_u_diff, u_u = u_u_matrixs
    i_i_norm, i_i_dok, i_i_one, i_i_diff, i_i = i_i_matrixs
    
    diffusion_training(model, diffusion_loader_i)
    diffusion_training(model, diffusion_loader_u, item=False)
    
    i_i_aug = diffusion_inference(model, diffusion_loader_i, i_i_one, rebuild_k=opt.rebuild_k, noise_ratio=opt.noise_ratio, degree=i_i_degree)
    u_u_aug = diffusion_inference(model, diffusion_loader_u, u_u_one, rebuild_k=opt.rebuild_k, noise_ratio=opt.noise_ratio, item=False, degree=u_u_degree)

    if opt.graph_iter:
        diffusion_loader_i.dataset.update_graph(i_i_aug)
        diffusion_loader_u.dataset.update_graph(u_u_aug)
        

    for i, data in enumerate(tqdm(train_loader)):

        user_id, target_item, item_list, mask_list = data
        user_id = user_id.to(device)
        neg_sample = negsamp_vectorized_bsearch_preverif(target_item.numpy(), opt.item_num, 10 * model.batch_size)
        neg_sample = torch.Tensor(neg_sample).long().to(device)
        
        target_item = target_item.to(device)
        item_list = item_list.to(device)
        mask_list = mask_list.to(device)
        
        neg_item = negsamp_vectorized_bsearch_preverif(pos_item, opt.item_num, 10 * len(pos_item))
        neg_item = torch.Tensor(neg_item).long().to(device)
        
        neg_user = negsamp_vectorized_bsearch_preverif(pos_user, opt.user_num, 10 * len(pos_user))
        neg_user = torch.Tensor(neg_user).long().to(device)

        # co-occurrence level
        item_embedding = model.co_encoder(i_i_aug, model.item_embedding)
        user_embedding = model.co_encoder(u_u_aug, model.user_embedding)

        # local-cl loss
        interest, local_cl_loss, user_emb, item_emb = model(item_list, user_id, item_embedding[0], user_embedding[0], u_i_matrixs)
        
        final_item_emb = item_emb
        target_item_emb = final_item_emb[target_item]
        neg_item_emb = final_item_emb[neg_sample]
        
        # recommend loss
        recommend_loss = model.get_recommend_loss(interest, target_item_emb, neg_item_emb)
        cl_loss = recommend_loss + opt.lambda_cl * local_cl_loss
        
        # new cl loss
        src_item_embedding = model.co_encoder(i_i_norm, model.item_embedding)
        src_user_embedding = model.co_encoder(u_u_norm, model.user_embedding)
        
        ii_gcl_loss = model.get_diff_cl_loss(item_embedding[0], src_item_embedding[0], pos_item, neg_item)
        uu_gcl_loss = model.get_diff_cl_loss(user_embedding[0], src_user_embedding[0], pos_user, neg_user)

        # co-info loss
        co_info_loss = model.get_co_info_loss(user_emb, item_emb, target_item, neg_sample)    
        gcl_loss = opt.lambda_bpr * co_info_loss + opt.lambda_diffcl * (ii_gcl_loss + uu_gcl_loss)

        loss = cl_loss + gcl_loss
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        batch_cl_loss += cl_loss.item()
        batch_gcl_loss += gcl_loss.item()

        if (i+1) % int(len(train_loader) / 10) == 0:
            total_cl_loss += batch_cl_loss
            batch_cl_loss = 0

        if (i+1) % int(len(train_loader) / 10) == 0:
            total_gcl_loss += batch_gcl_loss
            batch_gcl_loss = 0


    logging.info('\tRecLoss:\t%.3f %3.f' % (total_cl_loss / len(train_loader), total_cl_loss))
    logging.info('\tReconLoss:\t%.3f %3.f' % (total_gcl_loss / len(train_loader), total_gcl_loss))

    result_test_50, result_test_20 = evaluate_score(model, test_data, _type='test', embedding=final_item_emb)
    result_test_20 = [i * 100 for i in result_test_20]
    result_test_50 = [i * 100 for i in result_test_50]
    result_valid_20, result_valid_50 = result_test_20, result_test_50

    return result_valid_20, result_valid_50, result_test_20, result_test_50

def sim(z1, z2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    return torch.matmul(z1, z2.transpose(-1, -2))

def main(opt):
    
    path = f'datasets/{opt.dataset}'
    item_map_file = path + '/' + opt.dataset + '_item_map.txt'
    user_map_file = path + '/' + opt.dataset + '_user_map.txt'

    with open(item_map_file, 'r') as f:
        lines = f.readlines()
        last = lines[-1].strip()
        opt.item_num = int(last.split(',')[-1]) + 1

    with open(user_map_file, 'r') as f:
        lines = f.readlines()
        last = lines[-1]
        opt.user_num = int(last.split(',')[-1]) + 1

    opt = fix_opt(opt)

    logging.info(str("torch_device_count: " + str(torch.cuda.device_count())))

    train_data = SequentialData(f"datasets/{opt.dataset}/train.txt", opt.item_num, user_num=opt.user_num, max_len=opt.max_len,
                                train_flag=0, path="datasets/{}".format(opt.dataset), split=opt.split_dataset)
    valid_data = SequentialData(f"datasets/{opt.dataset}/valid.txt", opt.item_num, max_len=opt.max_len, train_flag=1, split=False)
    test_data = SequentialData(f"datasets/{opt.dataset}/test.txt", opt.item_num, max_len=opt.max_len, train_flag=2, split=False)

    u_u = train_data.get_train_u_u_graph()[-1]
    i_i = train_data.get_train_i_i_graph()[-1]
    u_i = train_data.get_train_u_i_graph()[-1]
    
    # user-item graph
    u_i_edge = train_data.get_edge()
    bi_ui, bi_ui_dok = makeBiAdj(opt, u_i)
    i_i_dok = i_i.todok()
    i_i_norm = makeTorchAdj(i_i)
    i_i_diff = from_scipy((i_i + sp.eye(i_i.shape[0])).tocoo())
    i_i_diff = i_i_diff.coalesce() * 1.0
    i_i_degree = torch.sparse.sum(i_i_diff, dim=-1).to_dense()
    i_i_degree_inv = i_i_degree.pow(-1)
    i_i_diff = (i_i_degree_inv * i_i_diff).cuda()
    
    
    # user-user graph
    u_u_dok = u_u.todok()
    u_u_norm = makeTorchAdj(u_u)
    u_u_one = to_all_one(u_u_norm).float()
    u_u_diff = from_scipy((u_u + sp.eye(u_u.shape[0])).tocoo())
    u_u_diff = u_u_diff.coalesce() * 1.0    
    u_u_degree = torch.sparse.sum(u_u_diff, dim=-1).to_dense()
    u_u_degree_inv = u_u_degree.pow(-1)
    u_u_diff = (u_u_degree_inv * u_u_diff).cuda()
    

    i_i_one = to_all_one(i_i_norm).float()

    sparse_matrix = [[u_u_norm, u_u_dok, u_u_one, u_u_diff, u_u], [i_i_norm, i_i_dok, i_i_one, i_i_diff, i_i], [u_i_edge.to(device), bi_ui, bi_ui_dok]]
    

    logging.info(opt)
    model = CGDRec(opt.num_interest, opt.hidden_size, opt.batch_size, opt=opt).to(device)
    
    best_result20 = {'R':[0,0], 'N':[0,0], 'H':[0,0]}
    best_result50 = {'R':[0,0], 'N':[0,0], 'H':[0,0]}
    
    for epoch in range(opt.epoch):
    
        logging.info('-------------------------------------------------------')
        logging.info(str('epoch: ' + str(epoch)))
        
        if epoch % opt.sample_epoch == 0 :
            
            if epoch > 0:
                i_i_diff = diffusion_loader_i.dataset.primal_matrix
                u_u_diff = diffusion_loader_u.dataset.primal_matrix
                
            logging.info(epoch)
            logging.info(str(i_i_diff))
            logging.info(str(u_u_diff))
            i_i_samp, imax, imin, iperm = sampling(i_i_diff, opt)
            u_u_samp, umax, umin, uperm = sampling(u_u_diff, opt)
            
            diffusion_loader_i = torch.utils.data.DataLoader(DiffusionData(i_i_samp, imax, imin, iperm, i_i_diff), num_workers=4, batch_size=opt.batch_size_d,
                                                    shuffle=True, pin_memory=True)
            diffusion_loader_u = torch.utils.data.DataLoader(DiffusionData(u_u_samp, umax, umin, uperm, u_u_diff), num_workers=4, batch_size=opt.batch_size_d,
                                                    shuffle=True, pin_memory=True)
        
            
        result20_v, result50_v, result20_t, result50_t = train_test(opt, model, train_data, valid_data, test_data, sparse_matrix, epoch, i_i_degree, u_u_degree, diffusion_loader_i, diffusion_loader_u, iperm, uperm)

        logging.info('test result:')
        logging.info(str(result20_t))
        logging.info(str(result50_t))

        for (k, v), current_v in zip(best_result20.items(), result20_t):
            if v[1] < current_v:
                best_result20[k] = [epoch, current_v]

        for (k, v), current_v in zip(best_result50.items(), result50_t):
            if v[1] < current_v:
                best_result50[k] = [epoch, current_v]

        best_result20_str = best_result20['R'] + best_result20['N'] + best_result20['H']
        best_result50_str = best_result50['R'] + best_result50['N'] + best_result50['H']

        logging.info('best result:')
        logging.info(str(best_result20_str))
        logging.info(str(best_result50_str))

        if opt.scheduler:
            model.scheduler.step()

    logging.info('-------------------------------------------------------')


def sampling(matrix, opt):
    matrix = matrix.cpu()
    row, col = matrix._indices()
    values = matrix._values()
    length = len(values)
    count = matrix.shape[0]
    
    perm, _ = torch.sort(torch.randperm(count)[:int(opt.train_ratio * count)])
    perm = perm.numpy()

    mapping_p2s = {}
    for i, r in enumerate(perm):
        mapping_p2s.update({r:i})

    perm_set = set(list(perm))
    num = len(perm_set)
    c, r, v = [], [], []
    perm_max = np.zeros_like(perm) * 1.0
    perm_min = np.ones_like(perm) * 1.0

    for i in tqdm(range(length)):

        if values[i] != 0 and int(row[i]) in perm_set:
            r_tmp = mapping_p2s[int(row[i])]
            r.append(r_tmp)
            c.append(col[i])
            v.append(values[i])
            
            if perm_max[r_tmp] < values[i]:
                perm_max[r_tmp] = values[i]
            if perm_min[r_tmp] > values[i]:
                perm_min[r_tmp] = values[i]
                
    r = torch.tensor(r).unsqueeze(0)   
    c = torch.tensor(c).unsqueeze(0)
    indices = torch.cat([r, c], dim=0).to(torch.int64)
    v = torch.tensor(v)
    sampling_matrix = torch.sparse.FloatTensor(indices, v, [num, matrix.shape[-1]])
    return sampling_matrix, perm_max, perm_min, perm

if __name__ == '__main__':
    opt = generate_opt()
    main(opt)
