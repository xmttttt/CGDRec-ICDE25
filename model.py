import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import APPNP
from utils import *
from tqdm import tqdm
import scipy.sparse as sp
from torch_geometric.nn import GCNConv, SAGEConv, SGConv, APPNP

class CGDRec(nn.Module):
    def __init__(self, num_interest, hidden_size, batch_size, opt=None):
        super(CGDRec, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_interest = num_interest
        self.tau = 0.5
        self.local_k = 2
        self.opt = opt

        self.item_embedding = nn.Embedding(self.opt.item_num, self.hidden_size)
        self.user_embedding = nn.Embedding(self.opt.user_num, self.hidden_size)
        self.pos_embedding = nn.Embedding(200, self.hidden_size)

        self.conv_appnp = APPNP(K=self.local_k, alpha=0)


        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size * self.num_interest)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.W_3 = nn.Linear(self.hidden_size * 4, num_interest)
 
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.diffusion_model = GaussianDiffusion(opt)

        self.denoise_model_i = Denoise(self.opt, self.opt.item_num, self.opt.middle_size, self.opt.time_size)
        self.denoise_model_u = Denoise(self.opt, self.opt.user_num, self.opt.middle_size, self.opt.time_size)
        
        self.diffusion_opt_i = torch.optim.Adam(self.denoise_model_i.parameters(), lr=opt.lr)
        self.diffusion_opt_u = torch.optim.Adam(self.denoise_model_u.parameters(), lr=opt.lr)

        self.encoder_co = Encoder(opt.co_layers)
        self.encoder_cl = Encoder(opt.cl_layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
        self.reset_parameters()

    def forward(self, item_seq, user_id, item_embedding, user_embedding, u_i_matrixs):

        # collaborative-level
        global_item_emb, global_user_emb = self.cl_autoencoder(item_embedding, user_embedding, u_i_matrixs)

        # sequence-modeling
        sequential_interest = self.sequence_modeling(item_seq, item_embedding)

        # graph-modeling
        graph_interest, user_emb, item_emb = self.graph_modeling(item_seq, user_id, global_item_emb, global_user_emb, u_i_matrixs)
        interest = self.opt.r * sequential_interest + (1 - self.opt.r) * graph_interest
        local_cl_loss = self.get_local_cl_loss(sequential_interest, graph_interest)
        return interest, local_cl_loss, user_emb, item_emb
    
    
    def co_encoder(self, graph, embedding):
        embedsLst = [embedding.weight]
        emb, embedsLst = self.encoder_co(graph, embedding.weight)
        return [emb, embedsLst]
    
    def cl_autoencoder(self, item_embedding, user_embedding, u_i_matrixs):
   
        graph_edge, bi_ui, bi_ui_dok = u_i_matrixs
        embeds = torch.concat([user_embedding, item_embedding], axis=0)

        # encode
        embeds, embedsLst = self.encoder_cl(bi_ui, embeds)
        user_embedding, item_embedding = embeds[:self.opt.user_num], embeds[self.opt.user_num:]
       
        return item_embedding, user_embedding
    

    def sequence_modeling(self, item_seq, item_embedding, mask=None):
        mask = item_seq.gt(0)

        item_emb = item_embedding[item_seq]

        pos_emb = self.pos_embedding.weight[:item_seq.shape[1], ].view(-1, item_seq.shape[1], self.hidden_size)
        item_emb_w_pos = item_emb + pos_emb

        item_hidden = torch.tanh(self.W_2(item_emb_w_pos))
        item_attn_w = self.W_3(item_hidden).permute(0, 2, 1)
        attn_mask = mask.unsqueeze(-2).repeat(1, self.num_interest, 1)
        padding = -9e15 * torch.ones_like(attn_mask)

        item_attn_w = torch.where(attn_mask, item_attn_w, padding)
        item_attn_w = torch.softmax(item_attn_w, -1)
        interest_emb = torch.matmul(item_attn_w, item_emb)

        return interest_emb

    def graph_modeling(self, item_seq, user_id, item_embedding, user_embedding, u_i_matrixs):

        graph_edge, bi_ui, bi_ui_dok = u_i_matrixs
        batch_size = item_seq.shape[0]
        seq_len = item_seq.shape[1]
 
        node_id = torch.cat([torch.unique(item_seq), user_id])
        index = torch.randint(high=len(node_id), size=(40,), device=device)
        node_id = node_id[index]
        subset, graph_edge, _, _ = k_hop_subgraph(node_id, self.local_k, graph_edge)

        # gcn
        node_emb = torch.cat([item_embedding, user_embedding], 0)
        node_emb1 = self.conv_appnp(node_emb, graph_edge)
 
        user_emb = node_emb1[user_id]
        item_emb = node_emb1[item_seq]
        mask = item_seq.gt(0)
        user_embed = torch.tanh(self.W_1(user_emb)).view(batch_size, self.num_interest, self.hidden_size)
        alpha = torch.matmul(user_embed, item_emb.transpose(-1, -2))
        attn_mask = mask.view(batch_size, 1, seq_len).repeat(1, self.num_interest, 1)
        padding = -9e15 * torch.ones_like(attn_mask)
        item_attn_w = torch.where(attn_mask, alpha, padding)
        item_attn_w = torch.softmax(item_attn_w, -1)
        interest_emb = torch.matmul(item_attn_w, item_emb)

        return interest_emb, node_emb[user_id], node_emb[:self.opt.item_num]

    def sim(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return torch.matmul(z1, z2.transpose(-1, -2))
    
    def get_local_cl_loss(self, z1: torch.Tensor, z2: torch.Tensor):

        z1 = z1.permute(1, 0, 2)
        z2 = z2.permute(1, 0, 2)

        h1 = self.fc2(F.elu(self.fc1(z1)))
        h2 = self.fc2(F.elu(self.fc1(z2)))

        f = lambda x: torch.exp(x / self.tau)
        refl_score = f(self.sim(h1, h1))
        bet_score = f(self.sim(h1, h2))

        diag_m = torch.eye(refl_score.shape[-1]).view(-1, refl_score.shape[-1], refl_score.shape[-1])
        diag_m = diag_m.to(device)
        diag_m = diag_m.repeat(refl_score.shape[0], 1, 1)

        diag_refl = diag_m * refl_score
        diag_bet = diag_m * bet_score

        cl_loss = -torch.log(diag_bet.sum(-1) /
                                      (refl_score.sum(-1) + bet_score.sum(-1) - diag_refl.sum(-1)))

        cl_loss = torch.mean(cl_loss)

        return cl_loss

    def get_recommend_loss(self, user_emb, target_item_emb, neg_item_emb):
        target_emb = target_item_emb.unsqueeze(-2)
        attn = torch.matmul(target_emb, user_emb.permute(0, 2, 1)).squeeze(1)
        attn = torch.argmax(attn, dim=-1) + torch.arange(attn.shape[0]).to(device) * self.num_interest
        user_emb = user_emb.view(-1, self.hidden_size)
        readout = torch.index_select(input=user_emb, dim=0, index=attn)

        # sample loss
        readout = readout.unsqueeze(-2)
        pos_score = torch.matmul(readout, target_emb.transpose(-2, -1)).squeeze(1)
        neg_emb = neg_item_emb.view(1, -1, self.hidden_size).repeat(readout.shape[0], 1, 1)
        neg_score = torch.matmul(readout, neg_emb.transpose(-2, -1)).squeeze(1)

        score = torch.cat([pos_score, neg_score], -1)
        score = torch.log_softmax(score, -1)[:, 0]
        loss = -torch.mean(score)
        return loss

    def get_co_info_loss(self, user_emb, item_emb, target_item, neg_item):
        target_emb = item_emb[target_item].unsqueeze(-2)
        user_emb = user_emb.unsqueeze(1)
        attn = torch.matmul(target_emb, user_emb.permute(0, 2, 1)).squeeze(1)
        attn = torch.argmax(attn, dim=-1) + torch.arange(attn.shape[0]).to(device) 
        user_emb = user_emb.view(-1, self.hidden_size)
        readout = torch.index_select(input=user_emb, dim=0, index=attn)

        # sample loss
        readout = readout.unsqueeze(-2)
        pos_score = torch.sigmoid(torch.matmul(readout, target_emb.transpose(-2, -1)).squeeze(1)) / self.opt.cl_tau
        neg_emb = item_emb[neg_item[0]].view(1, -1, self.hidden_size).repeat(readout.shape[0], 1, 1)
        neg_score = torch.sigmoid(torch.matmul(readout, neg_emb.transpose(-2, -1)).squeeze(1)) / self.opt.cl_tau
    
        score = torch.cat([pos_score, neg_score], -1)
        score = torch.log_softmax(score, -1)[:, 0]
        loss = -torch.mean(score)
        return loss
    
    def get_diff_cl_loss(self, emb_diff, emb_src, pos, neg):
        
        # bs * hidden_size
        target_emb = emb_diff[pos]
        pos_emb = emb_src[pos]
        neg_emb = emb_src[neg]
        
        # sample loss
        pos_score = self.sim(target_emb, pos_emb) / self.opt.cl_tau
        neg_score = self.sim(target_emb, neg_emb) / self.opt.cl_tau
        
        score = torch.cat([pos_score, neg_score], -1)
        score = torch.log_softmax(score, -1)[:, 0]
        loss = -torch.mean(score)
        return loss
    
    def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
        pos_preds = (anc_embeds * pos_embeds).sum(-1)
        neg_preds = (anc_embeds * neg_embeds).sum(-1)
        return torch.sum(F.softplus(neg_preds - pos_preds))
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            

class GaussianDiffusion(nn.Module):
    def __init__(self, opt):
        super(GaussianDiffusion, self).__init__()
        self.train_steps = opt.train_steps
        self.denoise_steps = opt.denoise_steps
        self.noise_steps = opt.noise_steps

    def noise_calculation(self, max, min):

        beta_min = (torch.pow(min, 2) / (9 + torch.pow(min, 2)))
        beta_max = torch.pow(max, 2) / (1 + torch.pow(max, 2))
        
        self.alphas_cumprod = []
        self.sqrt_alphas_cumprod = []
        self.sqrt_one_minus_alphas_cumprod = []
        self.posterior_variance = []
        self.posterior_log_variance_clipped = []
        self.coef1 = []
        self.coef2 = []
        
        # calcualte beta for each x
        for i in range(len(beta_min)):
            
            betas = torch.linspace(beta_min[i], beta_max[i], self.train_steps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            
            alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

            posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
            )
            posterior_log_variance_clipped = torch.log(torch.cat([posterior_variance[1].unsqueeze(0), posterior_variance[1:]]) + 1e-8)
            
            coef1 = torch.sqrt(alphas)
            coef2 = torch.sqrt(alphas) * betas / (sqrt_one_minus_alphas_cumprod + 1e-8)
            
            self.alphas_cumprod.append(alphas_cumprod.unsqueeze(0))
            self.sqrt_alphas_cumprod.append(sqrt_alphas_cumprod.unsqueeze(0))
            self.sqrt_one_minus_alphas_cumprod.append(sqrt_one_minus_alphas_cumprod.unsqueeze(0))
            self.posterior_variance.append(posterior_variance.unsqueeze(0))
            self.posterior_log_variance_clipped.append(posterior_log_variance_clipped.unsqueeze(0))
            self.coef1.append(coef1.unsqueeze(0))
            self.coef2.append(coef2.unsqueeze(0))

        self.alphas_cumprod = torch.cat(self.alphas_cumprod, 0)
        self.sqrt_alphas_cumprod = torch.cat(self.sqrt_alphas_cumprod, 0)
        self.sqrt_one_minus_alphas_cumprod = torch.cat(self.sqrt_one_minus_alphas_cumprod, 0)
        self.posterior_variance = torch.cat(self.posterior_variance, 0)
        self.posterior_log_variance_clipped = torch.cat(self.posterior_log_variance_clipped, 0)
        self.coef1 = torch.cat(self.coef1, 0)
        self.coef2 = torch.cat(self.coef2, 0)

    def p_sample(self, model, x_start, cond):
        if self.noise_steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([self.noise_steps-1] * x_start.shape[0]).to(device)
            x_t = self.add_noise(x_start, t)

        indices = list(range(self.denoise_steps))[::-1]
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(device)
            
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, cond, t)
            x_t = model_mean
        return x_t
            
    def add_noise(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod,  t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        tmp1 = torch.arange(timesteps.shape[0]).reshape([-1,1]).to(device)
        tmp2 = timesteps.reshape([-1,1]).to(device)
        arr = arr.to(device)
        res = arr[tmp1, tmp2].float()
        
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def _extract_into_tensor_(self, arr, timesteps, broadcast_shape):
        arr = arr.to(device)
        res = arr[timesteps].float()

        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def p_mean_variance(self, model, x, cond, t):
        model_output = model(x, cond, t)
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        model_mean = (self._extract_into_tensor(self.coef1, t, x.shape) * x - self._extract_into_tensor(self.coef2, t, x.shape) * model_output)
        return model_mean, model_log_variance

    def training_losses(self, model, x_0, cond):
        batch_size = x_0.size(0)
        ts = torch.randint(0, self.train_steps, (batch_size,)).long().to(device)
        noise = torch.randn_like(x_0)
        x_t = self.add_noise(x_0, ts, noise)
        model_output = model(x_t, cond, ts)
        mse = self.mean_flat((noise - model_output) ** 2)
        diff_loss = mse
        return diff_loss
        
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        alphas_cumprod = self.alphas_cumprod.to(device)
        tmp1 = torch.arange(t.shape[0]).reshape([-1,1]).to(device)
        tmp2 = t.reshape([-1,1]).to(device)
        res = alphas_cumprod[tmp1, tmp2].squeeze(-1)
        return res / (1 - res)
    
    


class Denoise(nn.Module):
    def __init__(self, opt, node_num, middle_size, time_emb_dim, norm=False):
        super(Denoise, self).__init__()
        
        self.in_dims = [node_num, middle_size]
        self.out_dims = [middle_size, node_num]
        self.node_num = node_num
        self.middle_size = middle_size
        self.time_emb_dim = time_emb_dim
        self.norm = norm
        self.hidden_size = opt.hidden_size
        self.atten_mode = opt.atten_mode

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        self.encoder_x2e = nn.Sequential(    
            nn.Linear(self.node_num + self.time_emb_dim, self.middle_size),
            nn.ReLU(),
            nn.Linear(self.middle_size, self.hidden_size)
        )
        self.encoder_xc2e = nn.Sequential(    
            nn.Linear(self.node_num, self.middle_size),
            nn.ReLU(),
            nn.Linear(self.middle_size, self.hidden_size)
        )
        self.decoder_e2x = nn.Sequential(    
            nn.Linear(self.hidden_size * 2, self.middle_size),
            nn.ReLU(),
            nn.Linear(self.middle_size, self.node_num)
        )

        self.drop5 = nn.Dropout(0.5)
        self.m1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.m2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()
        
    def sim(self, x0, xi):
            
        z1 = F.normalize(x0, dim=-1)
        z2 = F.normalize(xi, dim=-1)
        s = torch.matmul(z1, z2.transpose(-1, -2))
        return s

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.middle_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x_t, h_c, timesteps, mess_dropout=True):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
            
        emb = self.emb_layer(time_emb)
        
        if self.norm:
            x_t = F.normalize(x_t)
        if mess_dropout:
            x_t = self.drop5(x_t)
                        
        h_s = torch.cat([x_t, emb], dim=-1)
        target_emb, total_emb, mask = h_c

        # bs * hidden_size            
        e1 = self.m1(target_emb)
            
        # num * hidden_size
        e2 = self.m2(total_emb)
        
        sim = torch.softmax(torch.matmul(e1, e2.T), dim=-1)
        h = self.encoder_x2e(h_s)
        h_c = self.encoder_xc2e(sim)
        h = self.decoder_e2x(torch.cat([h, h_c], dim=-1))
        return h

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)


def diffusion_training(model, diffusion_loader, item=True):
    
    batch_loss = 0.
    for node_id, node_neighbor, maximum, minimum in tqdm(diffusion_loader):
        
        node_id, node_neighbor = node_id.to(device), node_neighbor.to(device)
        maximum, minimum = maximum.to(device), minimum.to(device)
        model.diffusion_model.noise_calculation(maximum, minimum)
        
        
        mask = node_neighbor.to(torch.bool).to(torch.int32)
        
        
        if item:
            model.diffusion_opt_i.zero_grad()
            
            item_emb = model.item_embedding(node_id)
            
            condition = [item_emb, model.item_embedding.weight, mask]
            
            diff_loss = model.diffusion_model.training_losses(model.denoise_model_i, node_neighbor, condition)
            
            loss = diff_loss.mean()
            batch_loss += loss.item()
            loss.backward()
            model.diffusion_opt_i.step()
        else:
            model.diffusion_opt_u.zero_grad()
            
            user_emb = model.user_embedding(node_id)
            
            condition = [user_emb, model.user_embedding.weight, mask]
            
            diff_loss = model.diffusion_model.training_losses(model.denoise_model_u, node_neighbor, condition)
            
            loss = diff_loss.mean()
            batch_loss += loss.item()
            loss.backward()
            model.diffusion_opt_u.step()
            
    if item:
        print('item_loss: ',batch_loss / len(diffusion_loader))
    else:
        print('user_loss: ',batch_loss / len(diffusion_loader))


def diffusion_inference(model, diffusion_loader, graph, rebuild_k, noise_ratio, item=True, degree=None):
    
    # 取出

    with torch.no_grad():
        
        gen_row_list = []
        gen_col_list = []
        gen_value_list = []
        
        res_row_list = []
        res_col_list = []
        res_value_list = []

        for node_id, node_neighbor, maximum, minimum in tqdm(diffusion_loader):
    
            node_id, node_neighbor = node_id.to(device), node_neighbor.to(device)
            maximum, minimum = maximum.to(device), minimum.to(device)
            
            # x_0
            model.diffusion_model.noise_calculation(maximum, minimum)
            
            mask = node_neighbor.to(torch.bool).to(torch.int32)
            
            # bs * node_num 
            if item:
                item_emb = model.item_embedding(node_id)
                condition = [item_emb, model.item_embedding.weight, mask]
                denoised_batch = model.diffusion_model.p_sample(model.denoise_model_i, node_neighbor, condition)
            else:
                
                user_emb = model.user_embedding(node_id)
                condition = [user_emb, model.user_embedding.weight, mask]
                denoised_batch = model.diffusion_model.p_sample(model.denoise_model_u, node_neighbor, condition)    
                
            for row_index, x_0, x_0_pred, max_norm in zip(node_id, node_neighbor, denoised_batch, maximum):
                
                res_col = torch.where(x_0 != 0)[0]
                res_row = torch.ones_like(res_col) * row_index
                res_value = x_0_pred[res_col] - x_0[res_col]
                res_row_list.append(res_row)            
                res_col_list.append(res_col)
                res_value_list.append(res_value)
                
                median = torch.median(x_0[x_0 != 0])
                
                select_col = torch.where(torch.logical_and((x_0_pred > median), (x_0 == 0)))[0]
                
                if len(select_col) != 0 and rebuild_k > 0:
                    
                    select_value = x_0_pred[select_col]
                    
                    k = len(select_value) if len(select_value) < rebuild_k else rebuild_k
                
                    value, indices = torch.topk(select_value, k=k)
                    
                    value = value / torch.max(value) * max_norm
                    
                    col = select_col[indices]
                    row = torch.ones_like(col) * row_index
                
                    gen_row_list.append(row)            
                    gen_col_list.append(col)
                    gen_value_list.append(value)
                    

        primal_graph = diffusion_loader.dataset.primal_matrix
        
        res_row = torch.cat(res_row_list).unsqueeze(0)
        res_col = torch.cat(res_col_list).unsqueeze(0)
        res_value = torch.cat(res_value_list)
        res_index = torch.cat([res_row, res_col], dim=0)
        res_graph = torch.sparse.FloatTensor(res_index, res_value, graph.shape).to(device)
            
        if len(gen_row_list) > 0:    
            gen_row = torch.cat(gen_row_list).unsqueeze(0)
            gen_col = torch.cat(gen_col_list).unsqueeze(0)
            gen_value = torch.cat(gen_value_list)
            gen_index = torch.cat([gen_row, gen_col], dim=0)
            gen_graph = torch.sparse.FloatTensor(gen_index, gen_value, graph.shape).to(device)
            aug_graph = (primal_graph + gen_graph + res_graph).coalesce()
        else:
            aug_graph = (primal_graph + res_graph).coalesce()
        
        
        reverse_norm_graph = degree.to(device) * aug_graph
    
        index = reverse_norm_graph._indices().cpu()
        values = reverse_norm_graph._values().cpu()
        graph_sp = sp.coo_matrix((values, (index[0], index[1])), graph.shape)
        graph_aug = makeTorchAdj_noself(graph_sp)
        
    return graph_aug


class Encoder(nn.Module):
    def __init__(self, layers=1):
        super(Encoder, self).__init__()
        
        self.gcn_layers = nn.Sequential(*[GCNLayer() for i in range(layers)])

    def forward(self, encoder_adj, init_emb):
        embeds = [init_emb]
        for i, gcn in enumerate(self.gcn_layers):
            embeds.append(gcn(encoder_adj, embeds[-1]))
        return sum(embeds), embeds