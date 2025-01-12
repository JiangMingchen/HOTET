import os, sys
sys.path.append("..")
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions.wishart import Wishart
import gc

from sklearn.decomposition import PCA

from src import distributions
from src import map_benchmark
from src.embedding import Embedding
from src.hnet_icnn import HyperDenseICNN
import src.map_benchmark as mbm

from src.e_tools import train_identity_map_with_emb
from src.tools import train_wishart_identity_map_with_emb, unfreeze, freeze
from src.h_plotters import plot_gaussian_emb_no_hnet, plot_W2, plot_benchmark_metrics, plot_benchmark_emb
from src.h_metrics import score_fitted_maps, score_baseline_maps, metrics_to_dict

from tqdm import tqdm
from IPython.display import clear_output

from src.mnet import MainDenseICNN

import yaml
from datetime import datetime
import ot
import random
from src.mlp import MMLP as MLP

def push(self, input, weights=None):
    output = autograd.grad(
        outputs=self.forward(input, weights=weights), inputs=input,
        create_graph=True, retain_graph=True,
        only_inputs=True,
        grad_outputs=torch.ones((input.size()[0], 1)).cuda().float()
    )[0]
    return output

def push_nograd(self, input, weights = None):
    '''
    Pushes input by using the gradient of the network. Does not preserve the computational graph.
    Use for pushing large batches (the function uses minibatches).
    '''
    output = torch.zeros_like(input, requires_grad=False)
    output.data = self.push(input, weights).data
    return output

# with open('batch_configs/hnet_MLP_MM_config.yaml', 'r') as file:
#     config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Process some configuration file.')
parser.add_argument('config_path', type=str, help='Path to the configuration file')

# 解析命令行参数
args = parser.parse_args()

# 使用传入的配置文件路径
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)


SEED = config['SEED']
BATCH_SIZE = config['BATCH_SIZE']
GPU_DEVICE = config['GPU_DEVICE']
BENCHMARK = config['BENCHMARK']
MAX_ITER = config['MAX_ITER']
LR = config['LR']
INNER_ITERS = config['INNER_ITERS']
COND_IN_SIZE = config['COND_IN_SIZE']


assert torch.cuda.is_available()

torch.manual_seed(SEED)

def compute_l1_norm(W):
    regularizer = 0.
    for param in W:
        regularizer += torch.sum(torch.abs(param))
    return regularizer

def main_diagonal_sum(matrix):
        size = len(matrix) 
        diagonal_sum = 0
        for i in range(size):
                diagonal_sum += matrix[i][i]
        return diagonal_sum

def l2_uvp(X, X_push, Y, Y_inv,  X_var, Y_var):
        L2_UVP_fwd = (100 * (((Y - X_push) ** 2).sum(dim=1).mean() / torch.tensor(Y_var,dtype=torch.float32).cuda())).item()
        L2_UVP_inv = (100 * (((X - Y_inv) ** 2).sum(dim=1).mean() / torch.tensor(X_var,dtype=torch.float32).cuda())).item()
        dif_fwd = (Y - X_push).detach().cpu().numpy().tolist()
        dif_inv = (Y - X_push).detach().cpu().numpy().tolist()
        return L2_UVP_fwd, L2_UVP_inv, dif_fwd, dif_inv

def calc_W(H:HyperDenseICNN, embedding_x:Embedding, x, config):
    '''
    Usage:
        * `calc_W(H, embedding_x, x, config)` -> `W`
        * `clac_W(H_conj, embedding_y, x, config)` -> `W_conj`
    '''
    if config['emb_method'] == 'none':
        W:MLP = H.forward(cond_in=0)
    else:
        emb_x_batch = embedding_x.forward(x=x, method=config['emb_method'])
        W:MainDenseICNN = H.forward(embedding=emb_x_batch)
    return W

current_time = datetime.now()
folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
for DIM in [2,4,8,16,32,64]:

    L1 = 1e-10
    LAMBDA = 1e3
    D_HYPERPARAMS = {
    'in_dim' : DIM,
    'rank' : 1,
    'hidden_layer_sizes' : [max(2*DIM, 64), max(2*DIM, 64), max(DIM, 32)],
    'strong_convexity' : 1e-4,
    'activation': 'celu'
    }

    D = MainDenseICNN(**D_HYPERPARAMS).cuda()
    D_conj = MainDenseICNN(**D_HYPERPARAMS).cuda()

    embedding_x = Embedding(
        dim=DIM,
        cond_in_size=config['h_hyperparams']['cond_in_size']
    ).cuda()
    embedding_y = Embedding(
        dim=DIM,
        cond_in_size=config['h_hyperparams']['cond_in_size']
    ).cuda()

    H = HyperDenseICNN(
        target_shapes=D.param_shapes, embedding_size = DIM, convexify_id=D.convex_layer_indices).cuda()
    H_conj = HyperDenseICNN(
        target_shapes=D_conj.param_shapes, embedding_size = DIM,  convexify_id=D.convex_layer_indices).cuda()



    setattr(MLP, "push", push)
    setattr(MLP, 'push_nograd', push_nograd)
    # D = MLP(
    #     n_in=DIM, n_out=1, hidden_layers= D_HYPERPARAMS['hidden_layer_sizes'],
    #     activation_fn=F.celu
    # ).cuda()
    # D_conj = MLP(
    #     n_in=DIM, n_out=1, hidden_layers=D_HYPERPARAMS['hidden_layer_sizes'],
    #     activation_fn=F.celu
    # ).cuda()


    pretrain_sampler = distributions.StandardNormalSampler(dim=DIM)
    print('Pretraining identity potential with embedding. Final MSE:', train_identity_map_with_emb(D, H, embedding_x, pretrain_sampler, blow=3, lr=1e-4))
    H_conj.load_state_dict(H.state_dict())
    embedding_y.load_state_dict(embedding_x.state_dict())
    del pretrain_sampler
    embedding_y.load_state_dict(embedding_x.state_dict())
    H_conj.load_state_dict(H.state_dict())

    emb_x_opt = torch.optim.Adam(embedding_x.parameters(), lr=LR)
    emb_y_opt = torch.optim.Adam(embedding_y.parameters(), lr=LR)
    H_opt = torch.optim.Adam(H.parameters(), lr=LR)
    H_conj_opt = torch.optim.Adam(H_conj.parameters(), lr=LR)



    W2_history = []
    W2_history_1000 = {'W2_history':[]}
    assert torch.cuda.is_available()

    benchmark = mbm.Mix3ToMix10Benchmark(DIM)
    emb_X = PCA(n_components=2).fit(benchmark.input_sampler.sample(2**14).cpu().detach().numpy())
    emb_Y = PCA(n_components=2).fit(benchmark.output_sampler.sample(2**14).cpu().detach().numpy())


    path = os.path.join(f"../evaluation/W2B_ALL_MM/", folder_name)
    path = os.path.join(path, f"W2B_{DIM}_hnet_results/")
    os.makedirs(path, exist_ok=True)

    with open(path + '/this_config.yaml', 'w') as file:
        yaml.dump(config, file)

    W_dis_dict = {}
    L2_UVP_dict = {'L2_UVP_fwd':[], 'L2_UVP_inv':[]}
    dif_dict = {'dif_fwd':[], 'dif_inv':[]}
    metrics = dict(L2_UVP_fwd=[], cos_fwd=[], L2_UVP_inv=[], cos_inv=[])
    baselines = {
        baseline : metrics_to_dict(*score_baseline_maps(benchmark, baseline))
        for baseline in ['identity', 'constant', 'linear']
    }
    L2_UVP_fwd_min, L2_UVP_inv_min = np.inf, np.inf

    for iteration in tqdm(range(MAX_ITER)):
        X = benchmark.input_sampler.sample(BATCH_SIZE)
        X.requires_grad_(True)
        Y = benchmark.output_sampler.sample(BATCH_SIZE)
        Y.requires_grad_(True)
        
        unfreeze(H)
        freeze(H_conj)
        unfreeze(embedding_x)
        freeze(embedding_y)

        # Negative Wasserstein distance
        W = calc_W(H, embedding_x, X, config)
        W_conj = calc_W(H_conj, embedding_y, Y, config)

        Y_inv = D_conj.push(Y, W_conj).detach()
        H.zero_grad()
        H_opt.zero_grad()
        embedding_x.zero_grad()
        emb_x_opt.zero_grad()
        W_loss = (D(X, W) - D(Y_inv, W)).mean()

        # Non-backpropagated part
        with torch.no_grad():
            W_loss_nograd = (- (X ** 2).sum(dim=1) / 2).mean() +\
                ((Y_inv * Y).sum(dim=1) - (Y_inv ** 2).sum(dim=1) / 2).mean()

        W2_history.append(-W_loss.item() - W_loss_nograd.item())
        W_loss.backward()
        H_opt.step(); emb_x_opt.step()

        freeze(H)
        unfreeze(H_conj)
        freeze(embedding_x)
        unfreeze(embedding_y)

        for inner_it in range(INNER_ITERS):
            # TODO: 用点集训练而不是从分布中sample, 按batch取点，Y只有一个distribution

            Y = benchmark.output_sampler.sample(BATCH_SIZE)
            Y.requires_grad_(True)
            
            H_conj.zero_grad()
            H_conj_opt.zero_grad()
            embedding_y.zero_grad()
            emb_y_opt.zero_grad()

            W = calc_W(H, embedding_x, X, config)
            W_conj = calc_W(H_conj, embedding_y, Y, config)
            
            Y_push = D_conj.push(Y, W_conj)
            conj_loss = (D(Y_push, W) - (Y_push * Y).sum(dim=1, keepdims=True)).mean()
            conj_loss.backward()
            H_conj_opt.step(); emb_y_opt.step()
        
        if iteration % 1000 == 0:
            L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv = score_fitted_maps(
                benchmark, D, D_conj, W, W_conj)
            W2_history_1000['W2_history'].append(-W_loss.item() - W_loss_nograd.item())
            metrics['L2_UVP_fwd'].append(L2_UVP_fwd)
            metrics['cos_fwd'].append(cos_fwd)
            metrics['L2_UVP_inv'].append(L2_UVP_inv)
            metrics['cos_inv'].append(cos_inv)

            fig, axes, left = plot_benchmark_emb(
                benchmark, emb_X, emb_Y, D, D_conj, W, W_conj)
            fig.savefig(os.path.join(path, "benchmark_emb_"+str(iteration)+".png"))

            fig, axes = plot_W2(benchmark, W2_history)
            fig.savefig(os.path.join(path, "W2_history_"+str(iteration)+".png"))

            fig, axes = plot_benchmark_metrics(benchmark, metrics, baselines)
            fig.savefig(os.path.join(path, "benchmark_metrics_"+str(iteration)+".png"))

    dicts = [W2_history_1000, metrics]
    with open(os.path.join(path, "logs.json"), 'w') as file:
        json.dump(dicts, file)
    with open(os.path.join(path, "difs.json"), 'w') as file:
        json.dump(dif_dict, file)

    torch.save(embedding_x.state_dict(), os.path.join(path, 'emb_x.pth'))
    torch.save(embedding_y.state_dict(), os.path.join(path, 'emb_y.pth'))
    torch.save(H.state_dict(), os.path.join(path, 'H.pth'))
    torch.save(H_conj.state_dict(), os.path.join(path, 'H_conj.pth'))