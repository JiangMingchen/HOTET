import os, sys
sys.path.append("..")
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

INPUT = '../data/color_transfer/rio-dell-angelo-1902.jpg'
OUTPUT = '../data/color_transfer/sarumaru-daiyu.jpg'
SEED = config['SEED']
BATCH_SIZE = config['BATCH_SIZE']
GPU_DEVICE = config['GPU_DEVICE']
MAX_ITER = config['MAX_ITER']
LR = config['LR']
COND_IN_SIZE = config['COND_IN_SIZE']


assert torch.cuda.is_available()

torch.manual_seed(SEED)

def compute_l1_norm(W):
    regularizer = 0.
    for param in W:
        regularizer += torch.sum(torch.abs(param))
    return regularizer

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

# L1 = 1e-10
# LAMBDA = 100.
# DIM = 3
# D_HYPERPARAMS = {
# 'in_dim' : DIM,
# 'rank' : 1,
# 'hidden_layer_sizes' : [max(2*DIM, 64), max(2*DIM, 64), max(DIM, 32)],
# 'strong_convexity' : 1e-4,
# 'activation': 'celu'
# }

# D = MainDenseICNN(**D_HYPERPARAMS).cuda()
# D_conj = MainDenseICNN(**D_HYPERPARAMS).cuda()

# embedding_x = Embedding(
#     dim=DIM,
#     cond_in_size=config['h_hyperparams']['cond_in_size']
# ).cuda()
# embedding_y = Embedding(
#     dim=DIM,
#     cond_in_size=config['h_hyperparams']['cond_in_size']
# ).cuda()

# H = HyperDenseICNN(
#     target_shapes=D.param_shapes, convexify_id=D.convex_layer_indices,embedding_size=DIM,).cuda()
# H_conj = HyperDenseICNN(
#     target_shapes=D_conj.param_shapes, convexify_id=D.convex_layer_indices,embedding_size=DIM,).cuda()

L1 = 1e-10
LAMBDA = 100.
DIM = 3
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
    cond_in_size=config['COND_IN_SIZE']
).cuda()
embedding_y = Embedding(
    dim=DIM,
    cond_in_size=config['COND_IN_SIZE']
).cuda()
h_layers = [max(2*DIM, 64), max(2*DIM, 64), max(2*DIM, 64)]
H = HyperDenseICNN(
    target_shapes=D.param_shapes, convexify_id=D.convex_layer_indices, embedding_size=DIM, layers=h_layers).cuda()
H_conj = HyperDenseICNN(
    target_shapes=D_conj.param_shapes, convexify_id=D.convex_layer_indices, embedding_size=DIM, layers=h_layers).cuda()


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

im_X = Image.open(INPUT).convert('RGB')
im_Y = Image.open(OUTPUT).convert('RGB')

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].imshow(im_X)
axes[1].imshow(im_Y)

X_sampler = distributions.TensorDatasetSampler(
    (np.asarray(im_X).transpose(2, 0, 1).reshape(3, -1) / 255.).T,
    requires_grad=True
)
Y_sampler = distributions.TensorDatasetSampler(
    (np.asarray(im_Y).transpose(2, 0, 1).reshape(3, -1) / 255.).T,
    requires_grad=True
)

class ImagesSampler:
    def __init__(self, im_x_path_list, im_y_path):
        self.X_samplers = []
        for im_x_path in im_x_path_list:
            im_X = Image.open(im_x_path).convert('RGB')
            X_sampler = distributions.TensorDatasetSampler(
                (np.asarray(im_X).transpose(2, 0, 1).reshape(3, -1) / 255.).T,
                requires_grad=True
                )
            self.X_samplers(X_sampler)
        
        im_Y = Image.open(im_y_path).convert('RGB')
        self.Y_sampler = distributions.TensorDatasetSampler(
            (np.asarray(im_Y).transpose(2, 0, 1).reshape(3, -1) / 255.).T,
            requires_grad=True
        )
    def sample(self, batch_size=8):
        batch = random.sample(self.X_samplers, batch_size)
        return batch
            

    
torch.cuda.empty_cache()

pretrain_sampler = distributions.StandardNormalSampler(dim=DIM)
print('Pretraining identity potential with embedding. Final MSE:', train_identity_map_with_emb(D, H, embedding_x, pretrain_sampler, max_iter=5000, blow=3, lr=1e-3, tol=1e-2))
H_conj.load_state_dict(H.state_dict())
embedding_y.load_state_dict(embedding_x.state_dict())
del pretrain_sampler
embedding_y.load_state_dict(embedding_x.state_dict())
H_conj.load_state_dict(H.state_dict())

emb_x_opt = torch.optim.Adam(embedding_x.parameters(), lr=LR , betas=(0.8, 0.99))
emb_y_opt = torch.optim.Adam(embedding_y.parameters(), lr=LR, betas=(0.4, 0.4))
H_opt = torch.optim.Adam(H.parameters(), lr=LR, betas=(0.8, 0.99))
H_conj_opt = torch.optim.Adam(H_conj.parameters(), lr=LR, betas=(0.4, 0.4))



W2_history = []
assert torch.cuda.is_available()


path = os.path.join(f"../evaluation/Color_Transfer/", folder_name)

os.makedirs(path, exist_ok=True)

for iteration in tqdm(range(MAX_ITER)):
    X = X_sampler.sample(BATCH_SIZE)
    Y = Y_sampler.sample(BATCH_SIZE)
    X.requires_grad_(True)
    Y.requires_grad_(True)
    
    unfreeze(H)
    unfreeze(H_conj)
    unfreeze(embedding_x)
    unfreeze(embedding_y)

    # clean gradients
    H.zero_grad()
    H_opt.zero_grad()
    embedding_x.zero_grad()
    emb_x_opt.zero_grad()
    H_conj.zero_grad()
    H_conj_opt.zero_grad()
    embedding_y.zero_grad()
    emb_y_opt.zero_grad()

    W_conj = calc_W(H_conj, embedding_y, Y, config)
    Y_push = D_conj.push(Y, W_conj).detach()

    # calc W
    W = calc_W(H, embedding_x, X, config)
    W_conj = calc_W(H_conj, embedding_y, Y, config)

    W_loss = (D(X, W) - D(Y_push, W)).mean()

    # Non-backpropagated part
    with torch.no_grad():
        W_loss_nograd = (- (X ** 2).sum(dim=1) / 2).mean() +\
            ((Y_push * Y).sum(dim=1) - (Y_push ** 2).sum(dim=1) / 2).mean()

    D_reg = compute_l1_norm(W)
    D_conj_reg = compute_l1_norm(W_conj)
    
    W_loss += (L1 * (D_reg + D_conj_reg))
    cycle_loss_YXY = ((D.push(D_conj.push(Y, W_conj), W) - Y.detach()) ** 2).mean()
    # cycle_loss_XYX = ((D_conj.push(D.push(X, W_conj), W) - X.detach()) ** 2).mean()
    
    W_loss += LAMBDA * (cycle_loss_YXY)

    W2_history.append(-W_loss.item() - W_loss_nograd.item())
    W_loss.backward()
    H_opt.step(); emb_x_opt.step()
    H_conj_opt.step();emb_y_opt.step()

    if iteration % 5000 == 0:
        # clear_output(wait=True)
        print("Iteration", iteration)
        # print("Plotting takes time!")
        
        freeze(H)
        freeze(H_conj)
        freeze(embedding_x)
        freeze(embedding_y)
        
        fig = plt.figure(figsize=(13, 9), dpi=100)
        fig.add_subplot(221)
        plt.imshow(im_X)

        fig.add_subplot(222)
        X = (np.asarray(im_X).transpose(2, 0, 1).reshape(3, -1) / 255.).T
        X_pushed = np.zeros_like(X)
        pos = 0; batch = 4999
        while pos < len(X):
            W = calc_W(H, embedding_x, torch.tensor(X[pos:pos+batch], dtype=torch.float32).cuda(), config)
            X_pushed[pos:pos+batch] = D.push(
                torch.tensor(X[pos:pos+batch], device='cuda', requires_grad=True).float(), W
            ).detach().cpu().numpy()
            pos += batch

        im_X_pushed = (
            np.clip(
                (X_pushed.T.reshape(
                    np.asarray(im_X).transpose(2, 0, 1).shape
                )).transpose(1, 2, 0), 0, 1) * 255
        ).astype(int)
        plt.imshow(im_X_pushed)
        
        image_pil_x = Image.fromarray(im_X_pushed.astype('uint8'), 'RGB')           
        image_pil_x.save(os.path.join(path, "tran_x.png"))

        
        fig.add_subplot(223)
        plt.imshow(im_Y)

        fig.add_subplot(224)
        Y = (np.asarray(im_Y).transpose(2, 0, 1).reshape(3, -1) / 255.).T
        Y_pushed = np.zeros_like(Y)
        pos = 0; batch = 4999
        while pos < len(Y):
            W_conj = calc_W(H_conj, embedding_y, torch.tensor(Y[pos:pos+batch], dtype=torch.float32).cuda(), config)
            Y_pushed[pos:pos+batch] = D_conj.push(
                torch.tensor(Y[pos:pos+batch], device='cuda', requires_grad=True).float(), W_conj
            ).detach().cpu().numpy()
            pos += batch
        
        im_Y_pushed = (
            np.clip(
                (Y_pushed.T.reshape(
                    np.asarray(im_Y).transpose(2, 0, 1).shape
                )).transpose(1, 2, 0), 0, 1) * 255
        ).astype(int)
        plt.imshow(im_Y_pushed)
        
        image_pil_y = Image.fromarray(im_Y_pushed.astype('uint8'), 'RGB')           
        image_pil_y.save(os.path.join(path, "tran_y.png"))
        fig.tight_layout()
        fig.savefig(os.path.join(path, "color_transfer_"+str(iteration)+".png"))

        # fig.tight_layout()
    
        # fig = plt.figure(figsize=(12, 12), dpi=100)
        # ax = fig.add_subplot(221, projection='3d')
        # X = X_sampler.sample(1024)
        # plot_rgb_cloud(X.cpu().detach().numpy(), ax)
        # ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        
        # ax = fig.add_subplot(222, projection='3d')
        # X_pushed = D.push(
        #     torch.tensor(X, device='cuda', dtype=torch.float32, requires_grad=True)
        # )
        # plot_rgb_cloud(X_pushed.cpu().detach().numpy(), ax)
        # ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        
        # ax = fig.add_subplot(223, projection='3d')
        # Y = Y_sampler.sample(1024)
        # plot_rgb_cloud(Y.cpu().detach().numpy(), ax)
        # ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        
        # ax = fig.add_subplot(224, projection='3d')
        # Y_pushed = D_conj.push(
        #     torch.tensor(Y, device='cuda', dtype=torch.float32, requires_grad=True)
        # )
        # plot_rgb_cloud(Y_pushed.cpu().detach().numpy(), ax)
        # ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        # plt.grid()
        
        # plt.show()

        # fig.tight_layout()