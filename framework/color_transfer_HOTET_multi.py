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

def list_files(directory):
    file_paths = []  # 创建一个空列表来存储文件路径
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)  # 拼接完整的文件路径
            file_paths.append(file_path)  # 将文件路径添加到列表中
    return file_paths

directory_path = '../data/paintings'
im_path_list = list_files(directory_path)

def get_image_name(im_x_path):
    # 使用os.path.basename获取文件名（包括扩展名）
    file_name_with_extension = os.path.basename(im_x_path)
    # 分割文件名和扩展名
    file_name, file_extension = os.path.splitext(file_name_with_extension)
    return file_name

class ImagesSampler:
    def __init__(self, im_x_path_list, im_y_path):
        self.X_samplers = []
        self.im_Xs = []
        for im_x_path in im_x_path_list:
            im_X = Image.open(im_x_path).convert('RGB')
            im_X_name = get_image_name(im_x_path)
            self.im_Xs.append((im_X, im_X_name))
            X_sampler = distributions.TensorDatasetSampler(
                (np.asarray(im_X).transpose(2, 0, 1).reshape(3, -1) / 255.).T,
                requires_grad=True
                )
            self.X_samplers.append(X_sampler)
        
        im_Y_name = get_image_name(im_y_path)
        print(im_Y_name)
        im_Y = Image.open(im_y_path).convert('RGB')
        
        self.Y_sampler = distributions.TensorDatasetSampler(
            (np.asarray(im_Y).transpose(2, 0, 1).reshape(3, -1) / 255.).T,
            requires_grad=True
        )
        self.im_Y = (im_Y, im_Y_name)

    def sample(self, batch_size=8):
        batch = random.sample(self.X_samplers, batch_size)
        return batch
            
images_sampler = ImagesSampler(im_x_path_list=im_path_list[0:-1], im_y_path=im_path_list[-1])
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



import torch
import torch
from tqdm import tqdm

def train_model_with_copies(MAX_ITER, BATCH_SIZE, L1, LAMBDA, X_sampler, Y_sampler, H, H_conj, embedding_x, embedding_y, D, D_conj, config):


    W2_history = []
    
    # 创建原始模型参数的副本

    embedding_x_copy = Embedding(
        dim=DIM,
        cond_in_size=config['COND_IN_SIZE']
    ).cuda()
    embedding_y_copy = Embedding(
        dim=DIM,
        cond_in_size=config['COND_IN_SIZE']
    ).cuda()
    h_layers = [max(2*DIM, 64), max(2*DIM, 64), max(2*DIM, 64)]
    H_copy = HyperDenseICNN(
        target_shapes=D.param_shapes, convexify_id=D.convex_layer_indices, embedding_size=DIM, layers=h_layers, verbose=False).cuda()
    H_conj_copy = HyperDenseICNN(
        target_shapes=D_conj.param_shapes, convexify_id=D.convex_layer_indices, embedding_size=DIM, layers=h_layers, verbose=False).cuda()
    

    emb_x_opt_copy = torch.optim.Adam(embedding_x_copy.parameters(), lr=LR , betas=(0.8, 0.99))
    emb_y_opt_copy = torch.optim.Adam(embedding_y_copy.parameters(), lr=LR, betas=(0.4, 0.4))
    H_opt_copy = torch.optim.Adam(H_copy.parameters(), lr=LR, betas=(0.8, 0.99))
    H_conj_opt_copy = torch.optim.Adam(H_conj_copy.parameters(), lr=LR, betas=(0.4, 0.4))
    
    
    embedding_x_copy.load_state_dict(embedding_x.state_dict())
    embedding_y_copy.load_state_dict(embedding_y.state_dict())
    H_copy.load_state_dict(H.state_dict())
    H_conj_copy.load_state_dict(H_conj.state_dict())

    for iteration in range(MAX_ITER):
        # 采样数据
        X = X_sampler.sample(BATCH_SIZE)
        Y = Y_sampler.sample(BATCH_SIZE)
        X.requires_grad_(True)
        Y.requires_grad_(True)
        
        # 将副本的参数解封，以便进行梯度更新
        unfreeze(H_copy)
        unfreeze(H_conj_copy)
        unfreeze(embedding_x_copy)
        unfreeze(embedding_y_copy)

        # 清空梯度
        H_copy.zero_grad()
        H_opt_copy.zero_grad()
        embedding_x_copy.zero_grad()
        emb_x_opt_copy.zero_grad()
        H_conj_copy.zero_grad()
        H_conj_opt_copy.zero_grad()
        embedding_y_copy.zero_grad()
        emb_y_opt_copy.zero_grad()

        # 计算W和W_conj
        W_conj = calc_W(H_conj_copy, embedding_y_copy, X, config) # change change！
        Y_push = D_conj.push(Y, W_conj).detach()

        W = calc_W(H_copy, embedding_x_copy, X, config)

        # 计算损失
        W_loss = (D(X, W) - D(Y_push, W)).mean()

        # 非反向传播部分
        with torch.no_grad():
            W_loss_nograd = (- (X ** 2).sum(dim=1) / 2).mean() +\
                ((Y_push * Y).sum(dim=1) - (Y_push ** 2).sum(dim=1) / 2).mean()

        # 计算正则化项
        D_reg = compute_l1_norm(W)
        D_conj_reg = compute_l1_norm(W_conj)
        
        W_loss += (L1 * (D_reg + D_conj_reg))
        cycle_loss_YXY = ((D.push(D_conj.push(Y, W_conj), W) - Y.detach()) ** 2).mean()
        
        W_loss += LAMBDA * cycle_loss_YXY

        # 记录损失并执行反向传播
        W2_history.append(-W_loss.item() - W_loss_nograd.item())
        W_loss.backward()

        # 更新副本模型的参数
        H_opt_copy.step()
        emb_x_opt_copy.step()
        H_conj_opt_copy.step()
        emb_y_opt_copy.step()

    return H_copy, H_conj_copy, embedding_x_copy, embedding_y_copy

# 辅助函数，用于复制模型参数
def copy_model_parameters(model):
    new_model = model.__class__()
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), new_model.parameters()):
            new_param.data.copy_(param.data)
    return new_model

# 注意：这个函数假设calc_W, compute_l1_norm, unfreeze等函数已经定义。
# 另外，H_opt.step()等优化器的step方法需要能够接收模型参数作为参数，以便正确更新副本的参数。

W2_history = []
assert torch.cuda.is_available()


path = os.path.join(f"../evaluation/Color_Transfer_Multi/", folder_name)
os.makedirs(path, exist_ok=True)

images_sampler = ImagesSampler(im_x_path_list=im_path_list[0:-1], im_y_path=im_path_list[-1])

for iteration in tqdm(range(MAX_ITER)):
    x_samplers = images_sampler.sample(batch_size=8)
    Y_sampler = images_sampler.Y_sampler
    
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

    W_loss = 0
    W_loss_nograd = 0

    for X_sampler in x_samplers:
        X = X_sampler.sample(BATCH_SIZE)
        Y = Y_sampler.sample(BATCH_SIZE)
        X.requires_grad_(True)
        Y.requires_grad_(True)

        
        W_conj = calc_W(H_conj, embedding_y, X, config)
        Y_push = D_conj.push(Y, W_conj).detach()

        # calc W
        W = calc_W(H, embedding_x, X, config)
        W_conj = calc_W(H_conj, embedding_y, X, config)

        W_loss += (D(X, W) - D(Y_push, W)).mean() / len(x_samplers)

        # Non-backpropagated part
        with torch.no_grad():
            W_loss_nograd += (- (X ** 2).sum(dim=1) / 2).mean() +\
                ((Y_push * Y).sum(dim=1) - (Y_push ** 2).sum(dim=1) / 2).mean() / len(x_samplers)

        D_reg = compute_l1_norm(W)
        D_conj_reg = compute_l1_norm(W_conj)
        
        W_loss += (L1 * (D_reg + D_conj_reg)) / len(x_samplers)
        cycle_loss = ((D.push(D_conj.push(Y, W_conj), W) - Y.detach()) ** 2).mean()
        
        W_loss += (LAMBDA * cycle_loss) / len(x_samplers)
    
    W_loss.backward()
    H_opt.step(); emb_x_opt.step()
    H_conj_opt.step();emb_y_opt.step()



# Evaluation

# torch.save(embedding_x.state_dict(), os.path.join(path, 'emb_x.pth'))
# torch.save(embedding_y.state_dict(), os.path.join(path, 'emb_y.pth'))
# torch.save(H.state_dict(), os.path.join(path, 'H.pth'))
# torch.save(H_conj.state_dict(), os.path.join(path, 'H_conj.pth'))

# # 加载每个模型的状态字典
# emb_x_dict = torch.load('../evaluation/Color_Transfer_Multi/2024-08-12_18-57-19/emb_x.pth')
# emb_y_dict = torch.load('../evaluation/Color_Transfer_Multi/2024-08-12_18-57-19/emb_y.pth')
# H_dict = torch.load('../evaluation/Color_Transfer_Multi/2024-08-12_18-57-19/H.pth')
# H_conj_dict = torch.load('../evaluation/Color_Transfer_Multi/2024-08-12_18-57-19/H_conj.pth')

# # 使用 load_state_dict 方法更新模型参数
# embedding_x.load_state_dict(emb_x_dict)
# embedding_y.load_state_dict(emb_y_dict)
# H.load_state_dict(H_dict)
# H_conj.load_state_dict(H_conj_dict)

freeze(H)
freeze(H_conj)
freeze(embedding_x)
freeze(embedding_y)

im_Y, im_Y_path = images_sampler.im_Y
Y_sampler = images_sampler.Y_sampler
for im_X, im_X_path in tqdm(images_sampler.im_Xs):
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
    
    image_pil_x = Image.fromarray(im_X_pushed.astype('uint8'), 'RGB')           
    image_pil_x.save(os.path.join(path, f"{im_X_path}.png"))

    
    Y = (np.asarray(im_Y).transpose(2, 0, 1).reshape(3, -1) / 255.).T
    
    Y_pushed = np.zeros_like(Y)
    
    im_X_resized = im_X.resize(im_Y.size, Image.LANCZOS)
    
    X_resize_sampler = distributions.TensorDatasetSampler(
        (np.asarray(im_X_resized).transpose(2, 0, 1).reshape(3, -1) / 255.).T,
        requires_grad=True
        )
    
    H_copy, H_conj_copy, embedding_x_copy, embedding_y_copy = train_model_with_copies(10, BATCH_SIZE, L1, LAMBDA, X_resize_sampler, Y_sampler, H, H_conj, embedding_x, embedding_y, D, D_conj, config)

    X_resized = (np.asarray(im_X_resized).transpose(2, 0, 1).reshape(3, -1) / 255.).T
    pos = 0; batch = 4999
    while pos < len(Y):
        W_conj = calc_W(H_conj_copy, embedding_y_copy, torch.tensor(X_resized[pos:pos+batch], dtype=torch.float32).cuda(), config)
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
    image_pil_y = Image.fromarray(im_Y_pushed.astype('uint8'), 'RGB')           
    image_pil_y.save(os.path.join(path, f"tran_{im_X_path}.png"))

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