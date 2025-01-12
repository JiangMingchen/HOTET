import os, sys
sys.path.append("..")
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
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

from src.tools import train_wishart_identity_map_with_emb, unfreeze, freeze
from src.h_plotters import plot_mixgaussian_emb, plot_W2, plot_benchmark_metrics
from src.h_metrics import score_fitted_maps, score_baseline_maps, metrics_to_dict

from tqdm import tqdm
from IPython.display import clear_output

from src.mnet import MainDenseICNN

import yaml
from datetime import datetime
import ot
import random
from src.mlp import MMLP as MLP


class MixGaussianDistPair:
    def __init__(self, mean_xs, mean_ys, cov_xs, cov_ys, weights):
        self.mean_xs = mean_xs
        self.cov_xs = cov_xs

        self.mean_ys = mean_ys
        self.cov_ys = cov_ys
        self.weights = weights

        self.As = []
        self.bs = []
        for mean_x, mean_y, cov_x, cov_y in zip(mean_xs, mean_ys, cov_xs, cov_ys):
            A, b = ot.gaussian.bures_wasserstein_mapping(mean_x, mean_y, cov_x, cov_y, log=False)
            self.As.append(torch.tensor(A, dtype=torch.float32).cuda())
            self.bs.append(torch.tensor(b, dtype=torch.float32).cuda())
        
        weighted_mean = np.average(self.mean_xs, weights=self.weights, axis=0)       
        total_covariance = np.zeros(self.mean_xs[0].shape + (self.mean_xs[0].shape[0],))
      
        for w, mu, cov in zip(self.weights, self.mean_xs, self.cov_xs):
            mu_mu_T = np.outer(mu, mu)
            weighted_covariance = w * (cov + mu_mu_T - np.outer(weighted_mean, weighted_mean))
            total_covariance += weighted_covariance

        self.var_x = np.sum(np.diag(total_covariance))
        
        weighted_mean = np.average(self.mean_ys, weights=self.weights, axis=0)
        total_covariance = np.zeros(self.mean_ys[0].shape + (self.mean_xs[0].shape[0],))
       
        for w, mu, cov in zip(self.weights, self.mean_ys, self.cov_ys):
            mu_mu_T = np.outer(mu, mu)
            weighted_covariance = w * (cov + mu_mu_T - np.outer(weighted_mean, weighted_mean))
            total_covariance += weighted_covariance
        
        self.var_y = np.sum(np.diag(total_covariance))
    
    def sample(self, batch_size=512, inner_batch=64):
        if not (batch_size / inner_batch).is_integer():
            raise ValueError("batch_size must be divisible by inner_batch without remainder")
        times = int(batch_size/inner_batch)
        X_all = None
        Y_all = None
        for i in range(times):
            component = np.random.choice(range(len(self.weights)), p=self.weights)
            X = np.random.multivariate_normal(self.mean_xs[component], self.cov_xs[component], inner_batch)
            if i == 0:
                X_all = torch.tensor(X, dtype=torch.float32).cuda()
                Y_all = torch.matmul(X_all, self.As[component].T) + self.bs[component]
            else:
                X = torch.tensor(X, dtype=torch.float32).cuda()
                Y = torch.matmul(X, self.As[component].T) + self.bs[component]
                X_all = torch.cat((X_all, X), dim=0)
                Y_all = torch.cat((Y_all, Y), dim=0)
        return X_all, Y_all

class MixPairs:
    def __init__(self, weights, num_iid=500, num_ood=10, num_component=3):         
        
        self.dist_pair_list_iid = []
        self.dist_pair_list_ood = []
        
        cov_dist = Wishart(df=torch.Tensor([DIM+5]), covariance_matrix=torch.eye(DIM))
        
        cov_ys = []
        mean_ys = []
        for i in range(num_component):
            cov_ys.append(np.array(cov_dist.sample())[0])
            mean_ys.append(np.random.uniform(0, 3, DIM))
        
        for i in range(0, num_iid):
            cov_xs = []
            mean_xs = []            
            for i in range(num_component):
                cov_xs.append(np.array(cov_dist.sample())[0])
                mean_xs.append(np.random.uniform(0, 3, DIM))
            dist_pair = MixGaussianDistPair(mean_xs=mean_xs, mean_ys=mean_ys, cov_xs=cov_xs, cov_ys=cov_ys, weights=weights)
            self.dist_pair_list_iid.append(dist_pair)
        
        for i in range(0, num_ood):
            cov_xs = []
            mean_xs = []            
            for i in range(num_component):
                cov_xs.append(np.array(cov_dist.sample())[0])
                mean_xs.append(np.random.uniform(0, 3, DIM))
            dist_pair = MixGaussianDistPair(mean_xs=mean_xs, mean_ys=mean_ys, cov_xs=cov_xs, cov_ys=cov_ys, weights=weights)
            self.dist_pair_list_ood.append(dist_pair)
    
    def sample(self, batch_size=8):
        batch = random.sample(self.dist_pair_list_iid, batch_size)
        return batch
    

def Wasserstein_distance(X, Y):
    n_samples = len(X)
    cost_matrix = ot.dist(X, Y, metric='euclidean')

    # compute the weights of two sets of samples（every point has the same weight）
    a = np.ones((n_samples,)) / n_samples  # weights of the first set
    b = np.ones((n_samples,)) / n_samples  # weights of the second set

    # Compute Wasserstein Distance
    wasserstein_distance = ot.emd2(a, b, cost_matrix)
    return wasserstein_distance

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

DIM = config['DIM']
SEED = config['SEED']
BATCH_SIZE = config['BATCH_SIZE']
GPU_DEVICE = config['GPU_DEVICE']
BENCHMARK = config['BENCHMARK']
MAX_ITER = config['MAX_ITER']
LR = config['LR']
INNER_ITERS = config['INNER_ITERS']
D_HYPERPARAMS = config['D_HYPERPARAMS']
COND_IN_SIZE = config['COND_IN_SIZE']
D_HYPERPARAMS = config['D_HYPERPARAMS']

OUTPUT_PATH = '../logs/' + BENCHMARK
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

METHOD = 'MM'

assert torch.cuda.is_available()

torch.manual_seed(SEED)
gaussians_dif_dict = {}

w_dif_fwd_list = []
w_dif_inv_list = []

dict_final = {}

def main_diagonal_sum(matrix):
        size = len(matrix) 
        diagonal_sum = 0
        for i in range(size):
                diagonal_sum += matrix[i][i]
        return diagonal_sum

def metrics(X, X_push, Y, Y_inv,  X_var, Y_var):
        
        L2_UVP_fwd = (100 * (((Y - X_push) ** 2).sum(dim=1).mean() / torch.tensor(Y_var,dtype=torch.float32).cuda())).item()
        L2_UVP_inv = (100 * (((X - Y_inv) ** 2).sum(dim=1).mean() / torch.tensor(X_var,dtype=torch.float32).cuda())).item()
        
        cost = .5 * ((X - Y) ** 2).sum(dim=1).mean(dim=0).item()
        cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
        (np.sqrt((2 * cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
        cos_inv = (((X - Y) * (Y_inv - Y)).sum(dim=1).mean() / \
        (np.sqrt((2 * cost) * ((Y_inv - Y) ** 2).sum(dim=1).mean().item()))).item()
        return [L2_UVP_fwd, L2_UVP_inv, cos_fwd, cos_inv]

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
def approx_corr(H, embedding_x, X, Y):
    W = calc_W(H, embedding_x, X, config)
    D_X = D(X, W)
    with torch.no_grad():
        XY = X @ Y.transpose(1, 0)
        idx_Y = torch.argmax(XY - D_X, dim=0)
        Y_inv = X[idx_Y]
        
    W_loss_XY = (D_X - D(Y_inv, W)).mean()
    
    # Non-backpropagated part
    with torch.no_grad():
        W_loss_nograd_XY = (- (X ** 2).sum(dim=1) / 2).mean() +\
        ((Y_inv * Y).sum(dim=1) - (Y_inv ** 2).sum(dim=1) / 2 ).mean()
        
    return W_loss_XY, W_loss_nograd_XY

dist_pairs = MixPairs(weights=[0.3, 0.3, 0.4],num_iid=500, num_ood=10, num_component=3)

D = MainDenseICNN(**D_HYPERPARAMS).cuda()
D_conj = MainDenseICNN(**D_HYPERPARAMS).cuda()

embedding_x = Embedding(
    dim=config['DIM'],
    cond_in_size=config['COND_IN_SIZE']
).cuda()
embedding_y = Embedding(
    dim=config['DIM'],
    cond_in_size=config['COND_IN_SIZE']
).cuda()

h_layers = [max(2*DIM, 16), max(2*DIM, 16), max(2*DIM, 16)]
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

import time
start_time = time.time()
gc.collect(); torch.cuda.empty_cache()
print('Pretraining identity potential. Final MSE:', train_wishart_identity_map_with_emb(D, embedding_x, H, dist_pairs.sample(batch_size=8)[0], max_iter=5000, convex=False, blow=3, verbose=True, emb_method=config['emb_method']))

embedding_y.load_state_dict(embedding_x.state_dict())
H_conj.load_state_dict(H.state_dict())

emb_x_opt = torch.optim.Adam(embedding_x.parameters(), lr=LR)
emb_y_opt = torch.optim.Adam(embedding_y.parameters(), lr=LR)
H_opt = torch.optim.Adam(H.parameters(), lr=LR)
H_conj_opt = torch.optim.Adam(H_conj.parameters(), lr=LR)



W2_history = []
W2_history_500 = dict(W2_his=[])


current_time = datetime.now()
folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
path = os.path.join(f"../evaluation/Wishart_{DIM}_mix_hnet_MMB_results/", folder_name)
os.makedirs(path, exist_ok=True)

torch.save(embedding_x.state_dict(), os.path.join(path, 'pre_emb_x.pth'))
torch.save(H.state_dict(), os.path.join(path, 'pre_H.pth'))

with open(path + '/this_config.yaml', 'w') as file:
    yaml.dump(config, file)

W_dis_dict = {}
L2_UVP_dict = {}
dif_dict = {'dif_fwd':[], 'dif_inv':[]}
for iteration in tqdm(range(MAX_ITER)):
    pairs_sampled = dist_pairs.sample(batch_size=8)
    
    W_loss = 0
    W_loss_nograd = 0
    
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


    for pair in pairs_sampled:
        X, Y = pair.sample(batch_size=BATCH_SIZE, inner_batch=32)
        X.requires_grad_(True)
        Y.requires_grad_(True)
        
        W_loss_XY, W_loss_nograd_XY = approx_corr(H, embedding_x, X, Y)
        W_loss_YX, W_loss_nograd_YX = approx_corr(H_conj, embedding_y, Y, X)
        W_loss += ((W_loss_XY + W_loss_YX) / 2) / len(pairs_sampled)

        # Non-backpropagated part
        with torch.no_grad():
            W_loss_nograd += ((W_loss_nograd_XY + W_loss_nograd_YX) / 2) // len(pairs_sampled)
    
    W_loss.backward()
    H_opt.step(); emb_x_opt.step()
    H_conj_opt.step();emb_y_opt.step()
    

    if iteration % 1000 == 0:
        # W2_history_500['W2_his'].append(-W_loss.item() - W_loss_nograd.item())
        #print("Iteration", iteration)
        X_pca, Y_pca = dist_pairs.dist_pair_list_iid[0].sample(batch_size=2**14, inner_batch=512)
        X_pca = X_pca.detach().cpu().numpy()
        Y_pca = Y_pca.detach().cpu().numpy()
        emb_X = PCA(n_components=2).fit(X_pca)
        emb_Y = PCA(n_components=2).fit(Y_pca)
        X_plot, Y_plot = dist_pairs.dist_pair_list_iid[0].sample(batch_size=1024, inner_batch = 64)
        
        W = calc_W(H, embedding_x, X_plot, config)
        W_conj = calc_W(H_conj, embedding_y, Y_plot, config)
        
        fig, axes = plot_mixgaussian_emb(
            X_plot, Y_plot, emb_X, emb_Y, D, D_conj, W, W_conj)
        fig.savefig(os.path.join(path, "benchmark_emb_"+str(iteration)+".png"))
        # plt.show(fig)
        # plt.close(fig)
        eval_metrics = []
        for i, eval_pair in tqdm(enumerate(dist_pairs.dist_pair_list_iid), desc="Eval Iteration"):
            # if iteration == 0:
            #     W_dis_dict[f'dist_{i}'] = {}
            #     L2_UVP_dict[f'dist_{i}'] = {'L2_UVP_fwd':[], 'L2_UVP_inv':[]}
            X_eval, Y_eval = eval_pair.sample(batch_size=1024, inner_batch=64)
            X_eval.requires_grad_(True)
            Y_eval.requires_grad_(True)
            # compute wasserstein distance

            W = calc_W(H, embedding_x, X_eval, config)
            W_conj = calc_W(H_conj, embedding_y, X_eval, config)

            x_push = D.push_nograd(X_eval, W)
            Y_push = D_conj.push_nograd(Y_eval, W_conj) 
            
            eval_metrics.append(metrics(X=X_eval, X_push=x_push, Y=Y_eval, Y_inv = Y_push, X_var=eval_pair.var_x, 
                                            Y_var=eval_pair.var_y))
            

            # L2_UVP_dict[f'dist_{i}']['L2_UVP_fwd'].append(L2_UVP_fwd)
            # L2_UVP_dict[f'dist_{i}']['L2_UVP_inv'].append(L2_UVP_inv)
            # dif_dict[f'dist_{i}']['dif_fwd'].append(dif_fwd)
            # dif_dict[f'dist_{i}']['dif_inv'].append(dif_inv)
            # if W_dis_dict[f'dist_{i}'] == {}:
            #     W_dis_dict[f'dist_{i}']['X_to_Y'] = Wasserstein_distance(X_eval.detach().cpu().numpy(), Y_eval.detach().cpu().numpy())
            #     W_dis_dict[f'dist_{i}']['X_push_to_Y'] = []
            #     W_dis_dict[f'dist_{i}']['X_to_Y_inv'] = []
            #     W_dis_dict[f'dist_{i}']['X_push_to_Y_inv'] = []

            # W_dis_dict[f'dist_{i}']['X_push_to_Y'].append(Wasserstein_distance(x_push.detach().cpu().numpy(), Y_eval.detach().cpu().numpy()))
            # W_dis_dict[f'dist_{i}']['X_to_Y_inv'].append(Wasserstein_distance(X_eval.cpu().detach().numpy(), Y_push.detach().cpu().numpy()))
            # W_dis_dict[f'dist_{i}']['X_push_to_Y_inv'].append(Wasserstein_distance(x_push.detach().cpu().numpy(), Y_push.detach().cpu().numpy()))
        
        test_metrics = []
        for i, test_pair in tqdm(enumerate(dist_pairs.dist_pair_list_ood), desc="Test Iteration"):
            # if iteration == 0:
            #     # W_dis_dict[f'dist_ood_{i}'] = {}
            #     L2_UVP_dict[f'dist_ood_{i}'] = {'L2_UVP_fwd':[], 'L2_UVP_inv':[]}               
            X_test, Y_test = test_pair.sample(batch_size=1024, inner_batch=64)
            X_test.requires_grad_(True)
            Y_test.requires_grad_(True)
            # compute wasserstein distance

            W = calc_W(H, embedding_x, X, config)
            W_conj = calc_W(H_conj, embedding_y, X, config)

            x_push = D.push_nograd(X_test, W)
            Y_push = D_conj.push_nograd(Y_test, W_conj) 
            
            test_metrics.append(metrics(X=X_test, X_push=x_push, Y=Y_test, Y_inv = Y_push, X_var=test_pair.var_x, 
                                            Y_var=test_pair.var_y))
            # L2_UVP_dict[f'dist_ood_{i}']['L2_UVP_fwd'].append(L2_UVP_fwd)
            # L2_UVP_dict[f'dist_ood_{i}']['L2_UVP_inv'].append(L2_UVP_inv)
            # dif_dict[f'dist_{i}']['dif_fwd'].append(dif_fwd)
            # dif_dict[f'dist_{i}']['dif_inv'].append(dif_inv)
            # if W_dis_dict[f'dist_ood_{i}'] == {}:
            #     W_dis_dict[f'dist_ood_{i}']['X_to_Y'] = Wasserstein_distance(X_test.detach().cpu().numpy(), Y_test.detach().cpu().numpy())
            #     W_dis_dict[f'dist_ood_{i}']['X_push_to_Y'] = []
            #     W_dis_dict[f'dist_ood_{i}']['X_to_Y_inv'] = []
            #     W_dis_dict[f'dist_ood_{i}']['X_push_to_Y_inv'] = []

            # W_dis_dict[f'dist_ood_{i}']['X_push_to_Y'].append(Wasserstein_distance(x_push.detach().cpu().numpy(), Y_test.detach().cpu().numpy()))
            # W_dis_dict[f'dist_ood_{i}']['X_to_Y_inv'].append(Wasserstein_distance(X_test.cpu().detach().numpy(), Y_push.detach().cpu().numpy()))
            # W_dis_dict[f'dist_ood_{i}']['X_push_to_Y_inv'].append(Wasserstein_distance(x_push.detach().cpu().numpy(), Y_push.detach().cpu().numpy()))

        # if iteration == 5000:
        #     dict_final[f'distribution_{gaussian_id}'] = W_dis_dict
        
        df = pd.DataFrame(eval_metrics, columns=['L2_UVP_fwd', 'L2_UVP_inv', 'cos_fwd', 'cos_inv'])
        metrics_path = os.path.join(path, f'metrics/eval_iter_{iteration}.csv')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        df.to_csv(metrics_path, index=False)

        df = pd.DataFrame(test_metrics, columns=['L2_UVP_fwd', 'L2_UVP_inv', 'cos_fwd', 'cos_inv'])
        metrics_path = os.path.join(path, f'metrics/test_iter_{iteration}.csv')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        df.to_csv(metrics_path, index=False)

end_time = time.time()
run_time = end_time - start_time
print(run_time)        
        # dicts = [L2_UVP_dict]
        # with open(os.path.join(path, f"logs_{iteration}.json"), 'w') as file:
        #     json.dump(dicts, file)
# with open(os.path.join(path, "difs.json"), 'w') as file:
#     json.dump(dif_dict, file)


# 加载每个模型的状态字典
# emb_x_dict = torch.load('../evaluation/Wishart_4_mix_hnet_MMB_results/2024-08-10_19-48-34/emb_x.pth')
# emb_y_dict = torch.load('../evaluation/Wishart_4_mix_hnet_MMB_results/2024-08-10_19-48-34/emb_y.pth')
# H_dict = torch.load('../evaluation/Wishart_4_mix_hnet_MMB_results/2024-08-10_19-48-34/H.pth')
# H_conj_dict = torch.load('../evaluation/Wishart_4_mix_hnet_MMB_results/2024-08-10_19-48-34/H_conj.pth')

# 使用 load_state_dict 方法更新模型参数
# embedding_x.load_state_dict(emb_x_dict)
# embedding_y.load_state_dict(emb_y_dict)
# H.load_state_dict(H_dict)
# H_conj.load_state_dict(H_conj_dict)

# X_pca, Y_pca = dist_pairs.dist_pair_list_iid[0].sample(batch_size=2**14, inner_batch=512)
# X_pca = X_pca.detach().cpu().numpy()
# Y_pca = Y_pca.detach().cpu().numpy()
# emb_X = PCA(n_components=2).fit(X_pca)
# emb_Y = PCA(n_components=2).fit(Y_pca)
# X_plot, Y_plot = dist_pairs.dist_pair_list_iid[0].sample(batch_size=1024, inner_batch = 64)

# W = calc_W(H, embedding_x, X_plot, config)
# W_conj = calc_W(H_conj, embedding_y, Y_plot, config)

# fig, axes = plot_mixgaussian_emb(
#     X_plot, Y_plot, emb_X, emb_Y, D, D_conj, W, W_conj)
# fig.savefig(os.path.join(path, "benchmark_emb_"+str(5000)+".png"))

# eval_metrics = []
# for i, eval_pair in tqdm(enumerate(dist_pairs.dist_pair_list_iid), desc="Eval Iteration"):

#     X_eval, Y_eval = eval_pair.sample(batch_size=1024, inner_batch=64)
#     X_eval.requires_grad_(True)
#     Y_eval.requires_grad_(True)

#     W = calc_W(H, embedding_x, X_eval, config)
#     W_conj = calc_W(H_conj, embedding_y, X_eval, config)

#     x_push = D.push_nograd(X_eval, W)
#     Y_push = D_conj.push_nograd(Y_eval, W_conj) 
    
#     eval_metrics.append(metrics(X=X_eval, X_push=x_push, Y=Y_eval, Y_inv = Y_push, X_var=eval_pair.var_x, 
#                                     Y_var=eval_pair.var_y))


# test_metrics = []
# for i, test_pair in tqdm(enumerate(dist_pairs.dist_pair_list_ood), desc="Test Iteration"):
              
#     X_test, Y_test = test_pair.sample(batch_size=1024, inner_batch=64)
#     X_test.requires_grad_(True)
#     Y_test.requires_grad_(True)

#     W = calc_W(H, embedding_x, X_test, config)
#     W_conj = calc_W(H_conj, embedding_y, X_test, config)

#     x_push = D.push_nograd(X_test, W)
#     Y_push = D_conj.push_nograd(Y_test, W_conj) 
    
#     test_metrics.append(metrics(X=X_test, X_push=x_push, Y=Y_test, Y_inv = Y_push, X_var=test_pair.var_x, 
#                                     Y_var=test_pair.var_y))

# df = pd.DataFrame(eval_metrics, columns=['L2_UVP_fwd', 'L2_UVP_inv', 'cos_fwd', 'cos_inv'])
# metrics_path = os.path.join(path, f'metrics/eval_iter_{5000}.csv')
# os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
# df.to_csv(metrics_path, index=False)

# df = pd.DataFrame(test_metrics, columns=['L2_UVP_fwd', 'L2_UVP_inv', 'cos_fwd', 'cos_inv'])
# metrics_path = os.path.join(path, f'metrics/test_iter_{5000}.csv')
# os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
# df.to_csv(metrics_path, index=False)
# torch.save(embedding_x.state_dict(), os.path.join(path, 'emb_x.pth'))
# torch.save(embedding_y.state_dict(), os.path.join(path, 'emb_y.pth'))
# torch.save(H.state_dict(), os.path.join(path, 'H.pth'))
# torch.save(H_conj.state_dict(), os.path.join(path, 'H_conj.pth'))