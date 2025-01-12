import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from torch import Tensor
import random

def plot_rgb_cloud(cloud, ax):
    colors = np.clip(cloud, 0, 1)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=colors)
    ax.set_xlabel('Red'); ax.set_ylabel('Green'); ax.set_zlabel('Blue');
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    
def plot_training_phase(
    benchmark, pca, D_list, D_conj_list,
    G=None, Z_sampler=None,
    plot_batchsize=250, partsize=(5, 5), dpi=200
):
    plot_G = True if ((G is not None) and (Z_sampler is not None)) else False
    plot_B = True if (benchmark.bar_sampler is not None) or (benchmark.gauss_bar_sampler is not None) else False
    
    fig, axes = plt.subplots(
        3, benchmark.num + 2,
        figsize=(partsize[0] * (benchmark.num+2), 3 * partsize[1]),
        sharex=True, sharey=True, dpi=dpi
    )
    
    # Original distributions, pushed and inverse pushed from G(Z)
    if plot_G:
        Z = Z_sampler.sample(plot_batchsize).detach()
        Y = G(Z).detach()
        Y.requires_grad_(True) 
        
        Y_pca = pca.transform(Y.cpu().detach().numpy())
        axes[1,-2].scatter(Y_pca[:, 0], Y_pca[:, 1], edgecolors='black', color='gold')
        axes[1,-2].set_title(f'Generated Barycenter', fontsize=12)
        Y_push_sum = 0.
        
    for n in range(benchmark.num):
        X = benchmark.samplers[n].sample(plot_batchsize)
        X_pca = pca.transform(X.cpu().detach().numpy())
        X_push_pca = pca.transform(D_list[n].push(X).cpu().detach().numpy())

        axes[0, n].scatter(X_pca[:, 0], X_pca[:, 1], edgecolors='black')
        axes[0, n].set_title(f'Initial distribution {n}', fontsize=12)

        axes[1, n].scatter(X_push_pca[:, 0], X_push_pca[:, 1], edgecolors='black', color='orange')
        axes[1, n].set_title(f'Pushed distribution {n}', fontsize=12)
        
        if plot_G:
            Y_push = D_conj_list[n].push(Y).detach()
            Y_push_pca = pca.transform(Y_push.cpu().detach().numpy())
            with torch.no_grad():
                Y_push_sum += benchmark.alphas[n] * Y_push
            axes[2, n].set_title(f'Inverse pushed {n} from generated', fontsize=12)
        else:
            Y = D_list[n].push(X).detach()
            Y.requires_grad_(True)
            Y_push_pca = pca.transform(D_conj_list[n].push(Y).cpu().detach().numpy())
            axes[2, n].set_title(f'Inverse pushed {n}', fontsize=12)
        axes[2, n].scatter(Y_push_pca[:, 0], Y_push_pca[:, 1], edgecolors='black', color='lightblue')
         
    if plot_G:
        Y_push_sum_pca = pca.transform(Y_push_sum.cpu().detach().numpy())
        axes[2, -2].scatter(Y_push_sum_pca[:, 0], Y_push_sum_pca[:, 1], edgecolors='black', color='red')
        axes[2, -2].set_title(f'Generator Target', fontsize=12)
        
    if plot_B:
        if benchmark.bar_sampler is not None:
            axes[1, -1].set_title(f'True Barycenter', fontsize=12)
            axes[2, -1].set_title(f'True Barycenter', fontsize=12)
            Y = benchmark.bar_sampler.sample(plot_batchsize).cpu().detach().numpy()
        else:
            axes[1, -1].set_title(f'Gaussian Barycenter', fontsize=12)
            axes[2, -1].set_title(f'Gaussian Barycenter', fontsize=12)
            Y = benchmark.gauss_bar_sampler.sample(plot_batchsize).cpu().detach().numpy()
        Y_pca = pca.transform(Y)
        axes[1, -1].scatter(Y_pca[:, 0], Y_pca[:, 1], edgecolors='black', color='green')
        axes[2, -1].scatter(Y_pca[:, 0], Y_pca[:, 1], edgecolors='black', color='green')
            
    gc.collect()
    torch.cuda.empty_cache()
        
    return fig, axes

def plot_colored_cloud(cloud, ax):
    ax._axis3don = False
    colors = (cloud - cloud.min(axis=0)) / (cloud.max(axis=0) - cloud.min(axis=0))
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=colors)
    ax.set_xlabel('Red'); ax.set_ylabel('Green'); ax.set_zlabel('Blue');

def push_img(im, D):
    X = (np.asarray(im).transpose(2, 0, 1).reshape(3, -1) / 255.).T
    X_pushed = np.zeros_like(X)
    pos = 0; batch = 4999
    while pos < len(X):
        X_pushed[pos:pos+batch] = D.push(
            torch.tensor(X[pos:pos+batch], device='cuda', requires_grad=True).float()
        ).detach().cpu().numpy()
        pos += batch
    
    im_pushed = (
        np.clip(
            (X_pushed.T.reshape(
                np.asarray(im).transpose(2, 0, 1).shape
            )).transpose(1, 2, 0), 0, 1) * 255
    ).astype(int)
    return im_pushed

def plot_training_phase_palettes(
    benchmark, D_list, D_conj_list,
    plot_batchsize=250, partsize=(5, 5), dpi=200,
    elev=0., azim=40
):    
    fig, axes = plt.subplots(
        3, benchmark.num,
        figsize=(partsize[0] * (benchmark.num), 3 * partsize[1]),
        sharex=True, sharey=True, dpi=dpi,
        subplot_kw=dict(projection='3d')
    )
        
    for n in range(benchmark.num):
        X = benchmark.samplers[n].sample(plot_batchsize)
        X_np = np.clip(X.cpu().detach().numpy(), 0, 1)
        X_push_np = np.clip(D_list[n].push(X).cpu().detach().numpy(), 0, 1)
        
        plot_rgb_cloud(X_np, axes[0, n])
        axes[0, n].set_title(f'Initial distribution {n}', fontsize=12)

        plot_rgb_cloud(X_push_np, axes[1, n])
        axes[1, n].set_title(f'Pushed distribution {n}', fontsize=12)
        
        Y = D_list[n].push(X).detach()
        Y.requires_grad_(True)
        Y_push_np = np.clip(D_conj_list[n].push(Y).cpu().detach().numpy(), 0, 1)
        axes[2, n].set_title(f'Inverse pushed {n}', fontsize=12)
        plot_rgb_cloud(Y_push_np, axes[2, n])
              
    gc.collect()
    torch.cuda.empty_cache()
        
    return fig, axes

def plot_training_phase_im(
    imgs, D_list, D_conj_list,
    plot_batchsize=250, partsize=(5, 5), dpi=200,
    elev=0., azim=40
):    
    fig, axes = plt.subplots(
        3, len(imgs),
        figsize=(partsize[0] * (len(imgs)), 3 * partsize[1]),
        dpi=dpi
    )
        
    for n in range(len(imgs)):
        X = imgs[n]
        axes[0, n].imshow(X)
        axes[0, n].set_title(f'Initial distribution {n}', fontsize=12)
        
        X_push = push_img(X, D_list[n])
        axes[1, n].imshow(X_push)
        axes[1, n].set_title(f'Pushed distribution {n}', fontsize=12)
        
        X_push_inv = push_img(X_push, D_conj_list[n])
        axes[2, n].imshow(X_push_inv)
        axes[2, n].set_title(f'Inverse pushed {n}', fontsize=12)
              
    gc.collect()
    torch.cuda.empty_cache()
        
    return fig, axes

# functions in original plotters.py
# see https://github.com/iamalexkorotin/Wasserstein2Benchmark/blob/main/src/plotters.py

from .tools import ewma, freeze

def plot_benchmark_emb(benchmark, emb_X, emb_Y, D, D_conj, W, W_conj, size=1024):
    # 设置随机种子
    np.random.seed(42)  # 42是常用的种子值，你可以选择任何整数
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # 如果使用多GPU，也需要设置这个
    freeze(D); freeze(D_conj)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
    
    Y = benchmark.output_sampler.sample(size); Y.requires_grad_(True)
    X = benchmark.input_sampler.sample(size); X.requires_grad_(True)

    Y_inv = emb_X.transform(D_conj.push_nograd(Y, W_conj).cpu().numpy())
    Y = emb_Y.transform(Y.cpu().detach().numpy())
    
    X_push = emb_Y.transform(D.push_nograd(X, W).cpu().numpy())
    X = emb_X.transform(X.cpu().detach().numpy())

    axes[0, 0].scatter(X[:, 0], X[:, 1], edgecolors='black')
    axes[0, 1].scatter(Y[:, 0], Y[:, 1], edgecolors='black')
    axes[1, 0].scatter(Y_inv[:, 0], Y_inv[:, 1], c='peru', edgecolors='black')
    axes[1, 1].scatter(X_push[:, 0], X_push[:, 1], c='peru', edgecolors='black')

    axes[0, 0].set_title(r'Ground Truth Input $\mathbb{P}$', fontsize=12)
    axes[0, 1].set_title(r'Ground Truth Output $\mathbb{Q}$', fontsize=12)
    axes[1, 0].set_title(r'Inverse Map $\nabla\overline{\psi_{\omega}}\circ\mathbb{Q}$', fontsize=12)
    axes[1, 1].set_title(r'Forward Map $\nabla\psi_{\theta}\circ\mathbb{P}$', fontsize=12)
    
    fig.tight_layout()
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes, [X, X_push, Y, Y_inv]

def plot_gaussian_emb(num_train, iteration, train_means, emb_X, emb_Y, D, D_conj, W, W_conj, size=1024, dim = 2):
    freeze(D); freeze(D_conj)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
    std_dev = 1

    Y =  torch.tensor(np.random.normal(loc=np.zeros(dim), scale=std_dev, size=(size, dim)), dtype=torch.float32).cuda()
    Y.requires_grad_(True)
    train_data =  [torch.tensor(np.random.normal(loc=mean, scale=std_dev, size=(size, dim)), dtype=torch.float32).cuda() for mean in train_means]
    X = train_data[iteration % num_train]
    X.requires_grad_(True)

    Y_inv = emb_X.transform(D_conj.push_nograd(Y, W_conj).cpu().numpy())
    Y = emb_Y.transform(Y.cpu().detach().numpy())
    
    X_push = emb_Y.transform(D.push_nograd(X, W).cpu().numpy())
    X = emb_X.transform(X.cpu().detach().numpy())

    axes[0, 0].scatter(X[:, 0], X[:, 1], edgecolors='black')
    axes[0, 1].scatter(Y[:, 0], Y[:, 1], edgecolors='black')
    axes[1, 0].scatter(Y_inv[:, 0], Y_inv[:, 1], c='peru', edgecolors='black')
    axes[1, 1].scatter(X_push[:, 0], X_push[:, 1], c='peru', edgecolors='black')

    axes[0, 0].set_title(r'Ground Truth Input $\mathbb{P}$', fontsize=12)
    axes[0, 1].set_title(r'Ground Truth Output $\mathbb{Q}$', fontsize=12)
    axes[1, 0].set_title(r'Inverse Map $\nabla\overline{\psi_{\omega}}\circ\mathbb{Q}$', fontsize=12)
    axes[1, 1].set_title(r'Forward Map $\nabla\psi_{\theta}\circ\mathbb{P}$', fontsize=12)
    
    fig.tight_layout()
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_mixgaussian_emb(X,  Y, emb_X, emb_Y, D, D_conj, W, W_conj, size=1024, dim = 2):
    freeze(D); freeze(D_conj)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
    X.requires_grad_(True)
    Y.requires_grad_(True)
    Y_inv = emb_X.transform(D_conj.push_nograd(Y, W_conj).cpu().numpy())
    Y = emb_Y.transform(Y.cpu().detach().numpy())
    
    X_push = emb_Y.transform(D.push_nograd(X, W).cpu().numpy())
    X = emb_X.transform(X.cpu().detach().numpy())

    axes[0, 0].scatter(X[:, 0], X[:, 1], edgecolors='black')
    axes[0, 1].scatter(Y[:, 0], Y[:, 1], edgecolors='black')
    axes[1, 0].scatter(Y_inv[:, 0], Y_inv[:, 1], c='peru', edgecolors='black')
    axes[1, 1].scatter(X_push[:, 0], X_push[:, 1], c='peru', edgecolors='black')

    axes[0, 0].set_title(r'Ground Truth Input $\mathbb{P}$', fontsize=12)
    axes[0, 1].set_title(r'Ground Truth Output $\mathbb{Q}$', fontsize=12)
    axes[1, 0].set_title(r'Inverse Map $\nabla\overline{\psi_{\omega}}\circ\mathbb{Q}$', fontsize=12)
    axes[1, 1].set_title(r'Forward Map $\nabla\psi_{\theta}\circ\mathbb{P}$', fontsize=12)
    
    fig.tight_layout()
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_gaussian_emb_no_hnet(X_points, Y_points,  emb_X, emb_Y, D, D_conj, size=1024, dim = 2):
    freeze(D); freeze(D_conj)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)

    indices = np.random.choice(len(X_points), size, replace=True)
    X = torch.tensor(X_points[indices], dtype=torch.float32).cuda()
    X.requires_grad_(True)
    indices = np.random.choice(len(Y_points), size, replace=True)
    Y = torch.tensor(Y_points[indices], dtype=torch.float32).cuda()
    Y.requires_grad_(True)

    Y_inv = emb_X.transform(D_conj.push_nograd(Y).cpu().numpy())
    Y = emb_Y.transform(Y.cpu().detach().numpy())
    
    X_push = emb_Y.transform(D.push_nograd(X).cpu().numpy())
    X = emb_X.transform(X.cpu().detach().numpy())

    axes[0, 0].scatter(X[:, 0], X[:, 1], edgecolors='black')
    axes[0, 1].scatter(Y[:, 0], Y[:, 1], edgecolors='black')
    axes[1, 0].scatter(Y_inv[:, 0], Y_inv[:, 1], c='peru', edgecolors='black')
    axes[1, 1].scatter(X_push[:, 0], X_push[:, 1], c='peru', edgecolors='black')

    axes[0, 0].set_title(r'Ground Truth Input $\mathbb{\mu}$', fontsize=12)
    axes[0, 1].set_title(r'Ground Truth Output $\mathbb{\nu}$', fontsize=12)
    axes[1, 0].set_title(r'Inverse Map $\nabla\varphi^*\circ\mathbb{\nu}$', fontsize=12)
    axes[1, 1].set_title(r'Forward Map $\nabla\varphi\circ\mathbb{\mu}$', fontsize=12)
    
    fig.tight_layout()
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_gaussian_emb_v2(x:Tensor, x_push:Tensor, Y:Tensor, Y_inv:Tensor, x_pca:PCA, Y_pca:PCA):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
    
    Y_inv = x_pca.transform(Y_inv.cpu().numpy())
    Y = Y_pca.transform(Y.cpu().detach().numpy())
    
    X_push = Y_pca.transform(x_push.cpu().numpy())
    X = x_pca.transform(x.cpu().detach().numpy())
    
    axes[0, 0].scatter(X[:, 0], X[:, 1], edgecolors='black')
    axes[0, 1].scatter(Y[:, 0], Y[:, 1], edgecolors='black')
    axes[1, 0].scatter(Y_inv[:, 0], Y_inv[:, 1], c='peru', edgecolors='black')
    axes[1, 1].scatter(X_push[:, 0], X_push[:, 1], c='peru', edgecolors='black')

    axes[0, 0].set_title(r'Ground Truth Input $\mathbb{P}$', fontsize=12)
    axes[0, 1].set_title(r'Ground Truth Output $\mathbb{Q}$', fontsize=12)
    axes[1, 0].set_title(r'Inverse Map $\nabla\overline{\psi_{\omega}}\circ\mathbb{Q}$', fontsize=12)
    axes[1, 1].set_title(r'Forward Map $\nabla\psi_{\theta}\circ\mathbb{P}$', fontsize=12)
    
    fig.tight_layout()
    # torch.cuda.empty_cache()
    gc.collect()
    return fig, axes

def plot_W2(benchmark, W2):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
    ax.set_title('Wasserstein-2', fontsize=12)
    ax.plot(ewma(W2), c='blue', label='Estimated Cost')
    if hasattr(benchmark, 'linear_cost'):
        ax.axhline(benchmark.linear_cost, c='orange', label='Bures-Wasserstein Cost')
    if hasattr(benchmark, 'cost'):
        ax.axhline(benchmark.cost, c='green', label='True Cost')    
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig, ax

def plot_benchmark_metrics(benchmark, metrics, baselines=None):
    fig, axes = plt.subplots(2, 2, figsize=(12,6), dpi=100)
    cmap = {'identity' : 'red', 'linear' : 'orange', 'constant' : 'magenta'}
    
    if baselines is None:
        baselines = {}
    
    for i, metric in enumerate(["L2_UVP_fwd", "L2_UVP_inv", "cos_fwd", "cos_inv"]):
        axes.flatten()[i].set_title(metric, fontsize=12)
        axes.flatten()[i].plot(ewma(metrics[metric]), label='Fitted Transport')
        for baseline in cmap.keys():
            if not baseline in baselines.keys():
                continue
            axes.flatten()[i].axhline(baselines[baseline][metric], label=f'{baseline} baseline', c=cmap[baseline])
            
        axes.flatten()[i].legend(loc='upper left')
    
    fig.tight_layout()
    return fig, axes

def vecs_to_plot(input, shape=(3, 64, 64)):
    return input.reshape(-1, *shape).permute(0, 2, 3, 1).mul(0.5).add(0.5).cpu().numpy().clip(0, 1)
