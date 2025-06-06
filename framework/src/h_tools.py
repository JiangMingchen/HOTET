import os, sys
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import sqrtm

import os, sys
import argparse
import collections
from scipy.io import savemat
from tqdm import trange
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import multiprocessing
import itertools

import torch
import torch.nn as nn
from PIL import Image
sys.path.append("..")

import gc

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def score_forward_maps(benchmark, D_list, score_size=1024):
    assert (benchmark.bar_maps is not None) and (benchmark.bar_sampler is not None)
    L2_UVP = []
    Y = benchmark.bar_sampler.sample(score_size).detach()
    for n in range(benchmark.num):
        X = benchmark.samplers[n].sample(score_size)
        X_push = D_list[n].push(X).detach()
        with torch.no_grad():
            # print(X.dtype)
            X_push_true = benchmark.bar_maps[n](X)
            L2_UVP.append(
                100 * (((X_push - X_push_true) ** 2).sum(dim=1).mean() / benchmark.bar_sampler.var).item()
            )
    return L2_UVP

def score_pushforwards(benchmark, D_list, score_size=128*1024, batch_size=1024):
    assert (benchmark.bar_sampler is not None)
    BW2_UVP = []
    if score_size < batch_size:
        batch_size = score_size
    num_chunks = score_size // batch_size
    
    for n in range(benchmark.num):
        X_push = np.vstack([
            D_list[n].push(benchmark.samplers[n].sample(batch_size)).cpu().detach().numpy()
            for _ in range(num_chunks)
        ])
        X_push_cov = np.cov(X_push.T)
        X_push_mean = np.mean(X_push, axis=0)   
        UVP = 100 * calculate_frechet_distance(
            X_push_mean, X_push_cov,
            benchmark.bar_sampler.mean, benchmark.bar_sampler.cov,
        ) / benchmark.bar_sampler.var
        BW2_UVP.append(UVP)
    return BW2_UVP

def score_cycle_consistency(benchmark, D_list, D_conj_list, score_size=1024):
    cycle_UVP = []
    for n in range(benchmark.num):
        X = benchmark.samplers[n].sample(score_size)
        X_push = D_list[n].push(X).detach()
        X_push.requires_grad_(True)
        X_push_inv = D_conj_list[n].push(X_push).detach()
        with torch.no_grad():
            cycle_UVP.append(
                100 * (((X - X_push_inv) ** 2).sum(dim=1).mean() / benchmark.samplers[n].var).item()
            )
    return cycle_UVP

def score_congruence(benchmark, D_conj_list, score_size=1024):
    assert benchmark.bar_sampler is not None
    Y = benchmark.bar_sampler.sample(score_size)
    Y_sum = torch.zeros_like(Y).detach()
    for n in range(benchmark.num):
        Y_push = D_conj_list[n].push(Y).detach()
        with torch.no_grad():
            Y_sum += benchmark.alphas[n] * Y_push
    return 100 * (((Y - Y_sum) ** 2).sum(dim=1).mean() / benchmark.bar_sampler.var).item()


# function in original tools.py
# see https://github.com/iamalexkorotin/Wasserstein2Benchmark/blob/main/src/tools.py

from torch import nn
from .resnet2 import ResNet_G
from .icnn import View
from tqdm import tqdm
import torch.nn.functional as F

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)   

def load_resnet_G(cpkt_path, device='cuda'):
    resnet = nn.Sequential(
        ResNet_G(128, 64, nfilter=64, nfilter_max=512, res_ratio=0.1),
        View(64*64*3)
    )
    resnet[0].load_state_dict(torch.load(cpkt_path))
    resnet = resnet.to(device)
    freeze(resnet)
    gc.collect(); torch.cuda.empty_cache()
    return resnet

def train_identity_map(D, H, sampler, batch_size=1024, max_iter=5000, lr=1e-3, tol=1e-3, blow=3, verbose=False):
    "Trains potential D to satisfy D.push(x)=x by using MSE loss w.r.t. sampler's distribution"
    unfreeze(H)
    opt = torch.optim.Adam(H.parameters(), lr=lr, weight_decay=1e-10)
    if verbose:
        print('Training the potentials to satisfy push(x)=x')
    for iteration in tqdm_notebook(range(max_iter)) if verbose else range(max_iter):
        X = sampler.sample(batch_size)
        with torch.no_grad():
            X *= blow
        X.requires_grad_(True)

        # W = H.forward(cond_id=0)
        W = H.forward()
        loss = F.mse_loss(D.push(X, W), X.detach())
        loss.backward()
        opt.step(); opt.zero_grad()
        
        if loss.item() < tol:
            break
            
    loss = loss.item()
    gc.collect(); torch.cuda.empty_cache()
    return loss

def train_gaussians_identity_map(D, H, distributions, batch_size=1024, max_iter=5000, lr=1e-3, tol=1e-3, blow=3, verbose=False):
    "Trains potential D to satisfy D.push(x)=x by using MSE loss w.r.t. sampler's distribution"
    unfreeze(H)
    opt = torch.optim.Adam(H.parameters(), lr=lr, weight_decay=1e-10)
    num_distributions = len(distributions)
    if verbose:
        print('Training the potentials to satisfy push(x)=x')
    for iteration in tqdm_notebook(range(max_iter)) if verbose else range(max_iter):
        mean, cov = distributions[iteration % num_distributions]
        X = torch.tensor(np.random.multivariate_normal(mean, cov, batch_size), dtype=torch.float32).cuda()
        with torch.no_grad():
            X *= blow
        X.requires_grad_(True)

        W = H.forward(cond_id=0)
        loss = F.mse_loss(D.push(X, W), X.detach())
        loss.backward()
        opt.step(); opt.zero_grad()
        
        if loss.item() < tol:
            break
            
    loss = loss.item()
    gc.collect(); torch.cuda.empty_cache()
    return loss

def train_multi_gaussians_identity_map(D, H, benchmark, batch_size=1024, max_iter=5000, lr=1e-3, tol=1e-3, blow=3, verbose=False):
    "Trains potential D to satisfy D.push(x)=x by using MSE loss w.r.t. sampler's distribution"
    unfreeze(H)
    opt = torch.optim.Adam(H.parameters(), lr=lr, weight_decay=1e-10)
    W0 = None
    if verbose:
        print('Training the potentials to satisfy push(x)=x')
    for iteration in tqdm_notebook(range(max_iter)) if verbose else range(max_iter):
        X, X_pca, X_ids = benchmark.sample_train_input()
        with torch.no_grad():
            X *= blow
        X.requires_grad_(True)

        W = H.forward(cond_id=0)
        W0 = W
        loss = F.mse_loss(D.push(X, W), X.detach())
        loss.backward()
        opt.step(); opt.zero_grad()
        
        if loss.item() < tol:
            break
            
    loss = loss.item()
    gc.collect(); torch.cuda.empty_cache()
    return loss, W0

def train_multi_gaussians_identity_map(D, H, benchmark, batch_size=1024, max_iter=5000, lr=1e-3, tol=1e-3, blow=3, verbose=False):
    "Trains potential D to satisfy D.push(x)=x by using MSE loss w.r.t. sampler's distribution"
    unfreeze(H)
    opt = torch.optim.Adam(H.parameters(), lr=lr, weight_decay=1e-10)
    W0 = None
    if verbose:
        print('Training the potentials to satisfy push(x)=x')
    for iteration in tqdm_notebook(range(max_iter)) if verbose else range(max_iter):
        X, X_pca, X_ids = benchmark.sample_train_input()
        with torch.no_grad():
            X *= blow
        X.requires_grad_(True)

        W = H.forward(cond_id=0)
        W0 = W
        loss = F.mse_loss(D.push(X, W), X.detach())
        loss.backward()
        opt.step(); opt.zero_grad()
        
        if loss.item() < tol:
            break
            
    loss = loss.item()
    gc.collect(); torch.cuda.empty_cache()
    return loss, W0

def train_multi_gaussians_identity_map_with_emb(D, H, embedding_x, benchmark, batch_size=1024, max_iter=5000, lr=1e-3, tol=1e-3, blow=3, verbose=False, emb_method='transoformer', W_init = None):
    "Trains potential D to satisfy D.push(x)=x by using MSE loss w.r.t. sampler's distribution"
    unfreeze(H)
    opt = torch.optim.Adam(H.parameters(), lr=lr, weight_decay=1e-10)
    W0 = None
    if W_init == None:
        opt = torch.optim.Adam(H.parameters(), lr=lr, weight_decay=1e-10)
        if verbose:
            print('Training the potentials to satisfy push(x)=x')
        for iteration in tqdm(range(max_iter)) if verbose else range(max_iter):
            # X, X_pca, X_ids = benchmark.sample_train_input()
            X, X_pca, X_ids = benchmark.sample_train_input()
            flag = False
            for x in X:   
                with torch.no_grad():
                    x *= blow
                x.requires_grad_(True)
                emb_x = embedding_x(x=x, method = emb_method)
                # W = H.forward(cond_input=emb_x)
                W = H.forward(embedding=emb_x)
                W0 = W
                loss = F.mse_loss(D.push(x, W), x.detach())
                loss.backward()
                opt.step(); opt.zero_grad()
                
                if loss.item() < tol:
                    flag = True
                    break
            if flag:
                break
    else:
        opt = torch.optim.Adam(H.parameters(), lr=1e-3, weight_decay=1e-10)
        W = W_init
        if verbose:
            print('Training the delta potentials to satisfy push(x)=x')
        for iteration in tqdm(range(max_iter)) if verbose else range(max_iter):
            # X, X_pca, X_ids = benchmark.sample_train_input()
            X, X_pca, X_ids = benchmark.sample_train_input(pretrain_batch = batch_size)
            flag = False
            for x in X:   
                with torch.no_grad():
                    x *= blow
                x.requires_grad_(True)
                emb_x = embedding_x(x=x, method = emb_method)
                # delta_W_init = H.forward(cond_input=emb_x)
                # with torch.no_grad():
                #     for i in range(len(W)):
                #         W[i] = W[i] + 1e-6 * nn.Tanh()(delta_W_init[i])
                # W0 = delta_W_init
                # W = H.forward(cond_input=emb_x)
                W = H.forward(embedding=emb_x)
                W0 = W
                loss = F.mse_loss(D.push(x, W), x.detach())
                loss.backward()
                opt.step(); opt.zero_grad()
                
                if loss.item() < tol:
                    flag = True
                    break
            if flag:
                break
            
    loss = loss.item()
    gc.collect(); torch.cuda.empty_cache()
    return loss, W0

def train_multi_gaussians_pairs_identity_map_with_emb(D, H, embedding_x, benchmark, batch_size=1024, max_iter=5000, lr=1e-3, tol=1e-3, blow=3, verbose=False, emb_method='transoformer', W_init = None):
    "Trains potential D to satisfy D.push(x)=x by using MSE loss w.r.t. sampler's distribution"
    unfreeze(H)
    opt = torch.optim.Adam(H.parameters(), lr=lr, weight_decay=1e-10)
    W0 = None
    if W_init == None:
        opt = torch.optim.Adam(H.parameters(), lr=lr, weight_decay=1e-10)
        if verbose:
            print('Training the potentials to satisfy push(x)=x')
        for iteration in tqdm(range(max_iter)) if verbose else range(max_iter):
            # X, X_pca, X_ids = benchmark.sample_train_input()
            X, X_pca, X_ids ,_ ,_ ,_  = benchmark.sample_train()
            flag = False
            for x in X:   
                with torch.no_grad():
                    x *= blow
                x.requires_grad_(True)
                emb_x = embedding_x(x=x, method = emb_method)
                W = H.forward(cond_input=emb_x)
                W0 = W
                loss = F.mse_loss(D.push(x, W), x.detach())
                loss.backward()
                opt.step(); opt.zero_grad()
                
                if loss.item() < tol:
                    flag = True
                    break
            if flag:
                break
    else:
        opt = torch.optim.Adam(H.parameters(), lr=1e-3, weight_decay=1e-10)
        W = W_init
        if verbose:
            print('Training the delta potentials to satisfy push(x)=x')
        for iteration in tqdm(range(max_iter)) if verbose else range(max_iter):
            # X, X_pca, X_ids = benchmark.sample_train_input()
            X, X_pca, X_ids ,_ ,_ ,_ = benchmark.sample_train()
            flag = False
            for x in X:   
                with torch.no_grad():
                    x *= blow
                x.requires_grad_(True)
                emb_x = embedding_x(x=x, method = emb_method)
                delta_W_init = H.forward(cond_input=emb_x)
                with torch.no_grad():
                    for i in range(len(W)):
                        W[i] = W[i] + 1e-6 * nn.Tanh()(delta_W_init[i])
                W0 = delta_W_init
                loss = F.mse_loss(D.push(x, W), x.detach())
                loss.backward()
                opt.step(); opt.zero_grad()
                
                if loss.item() < tol:
                    flag = True
                    break
            if flag:
                break
            
    loss = loss.item()
    gc.collect(); torch.cuda.empty_cache()
    return loss, W0