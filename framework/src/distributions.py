import torch
import numpy as np
from scipy.linalg import sqrtm
import sklearn.datasets
import random

def symmetrize(X):
    return np.real((X + X.T) / 2)

class Sampler:
    def __init__(
        self, device='cuda',
        requires_grad=False,
    ):
        self.device = device
        self.requires_grad = requires_grad
    
    def sample(self, batch_size=5):
        pass
    
    def _estimate_mean(self, num_samples=100000):
        batch = self.sample(num_samples).cpu().detach().numpy()
        self.mean = batch.mean(axis=0).astype(np.float32)
    
    def _estimate_cov(self, num_samples=100000):
        batch = self.sample(num_samples).cpu().detach().numpy()
        self.cov = np.cov(batch.T).astype(np.float32)
        self.var = np.trace(self.cov)
    
    # function in original distributions.py Sampler
    # see https://github.com/iamalexkorotin/Wasserstein2Benchmark/blob/main/src/distributions.py line 21.
    def _estimate_moments(self, size=2**14, mean=True, var=True, cov=True):
        if (not mean) and (not var) and (not cov):
            return
        
        sample = self.sample(size).cpu().detach().numpy().astype(np.float32)
        if mean:
            self.mean = sample.mean(axis=0)
        if var:
            self.var = sample.var(axis=0).sum()
        if cov:
            self.cov = np.cov(sample.T).astype(np.float32)
    
class StandardNormalSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda',
        requires_grad=False
    ):
        super(StandardNormalSampler, self).__init__(device, requires_grad)
        self.dim = dim
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.cov = np.eye(self.dim, dtype=np.float32)
        self.var = self.dim
        
    def sample(self, batch_size=10):
        return torch.randn(
            batch_size, self.dim,
            device=self.device,
            requires_grad=self.requires_grad
        )
    
class NormalSampler(Sampler):
    def __init__(
        self, mean, cov=None, weight=None, device='cuda',
        requires_grad=False
    ):
        super(NormalSampler, self).__init__(device=device, requires_grad=requires_grad)
        self.mean = np.array(mean, dtype=np.float32)
        self.dim = self.mean.shape[0]
        
        if weight is not None:
            weight = np.array(weight, dtype=np.float32)
        
        if cov is not None:
            self.cov = np.array(cov, dtype=np.float32)
        elif weight is not None:
            self.cov = weight @ weight.T
        else:
            self.cov = np.eye(self.dim, dtype=np.float32)
            
        if weight is None:
            weight = symmetrize(sqrtm(self.cov))
            
        self.var = np.trace(self.cov)
        
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        self.bias = torch.tensor(self.mean, device=self.device, dtype=torch.float32)

    def sample(self, batch_size=4):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        batch.requires_grad_(self.requires_grad)
        return batch
    
class CubeUniformSampler(Sampler):
    def __init__(
        self, dim=1, centered=False, normalized=False, device='cuda',
        requires_grad=False
    ):
        super(CubeUniformSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = dim
        self.centered = centered
        self.normalized = normalized
        self.var = self.dim if self.normalized else (self.dim / 12)
        self.cov = np.eye(self.dim, dtype=np.float32) if self.normalized else np.eye(self.dim, dtype=np.float32) / 12
        self.mean = np.zeros(self.dim, dtype=np.float32) if self.centered else .5 * np.ones(self.dim, dtype=np.float32)
        
        self.bias = torch.tensor(self.mean, device=self.device)
        
    def sample(self, batch_size=10):
        return np.sqrt(self.var) * (torch.rand(
            batch_size, self.dim, device=self.device,
            requires_grad=self.requires_grad
        ) - .5) / np.sqrt(self.dim / 12)  + self.bias

class BoxUniformSampler(Sampler):
    # A uniform box with axes components and the range on each
    # axis i is [a_min[i], a_max[i]].
    def __init__(
        self, components, a_min, a_max, estimate_size=100000,
        device='cuda', requires_grad=False
    ):
        super(BoxUniformSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = components.shape[1]
        self.components = torch.from_numpy(components).float().to(device=device)
        self.a_min = torch.from_numpy(a_min).float().to(device=device)
        self.a_max = torch.from_numpy(a_max).float().to(device=device)
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
    
    def sample(self, batch_size):
        with torch.no_grad():
            batch = torch.rand(
                batch_size, self.dim,
                device=self.device
            )
            batch = (torch.unsqueeze(self.a_min, 0) + 
                     batch * torch.unsqueeze(self.a_max - self.a_min, 0))
            batch = torch.matmul(batch, self.components)
            return torch.tensor(
                batch, device=self.device,
                requires_grad=self.requires_grad
            )
        
class EmpiricalSampler(Sampler):
    def __init__(
        self, data, estimate_size=100000,
        device='cuda', requires_grad=False
    ):
        super(EmpiricalSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        # data is a np array NxD
        self.dim = data.shape[1]
        self.num_points = data.shape[0]
        self.data = torch.from_numpy(data).float().to(device=device)
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size):
        inds = torch.randperm(self.num_points)
        if batch_size <= self.num_points:
            inds = inds[:batch_size]
        else:
            additional_inds = torch.randint(0, self.num_points, (batch_size - self.num_points))
            inds = torch.cat([inds, additional_inds], dim=0)
        inds_repeated = torch.unsqueeze(inds, 1).repeat(1, self.dim)
        batch = torch.gather(self.data, 0, inds_repeated.to(device=self.device))
        return torch.tensor(
                batch, device=self.device,
                requires_grad=self.requires_grad
        )
    
class TensorDatasetSampler(Sampler):
    def __init__(
        self, dataset, transform=None, storage='cpu', storage_dtype=torch.float,
        device='cuda', requires_grad=False, estimate_size=100000,
    ):
        super(TensorDatasetSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.storage = storage
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda t: t
            
        self.storage_dtype = storage_dtype
        
        self.dataset = torch.tensor(
            dataset, device=storage, dtype=storage_dtype, requires_grad=False
        )  
        
        self.dim = self.sample(1).shape[1]
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size) 
        
    def sample(self, batch_size=10):
        if batch_size:
            ind = random.choices(range(len(self.dataset)), k=batch_size)
        else:
            ind = range(len(self.dataset))
            
        with torch.no_grad():
            batch = self.transform(torch.tensor(
                self.dataset[ind], device=self.device,
                dtype=torch.float32, requires_grad=False
            ))
        if self.requires_grad:
            batch.requires_grad_(True)
        return batch
        
    
class BallCrustUniformSampler(Sampler):
    def __init__(
        self, dim=2, r_min=0.8, r_max=1.2, estimate_size=100000,
        device='cuda', requires_grad=False
    ):
        super(BallCrustUniformSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = dim
        assert r_min >= 0
        assert r_min < r_max
        self.r_min, self.r_max = r_min, r_max
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(
                batch_size, self.dim,
                device=self.device
            )
            batch /= torch.norm(batch, dim=1)[:, None]
            ratio = (1 - (self.r_max - self.r_min) / self.r_max) ** self.dim
            r = (torch.rand(
                batch_size, device=self.device
            ) * (1 - ratio) + ratio) ** (1. / self.dim)
        
        return torch.tensor(
            (batch.transpose(0, 1) * r * self.r_max).transpose(0, 1),
            device=self.device,
            requires_grad=self.requires_grad
        )
    
class MixN2GaussiansSampler(Sampler):
    def __init__(self, n=5, std=1, step=9, device='cuda', estimate_size=100000,
        requires_grad=False
    ):
        super(MixN2GaussiansSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        
        self.dim = 2
        self.std, self.step = std, step
        
        self.n = n
        
        grid_1d = np.linspace(-(n-1) / 2., (n-1) / 2., n)
        xx, yy = np.meshgrid(grid_1d, grid_1d)
        centers = np.stack([xx, yy]).reshape(2, -1).T
        self.centers = torch.tensor(
            centers,
            device=self.device,
            dtype=torch.float32
        )
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        batch = torch.randn(
            batch_size, self.dim,
            device=self.device
        )
        indices = random.choices(range(len(self.centers)), k=batch_size)
        batch *= self.std
        batch += self.step * self.centers[indices, :]
        return torch.tensor(
            batch, device=self.device,
            requires_grad=self.requires_grad
        )
    
class CubeCrustUniformSampler(Sampler):
    def __init__(
        self, dim=2, r_min=0.8, r_max=1.2, estimate_size=100000, device='cuda',
        requires_grad=False
    ):
        super(CubeCrustUniformSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = dim
        assert r_min >= 0
        assert r_min < r_max
        self.r_min, self.r_max = r_min, r_max
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = 2 * torch.rand(
                batch_size, self.dim,
                device=self.device
            ) - 1
            axes = torch.randint(0, self.dim, size=(batch_size, 1), device=self.device)
            batch.scatter_(
                1, axes, 
                2 * ((batch.gather(1, axes) > 0)).type(torch.float32) - 1
            )
            
            ratio = (1 - (self.r_max - self.r_min) / self.r_max) ** self.dim
            r = (torch.rand(
                batch_size, device=self.device
            ) * (1 - ratio) + ratio) ** (1. / self.dim)
        
        return torch.tensor(
            (batch.transpose(0, 1) * self.r_max * r).transpose(0, 1),
            device=self.device, 
            requires_grad=self.requires_grad
        )
    
class SwissRollSampler(Sampler):
    def __init__(
        self, estimate_size=100000, device='cuda', requires_grad=False
    ):
        super(SwissRollSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = 2
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        batch = sklearn.datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype(np.float32)[:, [0, 2]] / 7.5
        return torch.tensor(
            batch, device=self.device,
            requires_grad=self.requires_grad
        )
    
class Mix8GaussiansSampler(Sampler):
    def __init__(
        self, with_central=False, std=1, r=12,
        estimate_size=100000, 
        device='cuda', requires_grad=False
    ):
        super(Mix8GaussiansSampler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        self.dim = 2
        self.std, self.r = std, r
        
        self.with_central = with_central
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(
            centers, device=self.device
        )
        
        self._estimate_mean(estimate_size)
        self._estimate_cov(estimate_size)
        
    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(
                batch_size, self.dim,
                device=self.device
            )
            indices = random.choices(range(len(self.centers)), k=batch_size)
            batch *= self.std
            batch += self.r * self.centers[indices, :]
        if self.requires_grad:
            batch.requires_grad_(True)
        return batch

# in original class Transformer, it inherits from class Sampler
# see https://github.com/iamalexkorotin/Wasserstein2Benchmark/blob/main/src/distributions.py line 191.

# class Transformer(object):
#     def __init__(
#         self, device='cuda',
#         requires_grad=False
#     ):
#         self.device = device
#         self.requires_grad = requires_grad

class Transformer(Sampler):
    def __init__(
        self, device='cuda',
        requires_grad=False
    ):
        self.device = device
        self.requires_grad = requires_grad


        
class LinearTransformer(Transformer):
    def __init__(
        self, weight, bias=None, base_sampler=None,
        device='cuda',
        requires_grad=False
    ):
        super(LinearTransformer, self).__init__(
            device=device,
            requires_grad=requires_grad
        )
        
        self.fitted = False
        self.dim = weight.shape[0]
        self.weight = torch.tensor(weight, device=device, dtype=torch.float32, requires_grad=False)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=torch.float32, requires_grad=False)
        else:
            self.bias = torch.zeros(self.dim, device=device, dtype=torch.float32, requires_grad=False)
        
        
        if base_sampler is not None:
            self.fit(base_sampler)

        
    def fit(self, base_sampler):
        self.base_sampler = base_sampler
        weight, bias = self.weight.cpu().numpy(), self.bias.cpu().numpy()
        
        self.mean = weight @ self.base_sampler.mean + bias
        self.cov = weight @ self.base_sampler.cov @ weight.T
        self.var = np.trace(self.cov)
        
        self.fitted = True
        return self
        
    def sample(self, batch_size=4):
        assert self.fitted == True
        
        batch = torch.tensor(
            self.base_sampler.sample(batch_size),
            device=self.device, requires_grad=False
        )
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        batch = batch.detach()
        batch.requires_grad_(self.requires_grad)
        return batch
    
class StandardNormalScaler(Transformer):
    def __init__(
        self, base_sampler=None, device='cuda', requires_grad=False
    ):
        super(StandardNormalScaler, self).__init__(
            device=device, requires_grad=requires_grad
        )
        if base_sampler is not None:
            self.fit(base_sampler)
        
    def fit(self, base_sampler, batch_size=1000):
        self.base_sampler = base_sampler
        self.dim = self.base_sampler.dim
        
        self.bias = torch.tensor(
            self.base_sampler.mean, device=self.device, dtype=torch.float32
        )
        
        weight = symmetrize(np.linalg.inv(sqrtm(self.base_sampler.cov)))
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.cov = weight @ self.base_sampler.cov @ weight.T
        self.var = np.trace(self.cov)
        
        return self
        
    def sample(self, batch_size=10):
        batch = torch.tensor(
            self.base_sampler.sample(batch_size),
            device=self.device, requires_grad=False
        )
        with torch.no_grad():
            batch -= self.bias
            batch @= self.weight
        if self.requires_grad:
            batch.requires_grad_(True)
        return batch
    
# classes in original distributions.py
# see https://github.com/iamalexkorotin/Wasserstein2Benchmark/blob/main/src/distributions.py

from .potentials import BasePotential

    
class RandomGaussianMixSampler(Sampler):
    def __init__(
        self, dim=2, num=10, dist=1, std=0.4,
        standardized=True, estimate_size=2**14,
        batch_size=1024, device='cuda'
    ):
        super(RandomGaussianMixSampler, self).__init__(device=device)
        self.dim = dim
        self.num = num
        self.dist = dist
        self.std = std
        self.batch_size = batch_size
        
        centers = np.zeros((self.num, self.dim), dtype=np.float32)
        for d in range(self.dim):
            idx = np.random.choice(list(range(self.num)), self.num, replace=False)
            centers[:, d] += self.dist * idx
        centers -= self.dist * (self.num - 1) / 2
        
        maps = np.random.normal(size=(self.num, self.dim, self.dim)).astype(np.float32)
        maps /= np.sqrt((maps ** 2).sum(axis=2, keepdims=True))
        
        if standardized:
            mult = np.sqrt((centers ** 2).sum(axis=1).mean() + self.dim * self.std ** 2) / np.sqrt(self.dim)
            centers /= mult
            maps /= mult
        
        self.centers = torch.tensor(centers, device=self.device, dtype=torch.float32)  
        self.maps = torch.tensor(maps, device=self.device, dtype=torch.float32)
        
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self._estimate_moments(mean=False) # This can be also be done analytically
        
    def sample(self, size=10):          
        if size <= self.batch_size:
            idx = np.random.randint(0, self.num, size=size)
            sample = torch.randn(size, self.dim, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                sample = torch.matmul(self.maps[idx], sample[:, :, None])[:, :, 0] * self.std
                sample += self.centers[idx]
            return sample
        
        sample = torch.zeros(size, self.dim, dtype=torch.float32, device=self.device)
        for i in range(0, size, self.batch_size):
            batch = self.sample(min(i + self.batch_size, size) - i)
            with torch.no_grad():
                sample[i:i+self.batch_size] = batch
            torch.cuda.empty_cache()
        return sample



       
class PotentialTransformer(Transformer):
    def __init__(
        self, potential,
        device='cuda'
    ):
        super(PotentialTransformer, self).__init__(
            device=device
        )
        
        self.fitted = False
        
        assert issubclass(type(potential), BasePotential)
        self.potential = potential.to(self.device)
        self.dim = self.potential.dim
        
    def fit(self, base_sampler, estimate_size=2**14, estimate_cov=True):
        assert base_sampler.device == self.device
        
        self.base_sampler = base_sampler
        self.fitted = True
        
        self._estimate_moments(estimate_size, True, True, estimate_cov)
        return self
        
    def sample(self, size=4):
        assert self.fitted == True
        sample = self.base_sampler.sample(size)
        sample.requires_grad_(True)
        return self.potential.push_nograd(sample)
    
class PushforwardTransformer(Transformer):
    def __init__(
        self, pushforward,
        batch_size=128,
        device='cuda'
    ):
        super(PushforwardTransformer, self).__init__(
            device=device
        )
        
        self.fitted = False
        self.batch_size = batch_size
        self.pushforward = pushforward

    def fit(self, base_sampler, estimate_size=2**14, estimate_cov=True):
        assert base_sampler.device == self.device
        
        self.base_sampler = base_sampler
        self.fitted = True
        
        self._estimate_moments(estimate_size, True, True, estimate_cov)
        self.dim = len(self.mean)
        return self
        
    def sample(self, size=4):
        assert self.fitted == True
        
        if size <= self.batch_size:
            sample = self.base_sampler.sample(size)
            with torch.no_grad():
                sample = self.pushforward(sample)
            return sample
        
        sample = torch.zeros(size, self.sample(1).shape[1], dtype=torch.float32, device=self.device)
        for i in range(0, size, self.batch_size):
            batch = self.sample(min(i + self.batch_size, size) - i)
            with torch.no_grad():
                sample.data[i:i+self.batch_size] = batch.data
            torch.cuda.empty_cache()
        return sample
    
    
    
class NormalNoiseTransformer(Transformer):
    def __init__(
        self, std=0.01,
        device='cuda'
    ):
        super(NormalNoiseTransformer, self).__init__(
            device=device
        )
        self.std = std
        
    def fit(self, base_sampler):
        self.base_sampler = base_sampler
        self.dim = base_sampler.dim
        self.mean = base_sampler.mean
        self.var = base_sampler.var + self.dim * (self.std ** 2)
        if hasattr(base_sampler, 'cov'):
            self.cov = base_sampler.cov + np.eye(self.dim, dtype=np.float32) * (self.std ** 2)
        return self
        
    def sample(self, batch_size=4):
        batch = self.base_sampler.sample(batch_size)
        with torch.no_grad():
            batch = batch + self.std * torch.randn_like(batch)
        return batch