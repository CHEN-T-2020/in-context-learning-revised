import math
import numpy as np
import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):  # use kwargs for optional arguments
    names_to_classes = {
        "gaussian": GaussianSampler,
        "skew": IllConditionedSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]

        # call the constructor of the class, for example GaussianSampler(n_dims, **kwargs)
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class IllConditionedSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, kappa=10):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        for j in range(b_size):
            if n_dims_truncated is None:
                n_dims_truncated = self.n_dims
            w = xs_b[j, :, :n_dims_truncated]
            U, S, V = np.linalg.svd(w, full_matrices=False)
            min_sv = S.min()
            desired_max_sv = min_sv * kappa
            if S[S != 0].max() < desired_max_sv:
                S[np.argmax(np.ma.masked_where(S == 0, S))] = desired_max_sv
            else:
                S[S > desired_max_sv] = desired_max_sv

            # print(f"[before] condition number: {torch.linalg.cond(xs_b[j, :, :n_dims_truncated].T @ xs_b[j, :, :n_dims_truncated])}")
            xs_b[j, :, :n_dims_truncated] = torch.from_numpy(
                np.matmul(np.matmul(U, np.diag(S)), V)
            ).float()

            # print(f"shape: {(xs_b[j, :, :n_dims_truncated].T @ xs_b[j, :, :n_dims_truncated]).shape}")
            # print(f"[after] condition number: {torch.linalg.cond(xs_b[j, :, :n_dims_truncated].T @ xs_b[j, :, :n_dims_truncated])}")
        return xs_b
