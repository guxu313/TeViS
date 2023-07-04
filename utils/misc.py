import os
import random
import numpy as np
import torch
import einops

def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def vector_gather(vectors, indices):
    """
        Gathers (batched) vectors according to indices.
        Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
        Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out