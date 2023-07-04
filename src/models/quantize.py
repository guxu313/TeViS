import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class VectorQuantizeReduceSimple(nn.Module):
    def __init__(self, n_e, e_dim, beta, 
                 legacy=True, e_dim_reduce=32):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.e_dim_reduce = e_dim_reduce
        self.beta = beta
        self.legacy = legacy

        self.code = nn.parameter.Parameter(torch.randn(self.n_e, e_dim_reduce))
        self.code.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.latent_proj_pre = nn.Linear(e_dim, e_dim_reduce,)
        self.latent_proj_post = nn.Linear(e_dim_reduce, e_dim,)

    def l2norm_code(self):
        return F.normalize(self.code, dim=self.code.dim()-1) 

    def forward(self, z,):
        z_reduce = self.latent_proj_pre(z)
        z_reduce_flattened = z_reduce.view(-1, self.e_dim_reduce)
        z_reduce_flattened = F.normalize(z_reduce_flattened, dim=z_reduce_flattened.dim()-1) 
        l2norm_reduce_code = self.l2norm_code()

        d = torch.sum(z_reduce_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(l2norm_reduce_code ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_reduce_flattened, rearrange(l2norm_reduce_code, 'n d -> d n'))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q_reduce = F.embedding(min_encoding_indices, l2norm_reduce_code).view(z_reduce.shape)
        

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q_reduce.detach()-z_reduce)**2) + \
                   torch.mean((z_q_reduce - z_reduce.detach()) ** 2)
        else:
            loss = torch.mean((z_q_reduce.detach()-z_reduce)**2) + self.beta * \
                   torch.mean((z_q_reduce - z_reduce.detach()) ** 2)

        # preserve gradients
        z_q_reduce = z_reduce + (z_q_reduce - z_reduce).detach()
        z_q = self.latent_proj_post(z_q_reduce)

        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # get quantized latent vectors
        l2norm_code = self.l2norm_code()
        z_q_reduce = F.embedding(indices, l2norm_code)
        z_q = self.latent_proj_post(z_q_reduce)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q    


class VectorQuantizeSimple(nn.Module):
    def __init__(self, n_e, e_dim, beta, 
                 legacy=True, ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.code = nn.parameter.Parameter(torch.randn(self.n_e, e_dim))
        self.code.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def l2norm_code(self):
        return F.normalize(self.code, dim=self.code.dim()-1) 

    def forward(self, z,):
        z_flattened = z.view(-1, self.e_dim)
        z_flattened = F.normalize(z_flattened, dim=z_flattened.dim()-1) 
        l2norm_code = self.l2norm_code()

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(l2norm_code ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(l2norm_code, 'n d -> d n'))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, l2norm_code).view(z.shape)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # get quantized latent vectors
        l2norm_code = self.l2norm_code()
        z_q = F.embedding(indices, l2norm_code)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q    