import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import math
from torch import Tensor
from transformers.modeling_utils import Conv1D

class Attention(nn.Module):
    def __init__(self, cfg, n_embd, is_cross_attention=False):
        super().__init__()

        self.cfg = cfg

        n_head=8
        attn_pdrop=0.5
        resid_pdrop=0.5
        n_state = n_embd  # in Attention: n_state=768 (nx=n_embd)
        n_ctx = 1024
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        # assert n_state % config.n_head == 0

        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = n_head
        self.split_size = n_state
        self.scale = True

        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(n_state * 2, n_state)
            self.q_attn = Conv1D(n_state , n_state)
        else:
            self.c_attn = Conv1D(n_state * 3, n_state)
            
        self.c_proj = Conv1D(n_state, n_state)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        if not self.is_cross_attention:
            if self.cfg.model_type == 'cross' or self.cfg.model_type == 'img_only':
                mask = self.bias[:, :, ns - nd : ns, :ns]               
            elif self.cfg.model_type == 'prefix':  
                text_mask = torch.cat((torch.ones(77,77),torch.zeros(ns-77,77)),dim=0).transpose(0,1)
                mask = torch.cat((text_mask.unsqueeze(0).unsqueeze(1).cuda(),self.bias[:, :, ns - nd +77 : ns, :ns]),dim=2)    
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        outputs = [torch.matmul(w, v)]

        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape) 
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, attention_mask, encoder_hidden = None, encoder_attention_mask = None):
        # cross attn
        if encoder_hidden is not None:
            query = self.q_attn(x)
            encoder_hidden = self.c_attn(encoder_hidden)
            key, value = encoder_hidden.split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask

        # masked self attn
        else:
            x = self.c_attn(x)
            query, key, value = x.split(self.split_size, dim=2)

        query = self.split_heads(query)         # B,head,L,C
        key = self.split_heads(key, k=True)     # B,head,C,L
        value = self.split_heads(value)         # B,head,L,C

        attn_outputs = self._attn(query, key, value, attention_mask)

        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return [a]

class MLP(nn.Module):
    def __init__(self, n_embd): 
        super().__init__()
        nx = n_embd
        n_state = n_embd
        resid_pdrop=0.5

        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(nn.Module):
    def __init__(self, cfg, n_embd, is_cross_attention=False):
        super().__init__()
        nx = n_embd

        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = Attention(cfg, n_embd, is_cross_attention)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, attention_mask, encoder_hidden = None, encoder_attention_mask = None):
        x = self.ln_1(x)
        output_attn = self.attn(x,attention_mask,encoder_hidden, encoder_attention_mask)
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_1(x))

        x = x + m
        x = self.ln_2(x)

        return [x]