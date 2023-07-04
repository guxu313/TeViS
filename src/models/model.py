import torch
import torch.nn.functional as F
from torch import nn
from random import choice

import numpy as np
import random
from src.utils.logger import LOGGER
from src.utils.dist import SyncFunction, concat_all_gather
from src.utils.misc import vector_gather

from src.models import clip
from src.models.attn_block import Block
from src.models.quantize import VectorQuantizeReduceSimple, VectorQuantizeSimple

import operator
import functools
import os
import json
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import math
import time
from torch import Tensor
from transformers.modeling_utils import Conv1D
import scipy.stats as stats
import sys


class MovieGPTDecoderPrefix(nn.Module):
    def __init__(self, args, cfg):
        super().__init__()

        self.cfg = cfg

        n_layer = self.cfg.n_layer
        
        quantize = self.cfg.quantize

        self.clip_model, _ = clip.load(os.path.join(args.blob_mount_dir, cfg.MODEL.clip_path), device='cpu')
        if quantize:
            self.clip_visual_quantize = VectorQuantizeSimple(n_e=cfg.QUANTIZE.embed_n, 
                                                            e_dim=cfg.QUANTIZE.embed_dim_reduce, 
                                                            beta=cfg.QUANTIZE.beta,
                                                            )
            self.clip_text_quantize_fc = nn.Linear(cfg.MODEL.emb_size, cfg.QUANTIZE.embed_dim_reduce,)
            self.clip_visual_quantize_fc_pre = nn.Linear(cfg.MODEL.emb_size, cfg.QUANTIZE.embed_dim_reduce,)

            self.sos_embd = nn.Parameter(0.02*torch.randn(cfg.QUANTIZE.embed_dim_reduce))
            self.eos_embd = nn.Parameter(0.02*torch.randn(cfg.QUANTIZE.embed_dim_reduce))
            self.pad_embd = nn.Parameter(0.02*torch.randn(cfg.QUANTIZE.embed_dim_reduce))

            self.wpe = nn.Embedding(128, cfg.QUANTIZE.embed_dim_reduce)            # positioon embedding matrix
            self.emb_norm = nn.LayerNorm(cfg.QUANTIZE.embed_dim_reduce)

            self.wpe_text = nn.Embedding(128, cfg.QUANTIZE.embed_dim_reduce)       # positioon embedding matrix for text

            self.cross_blocks = None
            if self.cfg.model_type == 'cross':
                self.cross_blocks = nn.ModuleList([Block(cfg, cfg.QUANTIZE.embed_dim_reduce, is_cross_attention=True) for _ in range(n_layer)])

            self.self_blocks = nn.ModuleList([Block(cfg, cfg.QUANTIZE.embed_dim_reduce, is_cross_attention=False) for _ in range(n_layer)])
        else:
            self.sos_embd = nn.Parameter(0.02*torch.randn(cfg.MODEL.emb_size))
            self.eos_embd = nn.Parameter(0.02*torch.randn(cfg.MODEL.emb_size))
            self.pad_embd = nn.Parameter(0.02*torch.randn(cfg.MODEL.emb_size))

            self.wpe = nn.Embedding(128, cfg.MODEL.emb_size)            # positioon embedding matrix
            self.emb_norm = nn.LayerNorm(cfg.MODEL.emb_size)

            self.wpe_text = nn.Embedding(128, cfg.MODEL.emb_size)       # positioon embedding matrix for text

            self.cross_blocks = None
            if self.cfg.model_type == 'cross':
                self.cross_blocks = nn.ModuleList([Block(cfg, cfg.MODEL.emb_size, is_cross_attention=True) for _ in range(n_layer)])

            self.self_blocks = nn.ModuleList([Block(cfg, cfg.MODEL.emb_size, is_cross_attention=False) for _ in range(n_layer)])

        self.res_dir = os.path.join(cfg.TRAINING.save_dir, "./res")
        os.makedirs(self.res_dir, exist_ok=True)


    def extract_feat(self, text, imgs):
        if self.cfg.quantize:
            # Encode text
            text_embds, _ = self.clip_model.encode_text(text, return_seq=True) 
            text_embds_reduce = self.clip_text_quantize_fc(text_embds)
            text_embds_reduce = F.normalize(text_embds_reduce, dim = -1).transpose(0,1)

            # Encode image
            B, L,  _, _, _ = imgs.shape
            img_embds = self.clip_model.encode_image(imgs.flatten(0, 1)) 
            img_embds = img_embds.view(B, L, -1).permute(1, 0, 2) 
            img_embds_reduce = self.clip_visual_quantize_fc_pre(img_embds)
            img_embds_reduce = F.normalize(img_embds_reduce, dim = -1)
            
            # Quantize
            img_embds_reduce_quantize, quantize_loss, info = self.clip_visual_quantize(img_embds_reduce)

        else:
             # Encode text
            text_embds, _ = self.clip_model.encode_text(text, return_seq=True) 
            text_embds = F.normalize(text_embds, dim = -1).transpose(0,1)

            # Encode image
            B, L,  _, _, _ = imgs.shape
            img_embds = self.clip_model.encode_image(imgs.flatten(0, 1)) 
            img_embds = img_embds.view(B, L, -1).permute(1, 0, 2) 
            img_embds = F.normalize(img_embds, dim = -1)

            text_embds_reduce = text_embds
            img_embds_reduce_quantize = img_embds
            quantize_loss = 0
        return text_embds_reduce, img_embds_reduce_quantize, img_embds, quantize_loss


    def cal_loss(self, predictions, img_embds, padding_mask , temp=0.05):
        sim = torch.matmul(predictions.flatten(0,1), img_embds.flatten(0,1).permute(1, 0)) / temp
        if padding_mask is not None:
            label = torch.arange(sim.shape[0], device=sim.device) 
            label=label.view(predictions.shape[0],-1)  
            label=torch.where(padding_mask.bool(),label,-100 * torch.ones_like(padding_mask,dtype=label.dtype))         # 1 for valid label, 0 for fake label i.e. -100
            label=label.flatten(0,1)
        else:
            label = torch.arange(sim.shape[0], device=sim.device)

        loss = F.cross_entropy(sim, label,ignore_index=-100, reduction='mean')
        return loss


    def generate_attn_mask(self, attention_mask):
        attention_mask = attention_mask.view(attention_mask.shape[0], -1)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0

        return attention_mask


    @torch.no_grad()
    def cal_kdll_prefix(self ,gt, input_embds, padding_mask,index_eos, text_len,res_dir):
        gt = gt.permute(1,0,2)
        B,L,C=gt.shape

        # Prompt frame num, default 0, i.e. predict from sos
        prompt_frame_num = self.cfg.prompt_frame_num

        # Init predictions
        all_predictions = input_embds.permute(1,0,2)[:, :(text_len+1+prompt_frame_num),:]  
        retrieved_index = torch.arange(prompt_frame_num+text_len+1,device=all_predictions.device).unsqueeze(0).repeat([B,1])        # [B,1+prompt_frame_num]

        # Mask retrieved and padded images 
        valid_pos = padding_mask.bool() 
        valid_pos[:,:(text_len+1+prompt_frame_num)] = False
        index_eos_w_sos = (index_eos[1], index_eos[0]+text_len+1) 
        valid_pos[index_eos_w_sos] = False

        for t in range(prompt_frame_num+text_len+1,L):
            # Forward
            hidden_states = all_predictions
            for i, self_block in enumerate(self.self_blocks):
                # Self
                outputs = self_block(hidden_states,attention_mask = None, 
                                encoder_hidden = None, encoder_attention_mask =  None)
                hidden_states = outputs[0]

                # Cross
                if self.cross_blocks is not None:
                    cross_block = self.cross_blocks[i]
                    outputs = cross_block(hidden_states,attention_mask=None, 
                                    encoder_hidden = text_embds, encoder_attention_mask =  text_attention_mask)
                    hidden_states = outputs[0]
            
            prediction = hidden_states[:,-1:,:]

            # Retrieval
            sim=torch.bmm(F.normalize(prediction, dim=-1),F.normalize(gt, dim=-1).permute(0,2,1))     
            masked_value = -1000 * torch.ones_like(sim)     
            sim = torch.where(valid_pos.unsqueeze(1), sim, masked_value)    
            rank_list = (-sim).argsort()                                    
            top1_index = rank_list[:,:,:1].squeeze()

            index_chosen=(torch.arange(B,device=top1_index.device),top1_index)
            valid_pos[index_chosen] = False

            retrieved_imgs=gt[index_chosen].unsqueeze(1)
            position_embds = self.wpe(torch.tensor(t,device=retrieved_imgs.device))
            retrieved_imgs = retrieved_imgs + position_embds
            all_predictions = torch.cat([all_predictions, retrieved_imgs],dim=1)
            retrieved_index = torch.cat([retrieved_index, top1_index.unsqueeze(1)],dim=1)
        
        # Cal kdll
        kdlls={"all":[],"2":[],"3_5":[],"6_max":[]}
        res_strs = []
        for i in range(B):
            seq_length=int((padding_mask[i][text_len:].sum()).item())+text_len-1

            gt_index = torch.arange(seq_length)
            pred_index = retrieved_index[i,:seq_length].cpu()
            gt_index=gt_index[text_len+1:]
            pred_index=pred_index[text_len+1:]

            # reduce text_length
            gt_index = gt_index-text_len
            pred_index = pred_index-text_len
            
            res_strs.append(','.join(map(str,pred_index.tolist())))
            kdll, p_value = stats.kendalltau(gt_index, pred_index)
            
            kdlls["all"].append(kdll)
            if seq_length-text_len-1 == 2:
                kdlls["2"].append(kdll)
            elif seq_length-text_len-1 >= 3 and seq_length-text_len-1 <= 5:
                kdlls["3_5"].append(kdll)
            elif seq_length-text_len-1 >= 6 :
                kdlls["6_max"].append(kdll)
        res_fs_num_str = str(len(os.listdir(res_dir))).zfill(5)
        if len(os.listdir(res_dir)) == 0:
            res_fs_num_str = str(int(res_fs_num_str)+1).zfill(5)
        else:
            with open(f"{res_dir}/{res_fs_num_str}.txt", "r") as f:
                if len(f.read().split("\n")) >= 906:
                    res_fs_num_str = str(int(res_fs_num_str)+1).zfill(5)
        with open(f"{res_dir}/{res_fs_num_str}.txt", "a") as f:
            f.write("\n".join(res_strs)+"\n")

        if self.cfg.is_test:
            with open(f"{res_dir}/00001.txt", "r") as f:
                if len(f.read().split("\n")) >= 906:
                    print('Test done!')
                    sys.exit()
        
        return kdlls, retrieved_index


    def forward(self, text, imgs, padding_mask, valid_len, is_train=True):
        text_embds, img_embds, img_embds_raw, quantize_loss = self.extract_feat(text, imgs)
        L, B, C, = img_embds.shape
        
        x_index = [[x for x in range(int(y)+1, L)] for y in valid_len] 
        y_index = [[x for y in range(len(x_index[x]))] for x in range(len(x_index))]   
        index_pad = (
            torch.tensor(functools.reduce(operator.concat, x_index), dtype=int).to(valid_len.device),
            torch.tensor(functools.reduce(operator.concat, y_index), dtype=int).to(valid_len.device)
        )
        img_embds[index_pad] = self.pad_embd


        # eos token
        x_index =  [min(x+1, torch.tensor(L-1).to(img_embds.device) ) for x in valid_len] 
        y_index = list(range(len(x_index)))
        index_eos = (
            torch.tensor((x_index), dtype=int).to(valid_len.device),
            torch.tensor((y_index), dtype=int).to(valid_len.device)
        )
        img_embds[index_eos] = self.eos_embd
        padding_mask[(index_eos[1],index_eos[0])] = 1 


        # sos token
        img_embds = torch.cat([self.sos_embd.repeat(1,B,1),img_embds],dim=0)  
        padding_mask = torch.cat([torch.ones(B,1).to(padding_mask.device),padding_mask],dim=1) 

         # add text 
        l_t, _, _ = text_embds.shape

        # add text_attention_mask
        text_attention_mask = torch.where(text!=0, torch.ones_like(text), torch.zeros_like(text))
        padding_mask = torch.cat([text_attention_mask.to(padding_mask.device),padding_mask],dim=1)  

        L=L+l_t+1    


        # position embed for imgs
        position_ids = torch.arange(L, dtype=torch.long, device=valid_len.device)
        position_embds = self.wpe(position_ids) 
        position_embds = position_embds.unsqueeze(1).repeat(1,B,1) 


        # img & text attention mask
        img_attention_mask = padding_mask 
        img_attention_mask = self.generate_attn_mask(img_attention_mask)
        # text_attention_mask = self.generate_attn_mask(text_attention_mask)

        # input
        hidden_states = torch.cat([text_embds,img_embds],dim=0) + position_embds 
        hidden_states = self.emb_norm(hidden_states) 


        # Cal kdll if not in training
        kdlls={"all":[],"2":[],"3_5":[],"6_max":[]}
        retrieved_index = 0

        if not is_train:
            kdlls, retrieved_index = self.cal_kdll_prefix(gt = torch.cat([text_embds,img_embds],dim=0), 
                                                   input_embds=hidden_states, 
                                                    padding_mask=padding_mask, index_eos=index_eos, 
                                                    text_len = l_t,
                                                    res_dir=self.res_dir)

        # Pass forward, teacher forcing
        hidden_states = hidden_states.permute(1,0,2) 
        text_embds = text_embds.permute(1,0,2).contiguous()
        for i, self_block in enumerate(self.self_blocks):
            # Self-Attn
            outputs = self_block(hidden_states,attention_mask=img_attention_mask, 
                            encoder_hidden = None, encoder_attention_mask =  None)
            hidden_states = outputs[0]

            # Cross-Attn
            # TODO add
            if self.cross_blocks is not None:
                cross_block = self.cross_blocks[i]
                outputs = cross_block(hidden_states,attention_mask=None, 
                                encoder_hidden = text_embds, encoder_attention_mask =  text_attention_mask)
                hidden_states = outputs[0]

        # Loss
        img_embds = img_embds[1:, ] 
        hidden_states=hidden_states.permute(1,0,2) 
        predictions = hidden_states [l_t:-1,] 

        predictions = F.normalize(predictions,dim=-1)
        img_embds = F.normalize(img_embds,dim=-1)

        padding_mask = padding_mask[:,l_t+1: ]  
        padding_mask = padding_mask.permute(1,0)
        loss = self.cal_loss(predictions, img_embds, padding_mask=padding_mask)
        loss += 1.0 * quantize_loss

        # Acc
        predictions = predictions[padding_mask.bool()]
        img_embds = img_embds[padding_mask.bool()]
        label = torch.arange(len(img_embds), device=img_embds.device)

        acc = sum(torch.matmul(predictions, img_embds.permute(1, 0)).max(dim=1).indices == label) / len(predictions)

        kdlls=[torch.tensor(kdll) for kdll in kdlls.values()]

        return loss, acc, kdlls, retrieved_index


        
if __name__ == '__main__':
    from easydict import EasyDict
    import json

    cfg = EasyDict({'MODEL':{'input_size': 512, 
                    'emb_size':512, 
                    'num_encoder_layers': 2,
                    'dim_feedforward': 2048,
                    'clip_path': '/blob_mount_model/project/script_translator/pretrained/ViT-B-32.pt',
                    'drop_prob': 0.1,
                    'nhead': 8,
                    'freeze_all_backbone': False,
                    'freeze_text_backbone': True,
                    },
                    'mlm_probability': 0.15,
                    'prompt_frame_num': 0,
                    'kdll_see_first_frame': True

                    })

    args = EasyDict({'blob_mount_dir': '/blob_mount'})
    model = MovieGPTDecoderCross(args, cfg).cuda()

    valid_len=torch.randint(1,20,(8,))
    padding_mask = torch.zeros(8,20)
    for i,length in enumerate(valid_len):
        padding_mask[i,:length+1]=1

    text_embds = torch.tensor([49406,   692,  2782,  2100,  6989,   962,   320, 39215,   753,   269,
                            49407,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                0,     0,     0,     0,     0,     0,     0], dtype=torch.int32)
    text_embds = text_embds.unsqueeze(0).repeat(8,1)

    input = {"text": text_embds.cuda(),
             "imgs": torch.randn(8, 20, 3, 224, 224).cuda(),
             "padding_mask": padding_mask.cuda(),
             'valid_len': valid_len.cuda(),
             'is_train': False
             }

    loss, acc, kdlls, retrieved_index = model(**input)



        
