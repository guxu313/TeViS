import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import transforms

from transformers import CLIPTokenizer

from src.utils.logger import LOGGER

from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import default_collate

from src.utils.dist import SequentialDistributedSampler
from src.models.clip import clip

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from mmcv.utils import Registry
from .videodataset import VideoDataset

DATASETS = Registry('datasets')

DATASETS.register_module(VideoDataset)


class PretrainCollator(object):
    """is_train is kept here if we want to remove
    the randomness during validation of MLM accuracy.
    In that case, instantiate two PretrainCollator"""

    def collate_batch(self, batch):
        # print('collating')

        imgs = default_collate([d["imgs"] for d in batch])
        text = default_collate([d["text"] for d in batch]) # (B, L)
        valid_len = default_collate([d["valid_len"] for d in batch]) # (B, L)
    
        if 'index' in batch[0]:
            index = default_collate([d["index"] for d in batch])
        else:
            index = None

        if 'padding_mask' in batch[0]:
            padding_mask = default_collate([d["padding_mask"] for d in batch])
        else:
            padding_mask = None

        return dict(
                imgs=imgs, # B, N, C, H, W
                text=text, # B, Seq_len
                valid_len = valid_len,
                index = index,
                padding_mask = padding_mask
                )


def init_transform_dict(input_res=(224, 224),
                        center_crop=(224, 224),
                        randcrop_scale=(0.8, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    transform_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale, interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_res, interpolation=BICUBIC),
            # transforms.CenterCrop(center_crop),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_res, interpolation=BICUBIC),
            # transforms.CenterCrop(center_crop),
            normalize,
        ])
    }
    return transform_dict

def build_dataset(args,cfg,tokenizer,split='train'):
    transform=init_transform_dict(cfg.DATA.input_res)[split]

    dataset_dicts = cfg.DATA.DATASET_train if split=='train' else cfg.DATA.DATASET_val
    if isinstance(dataset_dicts, dict):
        dataset_dicts = [dataset_dicts]
    datasets = {}
    for dataset_dict in dataset_dicts:
        
        name = dataset_dict['name']+'_'+dataset_dict['split']
        dataset_dict['metadata_dir'] = os.path.join(args.blob_mount_dir, dataset_dict['metadata_dir'])
        dataset_dict['video_path'] = os.path.join(args.blob_mount_dir, dataset_dict['video_path'])

        dataset_dict['tokenizer'] = tokenizer
        dataset_dict['transform'] = transform

        dataset = DATASETS.build(cfg=dataset_dict)
        LOGGER.info(f'build dataset: {name}, {len(dataset)}')

        datasets[name] = dataset
    return datasets



def build_dataloader(args, cfg):

    tokenizer = clip.tokenize

    dataset_trains = build_dataset(args, cfg,tokenizer, split='train')

    dataset_vals = build_dataset(args, cfg, tokenizer,split='val')

    data_collator = PretrainCollator()

    sampler_train, sampler_val = None, None

    if args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        LOGGER.info(f'using dist training, build sampler')
        
    data_loader_trains = {}
    for k,dataset_train in dataset_trains.items():
        if args.distributed:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=cfg.DATA.BATCH_SIZE_per_gpu,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            collate_fn = data_collator.collate_batch,
            drop_last=True,
        )

        data_loader_trains[k] = data_loader_train

    data_loader_vals = {}
    for k,dataset_val in dataset_vals.items():

        if args.distributed:
            sampler_val = SequentialDistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=cfg.DATA.BATCH_SIZE_per_gpu,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            collate_fn = data_collator.collate_batch,
            drop_last=False
        )
        data_loader_vals[k] = data_loader_val

    LOGGER.info(f'build dataloader done!')
    LOGGER.info(f'dataloader_train: {len(data_loader_train)}')
    for k,v in data_loader_vals.items():
        LOGGER.info(f'data_loader_val {k}: {len(v)}')
    return dataset_trains, dataset_vals, data_loader_trains, data_loader_vals

