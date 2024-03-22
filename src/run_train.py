import torch
import torch.distributed as dist
import deepspeed
import argparse
import os
from mmcv import Config

from src.models.model import MovieGPTDecoderPrefix
from src.models.model_cross import MovieGPTDecoderCross

from src.trainer import Trainer
from src.datasets.dataloader import build_dataloader
from src.optimization.lr_scheduler import build_scheduler
from src.optimization.optimizer import build_optimizer_parameters
from src.optimization.loss import build_loss_func

from src.utils.logger import LOGGER, add_log_to_file
from src.utils.dist import master_process
from src.utils.misc import mkdirp, set_random_seed
from src.utils.load import load_model_weights_with_mismatch

import json
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./src/configs/train.yaml')
    parser.add_argument('--blob_mount_dir', default="/blob_mount")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--fp16', action='store_true', help='enable fp16')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--distributed',action='store_true')
    parser.add_argument('--resume', action='store_true')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    cfg = Config.fromfile(args.cfg)

    LOGGER.info(cfg)
    LOGGER.info(args)


    if not master_process(args):
        LOGGER.disabled = True
    if master_process(args):
        mkdirp(os.path.join(args.blob_mount_dir, cfg.TRAINING.save_dir,"log"))
        add_log_to_file(os.path.join(args.blob_mount_dir, cfg.TRAINING.save_dir,"log/log.txt"))

    if cfg.model_type == 'prefix':
        model = MovieGPTDecoderPrefix(args, cfg)
    if cfg.model_type == 'cross':
        model = MovieGPTDecoderCross(args, cfg)

    if cfg.WEIGHTS.model_weight != '':
        LOGGER.info(f"Loading model weights from {cfg.WEIGHTS.model_weight}")
        load_model_weights_with_mismatch(model, os.path.join(args.blob_mount_dir, cfg.WEIGHTS.model_weight))


    parameter_group = build_optimizer_parameters(cfg, model)

    # init deepspeed
    if args.distributed:

        model_engine, optimizer, _, _ = deepspeed.initialize(args = args,
                                                            model=model,
                                                            model_parameters=parameter_group,
                                                            config=cfg.deepspeed_config
                                                            )
        print(dist.get_rank())
    

    LOGGER.info(f'Training with {dist.get_world_size()} gpus')
    

    dataset_trains, dataset_vals, dataloader_trains, dataloader_vals = build_dataloader(args, cfg)

    steps_per_epoch=sum([len(dataloader) for _,dataloader in dataloader_trains.items()])
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch)


    if args.fp16:
        LOGGER.info('Enable fp16 Training')
        fp16 = model_engine.fp16_enabled()


    trainer = Trainer(args, cfg, model_engine, optimizer, scheduler, dataloader_trains, dataloader_vals)

    LOGGER.info('start first evaluate')
    for dataloader_name,dataloader_val in dataloader_vals.items():
        trainer.evaluate(dataloader_val)

    trainer.train(args.resume)

if __name__ == '__main__':
    deepspeed.init_distributed()
    main()