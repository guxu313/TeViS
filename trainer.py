import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import deepspeed

from src.utils.dist import master_process
from src.utils.metrics import compute_rt_metrics

from tensorboardX import SummaryWriter

from src.utils.logger import LOGGER
from src.utils.dist import concat_all_gather
import shutil
import scipy.stats as stats


class Trainer():
    def __init__(self, args, cfg, model, optimizer, scheduler,
                    dataloader_trains, dataloader_vals=None):
        self.cfg = cfg
        self.local_rank = model.local_rank
        self.global_step = 0
        self.start_epoch = 0
        self.total_epochs = cfg.TRAINING.EPOCHS
        self.dataloader_trains = dataloader_trains
        self.dataloader_vals = dataloader_vals
        
        self.args = args

        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer

        if master_process(self.args) and cfg.log_tb:
            self.summary_writer = SummaryWriter(log_dir=os.path.join(args.blob_mount_dir,cfg.TRAINING.save_dir,'tb_log'))

    def _checkpoint(self,PATH, ckpt_id, epoch, global_step):

        """Utility function for checkpointing model + optimizer dictionaries
        The main purpose for this is to be able to resume training from that instant again
        """
        checkpoint_state_dict = {
            'epoch': epoch,
            'global_step': global_step,
        }
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'checkpoint')
        if self.args.local_rank == 0:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)

        save_trial = 0
        while save_trial < 10:
            try:
                LOGGER.info(f"checkpointing trial NO. {save_trial}")
                success = self.model.save_checkpoint(save_dir, ckpt_id, checkpoint_state_dict)
                status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(save_dir, ckpt_id)
                if success:
                    LOGGER.info(f"Success {status_msg}")
                    break
            except Exception as e:
                save_trial += 1
                LOGGER.warning(f"Failure {status_msg}")
        dist.barrier()


    def _save_model(self,PATH, step):
        
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'saved_model', 'step_{0:05d}'.format(step))
        save_trial = 0
        while save_trial < 10:
            try:
                sucess = self.model.save_fp16_model(save_dir)
                break
            except Exception as e:
                save_trial += 1
                LOGGER.warning(f"Failure save model")

    def _resume(self,PATH, tag=None):
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'checkpoint')
        LOGGER.info(f"resume from {save_dir}")
        _, checkpoint_state_dict = self.model.load_checkpoint(save_dir)
        self.start_epoch = checkpoint_state_dict['epoch']
        self.global_step = checkpoint_state_dict['global_step']
        del checkpoint_state_dict

    def report_step_metrics(self, lr=0, loss=None, acc=None, is_train=True):
        ##### Record the LR against global_step on tensorboard #####
        if master_process(self.args):

            if is_train:
                prefix = 'Train'
                if self.cfg.log_tb:
                    self.summary_writer.add_scalar(f'Train/lr', lr, self.global_step)
            else:
                prefix = 'Val'

            if self.cfg.log_tb:

                self.summary_writer.add_scalar(f'{prefix}/loss', loss,self.global_step)
                self.summary_writer.add_scalar(f'{prefix}/acc', acc,self.global_step)

            ##### Recording  done. #####

            if self.global_step % self.cfg.TRAINING.print_step == 0 or not is_train:

                LOGGER.info(f"training_progress: {prefix} step={self.global_step}, loss={loss}, acc={acc}")


    @torch.no_grad()
    def evaluate(self, dataloader_val):

        LOGGER.info(f"start evaluate on task dataset.")

        self.model.eval()
        st = time.time()

        all_loss = []
        all_acc = []

        all_kdll = []
        all_kdll_3_5 = [] 
        all_kdll_6_max = []

        for step, batch in enumerate(dataloader_val):

            text = batch['text'].to(self.local_rank)
            imgs = batch['imgs'].to(self.local_rank)
            padding_mask = batch['padding_mask'].to(self.local_rank)
            valid_len = batch['valid_len'].to(self.local_rank)

            if self.args.fp16:
                imgs = imgs.half()
                
            loss, acc, kdlls, retrieved_index= self.model(text, imgs, padding_mask, valid_len, is_train = False)
            [kdll, kdll_2, kdll_3_5, kdll_6_max] = kdlls

            all_loss.append(torch.tensor([loss.item()], device=loss.device))
            all_acc.append(torch.tensor([acc.item()], device=acc.device))

            all_kdll.append(kdll)
            all_kdll_3_5.append(kdll_3_5)
            all_kdll_6_max.append(kdll_6_max)


        all_loss = torch.cat(all_loss, dim=0).to(self.local_rank)
        all_acc = torch.cat(all_acc, dim=0).to(self.local_rank)
        all_kdll = torch.cat(all_kdll, dim=0).to(self.local_rank)
        all_kdll_3_5 = torch.cat(all_kdll_3_5, dim=0).to(self.local_rank)
        all_kdll_6_max = torch.cat(all_kdll_6_max, dim=0).to(self.local_rank)

        all_loss = concat_all_gather(all_loss).mean()
        all_acc = concat_all_gather(all_acc).mean()

        all_kdll =  concat_all_gather(all_kdll.mean().unsqueeze(0)).mean()
        all_kdll_3_5 = concat_all_gather(all_kdll_3_5.mean().unsqueeze(0)).mean()
        all_kdll_6_max = concat_all_gather(all_kdll_6_max.mean().unsqueeze(0)).mean()

        if master_process(self.args) and self.cfg.log_tb:
            self.summary_writer.add_scalar(f'Valid/loss', all_loss, self.global_step)
            self.summary_writer.add_scalar(f'Valid/acc', all_acc, self.global_step)
            self.summary_writer.add_scalar(f'Valid/kdll', all_kdll, self.global_step)
            self.summary_writer.add_scalar(f'Valid/kdll_3_5', all_kdll_3_5, self.global_step)
            self.summary_writer.add_scalar(f'Valid/kdll_6_max', all_kdll_6_max, self.global_step)

        LOGGER.info(f"finish evaluate on task dataset after {time.time() - st} seconds.")
        LOGGER.info(f"Loss: {all_loss:.4f}, ACC: {all_acc:.4f},\n \
                    Kdll: {all_kdll:.8f}, Kdll_3_5: {all_kdll_3_5:.8f}, Kdll_6_max: {all_kdll_6_max:.8f}")
        self.model.train()

  
    def train(self, resume):
        self.model.train()
        if resume:
            self._resume(self.cfg.TRAINING.save_dir)
            LOGGER.info(f'resume from {self.start_epoch}, global step {self.global_step}')
            steps_trained_in_current_epoch = self.global_step % (sum([len(dataloader) for _,dataloader in self.dataloader_trains.items()]))
        else:
            steps_trained_in_current_epoch = 0
        LOGGER.info(f'begin training from {self.start_epoch}')
        for epoch in range(self.start_epoch, self.total_epochs):
            for dataloader_name, dataloader_train in self.dataloader_trains.items():
                LOGGER.info('Train on '+dataloader_name+', Epoch'+str(epoch))

                if self.args.distributed:
                    dataloader_train.sampler.set_epoch(epoch)

                for step, batch in enumerate(dataloader_train):

                    text = batch['text'].to(self.local_rank)
                    imgs = batch['imgs'].to(self.local_rank)
                    padding_mask = batch['padding_mask'].to(self.local_rank)
                    valid_len = batch['valid_len'].to(self.local_rank)
    
                    if self.args.fp16:
                        imgs = imgs.half()

                    loss, acc, _, _ = self.model(text, imgs, padding_mask, valid_len)

                    self.model.backward(loss)
                    self.model.step()

                    self.global_step += 1
                    self.scheduler.step_update(self.global_step)
                    lr = self.scheduler._get_lr(self.global_step)[0]
                    self.report_step_metrics(lr, loss.item(), acc.item())

                    if self.global_step % self.cfg.TRAINING.eval_step == 0:
                        for dataloader_name,dataloader_val in self.dataloader_vals.items():
                            LOGGER.info('Eval on '+dataloader_name)
                            self.evaluate(dataloader_val)


                    if self.global_step % self.cfg.TRAINING.checkpoint_step == 0:
                        self._checkpoint(self.cfg.TRAINING.save_dir, self.global_step, epoch, self.global_step)


                    if self.global_step % self.cfg.TRAINING.save_step == 0:
                            self._save_model(self.cfg.TRAINING.save_dir, self.global_step)
                        
                    if self.global_step > self.cfg.TRAINING.BREAK_STEP:
                            LOGGER.info(f"Job finished")
                            break
                            
            self.start_epoch = epoch
            LOGGER.info('Epoch: '+ str(epoch))
            if self.global_step > self.cfg.TRAINING.BREAK_STEP:
                LOGGER.info(f"Job finished")
                break

