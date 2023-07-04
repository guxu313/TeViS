from abc import abstractmethod
from torch.utils.data import Dataset,DataLoader
import os
from collections import defaultdict
import numpy as np
from torch.utils.data.dataloader import default_collate

from src.utils.logger import LOGGER
import random
import torch
import jsonlines
import json
import operator
import functools
import decord
from PIL import Image
from decord import VideoReader, cpu
import traceback
from src.models.clip import clip
from tqdm import tqdm
from pathlib import Path
import pickle

decord.bridge.set_bridge("torch")


class VideoDataset(Dataset):

    def __init__(self,
                 name,
                 split,
                 metadata_dir,
                 video_path,
                 sample_rate,
                 max_length,
                 tokenizer=None,
                 transform=None,
                 return_rawtext=False,
                 return_index=False,
                 **kwargs
                 ):

        self.name =name
        self.split =split
        self.metadata_dir = metadata_dir
        self.transform = transform
        self.video_path = video_path
        self.return_rawtext = return_rawtext
        self.return_index = return_index
        
        self.tokenizer = tokenizer

        self.sample_rate = sample_rate
        self.max_length = max_length

        self._load_metadata()

    def _load_metadata(self):
        if self.name == 'TeViS':
            with open(self.metadata_dir, 'r') as f:
                data = json.load(f)
        elif self.name == 'CMD':
            data = []
            with open(self.metadata_dir) as f:
                for l in jsonlines.Reader(f):
                    data.append(l)    
        elif self.name == 'MovieNet':
            with open(self.metadata_dir, 'r') as f1:
                data = json.load(f1)
        self.metadata = data



    # lsmdc
    def _read_video(self, video_id, sample_rate):
        '''
        read imgs from long video
        args:
            video_id: str,
            num_frame: imgs used
        return: 
            img_arrays: [num_frm, 3, H, W]
        '''
        if isinstance(video_id,list):
            all_frame = []
            for idx in video_id:
                video_path = os.path.join(self.video_path, idx + '.mp4')
                if 'lsmdc' in self.metadata_dir:
                    video_path = os.path.join(self.video_path, idx + '.avi')
                vr = VideoReader(video_path, ctx=cpu(0))
                all_frame.append(vr.get_batch(range(len(vr))))

            all_frame = torch.cat(all_frame,dim=0)
            total_frame = len(all_frame)
            num_frame = total_frame // self.sample_rate
            frame_idx = np.linspace(0, total_frame-1, num=num_frame).astype(int).tolist()

            if len(frame_idx) < self.max_length:
                frame_idx_with_pad = frame_idx + [0] * (self.max_length - len(frame_idx))
            else:
                frame_idx_with_pad = frame_idx[:self.max_length]

            img_arrays = torch.index_select(all_frame, 0, torch.tensor(frame_idx_with_pad))
        
        else:
            video_path = os.path.join(self.video_path, video_id + '.mp4')
            if 'lsmdc' in self.metadata_dir:
                video_path = os.path.join(self.video_path, video_id + '.avi')
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame = len(vr)
            num_frame = total_frame // sample_rate

            frame_idx = np.linspace(0, total_frame-1, num=num_frame).astype(int).tolist()

            if len(frame_idx) < self.max_length:
                frame_idx_with_pad = frame_idx + [0] * (self.max_length - len(frame_idx))
            else:
                frame_idx_with_pad = frame_idx[:self.max_length]

            img_arrays = vr.get_batch(frame_idx_with_pad)

        img_arrays = img_arrays.float() / 255
        img_arrays = img_arrays.permute(0, 3, 1, 2) # N,C,H,W

        padding_mask = [1] * len(frame_idx) + [0] * (self.max_length - len(frame_idx))
        padding_mask = padding_mask[:self.max_length]

        valid_len = len(frame_idx)-1

        return img_arrays, padding_mask, valid_len

    
    def _read_imgs(self, movie_id, image_list):
        '''
        read imgs from json
        args:
            story_id: str,
            num_frame: imgs used
        return: 
            img_arrays: [num_frm, 3, H, W]
        '''
        imgs_list = image_list
        imgs_list = imgs_list[::self.sample_rate]
        tmp_img_list = []
        for image in imgs_list:
            imgs_path = os.path.join(self.video_path, movie_id, image)
            try:
                img_PIL = Image.open(imgs_path)
                img_PIL = np.array(img_PIL)
                tmp_img_list.append(img_PIL)
            except:
                LOGGER.error("error load imgs_path:", imgs_path)
                continue
        
        if len(tmp_img_list) < self.max_length:
            pad_imgs = [tmp_img_list[0]]
            imgs_list_with_pad = tmp_img_list + pad_imgs * (self.max_length - len(tmp_img_list))
        else:
            imgs_list_with_pad = tmp_img_list[:self.max_length]

        img_arrays = np.array(imgs_list_with_pad)
        img_arrays = torch.from_numpy(img_arrays)
        img_arrays = img_arrays.float() / 255
        img_arrays = img_arrays.permute(0, 3, 1, 2) # N,C,H,W

        padding_mask = [1] * len(tmp_img_list) + [0] * (self.max_length - len(tmp_img_list))
        padding_mask = padding_mask[:self.max_length]
       

        valid_len = len(tmp_img_list)-1
 
        return img_arrays, padding_mask, valid_len

    def _read_sents(self, description):
        '''
        concat description
        args:
            description: str,
        return: 
            descriptions: str
        '''
        description = description
        descriptions = ' '.join(description)
        return descriptions

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        num_retries = 10
        for j in range(num_retries):

            try:
                item = self.metadata[index]

                if self.name == 'TeViS':
                    movie_id = item['movie_id']
                    image_list = item['keyframes']
                    imgs, padding_mask, valid_len = self._read_imgs(movie_id, image_list)
                    description = item['synopses']
                    rawtext = self._read_sents(description)
                elif self.name == 'MovieNet':
                    movie_id = item['movie_id']
                    image_list = item['keyframes']
                    imgs, padding_mask, valid_len = self._read_imgs(movie_id, image_list)
                    rawtext = ' '
                elif self.name == 'CMD':
                    clip_id = item['clip_id']
                    imgs, padding_mask, valid_len = self._read_video(clip_id, self.sample_rate)
                    rawtext = item['text']
                    if isinstance(rawtext, list):
                        rawtext = random.choice(rawtext)


                text = self.tokenizer(rawtext, truncate=True)

                if self.transform is not None:
                    imgs = self.transform(imgs) # N, C, H, W

            except Exception as e:
                traceback.print_exc()
                LOGGER.info(f"Failed to load examples with video: {clip_id}. "
                                f"Will try again.")
                continue
            else:
                break

        data = {
                    'imgs': imgs, # N, C, H, W
                    'text': text.squeeze(0), 
                    'padding_mask': torch.tensor(padding_mask),
                    'valid_len': valid_len,
                }

        if self.return_rawtext:
            data['rawtext'] = rawtext

        if self.return_index:
            data['index'] = torch.tensor(index)

        return data


if __name__ == '__main__':

    from torchvision import transforms
    from PIL import Image
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC
    
    input_res=(224, 224)
    norm_mean=(0.48145466, 0.4578275, 0.40821073)
    norm_std=(0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    transform_dict = transforms.Compose([
            transforms.Resize(input_res, interpolation=BICUBIC),
            normalize,
        ])


    tokenizer = clip.tokenize

    # ourdataset
    dataset = VideoDataset(
        name = 'TeViS',
        split = 'test',
        metadata_dir = '/scripts-translator/metadata/MovieNet_TeViS/val.json',
        video_path = '/blob_mount_movienet',
        sample_rate=1,
        max_length=12,
        tokenizer=tokenizer,
        transform=transform_dict
        )

    temp = dataset[0]

