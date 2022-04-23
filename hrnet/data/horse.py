# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
import json

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Horse(data.Dataset):
    """AFLW
    """
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.json_file = cfg.DATASET.TRAINSET
        else:
            self.json_file = cfg.DATASET.TESTSET
        
        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # load annotations
        
        file = open(self.json_file,'r')
            
        self.landmarks_frame = json.load(file)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame[idx]['image']).replace('\\','/')
        scale = self.landmarks_frame[idx]['scale']
        box_size = scale*200

        center_w = self.landmarks_frame[idx]['center'][0]
        center_h = self.landmarks_frame[idx]['center'][1]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame[idx]['joints']
        pts = np.array(pts)

        
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0

        img = crop(img, center, scale, self.input_size, rot=0)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        # for i in range(nparts):
        #     if tpts[i, 1] > 0:
        #         tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
        #                                        scale, self.output_size, rot=r)
        #         target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
        #                                     label_type=self.label_type)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        # target = torch.Tensor(target)
        # tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'box_size': box_size,'image':self.landmarks_frame[idx]['image']}

        return img, meta


if __name__ == '__main__':

    pass