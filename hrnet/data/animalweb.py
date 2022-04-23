import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
import json
import copy

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


class AnimalWeb(data.Dataset):
    """AFLW
    """
    def __init__(self, cfg, args, label = True, is_train=True):
        # specify annotation file for dataset
        self.label = label
        self.is_train = is_train
        if is_train:
            if label:
                self.txt_file = cfg.DATASET.TRAINSET
            else:
                self.txt_file = cfg.DATASET.UNLABELSET
            # visfile = '../data/animal/vis.json'
            # with open(visfile,'r') as vis_file:
            #     self.visdata = json.load(vis_file)
        else:
            if args.evaluate:
                self.txt_file = cfg.DATASET.TESTSET
            else:
                self.txt_file = cfg.DATASET.VALSET
            
        self.bbox_file = cfg.DATASET.BBOX
        self.is_train = is_train
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA

        # self.scale_factor = cfg.DATASET.SCALE_FACTOR
        # self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.weak_scale_factor = cfg.DATASET.WEAK_SCALE_FACTOR
        self.weak_rot_factor = cfg.DATASET.WEAK_ROT_FACTOR
        self.strong_scale_factor = cfg.DATASET.STRONG_SCALE_FACTOR
        self.strong_rot_factor = cfg.DATASET.STRONG_ROT_FACTOR

        self.label_type = cfg.MODEL.TARGET_TYPE

        self.flip = cfg.DATASET.FLIP

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # load annotations
        self.landmarks_frame = []
        fw = open(self.txt_file, 'r')
        for line in fw:
            self.landmarks_frame.append(list(line.strip('\n').split(',')))
        self.bbox_data = pd.read_csv(self.bbox_file)
        

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        imgfile = self.landmarks_frame[idx][0]
        image_path = os.path.join(self.data_root,imgfile)
                                  
        bbox = self.bbox_data[self.bbox_data['img'] == imgfile].values[0][1:]
        x,y,w,h = bbox
        scale = max(w,h)*1.0/200
        box_size = w * h

        center_h = y+h/2.
        center_w = x+w/2.
        center = torch.Tensor([center_w, center_h])
        
        pts = read_pts(os.path.join(self.data_root,
                                  imgfile).replace('jpg','pts'))

        
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        img1 = copy.deepcopy(img)
        img2 = copy.deepcopy(img)
        pts1 = pts.copy()
        pts2 = pts.copy()
        scale1 = scale
        center1 = torch.Tensor([center_w, center_h])
        scale2 = scale
        center2 = torch.Tensor([center_w, center_h])
        rot_weak = 0
        rot_strong = 0
        # if self.is_train:
            
        #     # scale = scale * (random.uniform(1 - self.weak_scale_factor,
        #     #                                 1 + self.weak_scale_factor))
        #     # scale1 = scale1 * (random.uniform(1 - self.strong_scale_factor,
        #     #                                 1 + self.strong_scale_factor))
        #     rot_weak = random.uniform(-self.weak_rot_factor, self.weak_rot_factor) 
        #     rot_strong = random.uniform(-self.strong_rot_factor, self.strong_rot_factor) 

        #         # if random.random() <= 0.6 else 0
        #     if random.random() <= 0.5 and self.flip:
        #         img1 = np.fliplr(img1)
        #         pts1 = fliplr_joints(pts1, width=img.shape[1], dataset='AFLW')
        img1 = np.fliplr(img1)
        center1[0] = img1.shape[1] - center1[0]
        h = img1.shape[1]
        
        img = crop(img, center, scale, self.input_size, rot=0)
        img1 = crop(img1,center1, scale1, self.input_size, rot=0)
        if self.is_train:
            # vis = self.visdata[imgfile.replace('jpg','pts')]
            target = np.zeros((nparts, self.output_size[0], self.output_size[1]))#2*nparts
        else:
            target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()
        
        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=0)
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)
                # if self.is_train:
                #     if vis[i] == 0:
                #         target[i+nparts].fill(1)
                #     else:
                #         target[i+nparts].fill(0)
                
        
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        img1 = img1.astype(np.float32)
        img1 = (img1/255.0 - self.mean) / self.std
        img1 = img1.transpose([2, 0, 1])

        
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        
        meta = {'index': idx, 'center': center, 'scale': scale, 'scale1':scale1,'rot_weak':rot_weak,'rot_strong':rot_strong,'h':h,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'box_size': box_size,'image':imgfile}
        
        if self.label is False and self.is_train:
            return img1, img2, target, meta
        
        return img, target, meta



if __name__ == '__main__':

    pass