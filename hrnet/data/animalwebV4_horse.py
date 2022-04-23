import os
import random
import cv2
import torch
from torch import tensor
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
import json
from torch.utils.data import DataLoader
from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


class AnimalWeb(data.Dataset):
    """AFLW
    """
    def __init__(self, cfg, args, mode, pseudo_dict = None):
        # pseudo_dict: ['img.jpg': pts]
        # unlabel_list: ['img1.jpg', 'img2.jpg']
        # specify annotation file for dataset
        self.mode = mode
        if mode == 'test':            
            if args.evaluate:
                self.txt_file = cfg.DATASET.TESTSET
            else:
                self.txt_file = cfg.DATASET.VALSET
        else:
            if mode == 'label':
                self.txt_file = cfg.DATASET.TRAINSET
            else:
                self.txt_file = cfg.DATASET.UNLABELSET
        self.bbox_file = cfg.DATASET.BBOX
        self.unlabel_bbox = pd.read_csv('/home/cseadmin/hsc/mpl/MPL-pytorch/data/bbox.csv')
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.label_type = cfg.MODEL.TARGET_TYPE

        self.flip = cfg.DATASET.FLIP

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # load annotations
        self.landmarks_frame = []
        fw = open(self.txt_file, 'r')
        for line in fw:
            self.landmarks_frame.append(list(line.strip('\n').split(','))[0])
        fw.close()
        if self.mode == 'unlabel':
            with open(cfg.DATASET.TRAINSET,'r') as fw:
                for line in fw:
                    self.landmarks_frame.append(list(line.strip('\n').split(','))[0])
        self.bbox_data = pd.read_csv(self.bbox_file)
        

    def __len__(self):
        l = len(self.landmarks_frame)
        return l

    def __getitem__(self, idx):
        imgfile = self.landmarks_frame[idx]
        if imgfile[:8] == 'horse/im':
            image_path = os.path.join(self.data_root,imgfile)
            bbox = self.bbox_data[self.bbox_data['img'] == imgfile].values[0][1:]
        else:
            image_path = os.path.join('/home/cseadmin/hsc/animal_dataset_v1_c/animal_dataset_v1_clean_check',imgfile)
            bbox = self.unlabel_bbox[self.unlabel_bbox['img'] == imgfile].values[0][1:]
        x,y,w,h = bbox
        scale = max(w,h)*1.0/200
        box_size = w*h

        center_h = y+h/2.
        center_w = x+w/2.
        center = torch.Tensor([center_w, center_h])
        pts = read_pts(image_path.replace('jpg','pts'))
        
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)


        img = crop(img, center, scale, self.input_size, rot=0)

        if self.mode != 'unlabel':
            target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
            tpts = pts.copy()
            
            for i in range(nparts):
                if tpts[i, 1] > 0:
                    tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                                    scale, self.output_size, rot=0)
                    target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                                label_type=self.label_type)
            target = torch.Tensor(target)
            tpts = torch.Tensor(tpts)

                
        img1 = img.copy()
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])


        img1 = np.fliplr(img1)
        img1 = img1.astype(np.float32)
        img1 = (img1/255.0 - self.mean) / self.std
        img1 = img1.transpose([2, 0, 1])

        
        
        center = torch.Tensor(center)
        
        meta = {'index': idx, 'center': center, 'scale': scale, 
                'pts': torch.Tensor(pts), 'box_size': box_size,'image':imgfile}
        
        if self.mode == 'label':
            return img, target, meta
        elif self.mode == 'unlabel':
            return img, meta
        else:
            return img, target, meta

class dataloader_gen:
    def __init__(self,cfg,args):
        self.cfg = cfg
        self.args = args
        self.pseudo_dict = {}
        self.unlabel_list = []
    def run(self, mode, pseudo_dict = None):
        
        if pseudo_dict:
            if len(pseudo_dict) != 0:
                self.pseudo_dict.update(pseudo_dict)
        if mode == 'label':
            return DataLoader(
            AnimalWeb(self.cfg,self.args, 'label', pseudo_dict=self.pseudo_dict if len(self.pseudo_dict) != 0 else None),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=self.cfg.TRAIN.SHUFFLE,
            num_workers=self.cfg.WORKERS,
            pin_memory=self.cfg.PIN_MEMORY)

        elif mode == 'unlabel':
            return DataLoader(
            AnimalWeb(self.cfg,self.args, 'unlabel'),
            batch_size=4 * self.cfg.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            pin_memory=self.cfg.PIN_MEMORY)
        else:
            return
if __name__ == '__main__':

    pass