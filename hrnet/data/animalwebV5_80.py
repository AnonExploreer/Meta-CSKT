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
from ..utils.augmentation_pool import RandAugmentMC
import torchvision.transforms as transforms
from ..core.evaluation import  compute_shift_pre_data
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
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.label_type = cfg.MODEL.TARGET_TYPE

        self.flip = cfg.DATASET.FLIP
        # self.transform_s = transforms.Compose([RandAugmentMC(2, 10, 6),
        #                                       transforms.ToTensor(),
        #                                       normalize])
        # self.transform_w = transforms.Compose([
        #             transforms.ToTensor(),
        #             normalize
        #         ])
        self.transform = RandAugmentMC(2, 10, 6)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # load annotations
        self.landmarks_frame = []
        fw = open(self.txt_file, 'r')
        for line in fw:
            self.landmarks_frame.append(list(line.strip('\n').split(','))[0])
        self.bbox_data = pd.read_csv(self.bbox_file)
        
        self.pseudo = pseudo_dict
        self.all_pts = {}
        for i in self.landmarks_frame:
            self.all_pts[i] = read_pts(os.path.join(self.data_root,
                            i).replace('jpg','pts'))
        if self.pseudo:
            if len(self.pseudo) != 0:
                for k,v in self.pseudo.items():
                    if k not in self.landmarks_frame:
                        self.landmarks_frame.insert(0,k)
                        self.all_pts[k] = v
        
        if self.mode == 'unlabel':
            bad, good = {},{}
            self.human_shift_file = args.human_shift
            self.flip_shift_file = args.flip_shift
            self.human_shift = pd.read_csv(self.human_shift_file)
            self.flip_shift = pd.read_csv(self.flip_shift_file)

            self.landmarks = []
            for x in self.landmarks_frame:
                
                n = self.flip_shift[self.flip_shift['img'] == x].values[0][1]
                if n < args.f_threshold:
                    self.landmarks.append(x)
                else:
                    bad[x] = n

            self.landmarks_frame = self.landmarks
            print(f'Length of selected samples: {len(self.landmarks_frame)}')
            self.p = {}
            for x in self.landmarks_frame:
                n = self.human_shift[self.human_shift['img'] == x].values[0][1]
                self.p[x] = 0
                if n < args.l_threshold:
                    self.p[x]=1
                    good[x] = n
            with open('good.json','w') as f1, open('bad.json','w') as f2:
                json.dump(good,f1)
                json.dump(bad,f2)
            random.shuffle(self.landmarks_frame)
            print(f'length of positive samples: {len([x for x in self.landmarks_frame if self.p[x] == 1])}')


    def __len__(self):
        l = len(self.landmarks_frame)
        return l

    def __getitem__(self, idx):
        imgfile = self.landmarks_frame[idx]
        image_path = os.path.join(self.data_root,imgfile)
        bbox = self.bbox_data[self.bbox_data['img'] == imgfile].values[0][1:]
        x,y,w,h = bbox
        scale = max(w,h)*1.0/200
        box_size = w*h

        center_h = y+h/2.
        center_w = x+w/2.
        center = torch.Tensor([center_w, center_h])
        pts = read_pts(os.path.join(self.data_root,
                            imgfile).replace('jpg','pts'))

        if self.mode == 'label':
            pts = self.all_pts[imgfile]
            for idx,pt in enumerate(pts):
                if random.random() < 0.5:
                    if idx >4:
                        pts[idx,0] += random.uniform(-2,2) * scale
                        pts[idx,1] += random.uniform(-2,2) * scale


        
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


        img1 = Image.fromarray(img1)
        img1 = self.transform(img1)
        img1 = np.array(img1)
        img1 = img1.astype(np.float32)
        img1 = (img1/255.0 - self.mean) / self.std
        img1 = img1.transpose([2, 0, 1])

        
        
        center = torch.Tensor(center)
        
        meta = {'index': idx, 'center': center, 'scale': scale, 
                'pts': torch.Tensor(pts), 'box_size': box_size,'image':imgfile}
        
        if self.mode == 'label':
            return img, target, meta
        elif self.mode == 'unlabel':
            pseudo = self.p[imgfile]
            return img, img1, meta, pseudo 
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