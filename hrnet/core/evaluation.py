# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from ..utils.transforms import transform_preds


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def compute_nme_human(preds, meta):

    targets = meta['pts']
    scale = meta['scale'][0].item()
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        interocular = math.sqrt(meta['box_size'][i].item())
        errorsum = 0
        errorsum += np.sum(np.linalg.norm(pts_pred[6, ]-pts_gt[0, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[8, ]-pts_gt[1, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[9, ]-pts_gt[2, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[11, ]-pts_gt[3, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[13, ]-pts_gt[4, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[15, ]-pts_gt[5, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[17, ]-pts_gt[6, ]))
        rmse[i] = errorsum / (interocular * 7)
    return rmse

def compute_nme_horse(preds, meta):

    targets = meta['pts']
    scale = meta['scale'][0].item()
    box_size = meta['box_size'][0].item()
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        interocular = meta['box_size'][i].item()
        errorsum = 0
        cnt = 0
        if pts_gt[0,0] > 0 and pts_gt[0,1] > 0:
            cnt += 1
            errorsum += np.sum(np.linalg.norm((pts_pred[0, ]+pts_pred[1, ])/2.0-pts_gt[0, ]))
        if pts_gt[1,0] > 0 and pts_gt[1,1] > 0:
            cnt += 1
            errorsum += np.sum(np.linalg.norm((pts_pred[2, ]+pts_pred[3, ])/2.0-pts_gt[1, ]))
        if pts_gt[2,0] > 0 and pts_gt[2,1] > 0:
            cnt += 1
            errorsum += np.sum(np.linalg.norm(pts_pred[4, ]-pts_gt[2, ]))
        if pts_gt[3,0] > 0 and pts_gt[3,1] > 0:
            cnt += 1
            errorsum += np.sum(np.linalg.norm(pts_pred[5, ]-pts_gt[3, ]))
        if pts_gt[4,0] > 0 and pts_gt[4,1] > 0:
            cnt += 1
            errorsum += np.sum(np.linalg.norm(pts_pred[6, ]-pts_gt[4, ]))

        rmse[i] = errorsum / (interocular * cnt)

    return rmse


def compute_nme(preds, meta):

    targets = meta['pts']
    scale = meta['scale'][0].item()
    box_size = meta['box_size'][0].item()
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 5:
            interocular = math.sqrt(meta['box_size'][i].item())
            errorsum = 0
            cnt = 0
            for j in range(L):
                if pts_gt[j,0] > 0 and pts_gt[j,1] > 0:
                    cnt += 1
                    errorsum += np.sum(np.linalg.norm(pts_pred[j, ]-pts_gt[j, ]))
            rmse[i] = errorsum / (interocular * cnt) if cnt > 0 else 0
            continue
        elif L == 9:
            interocular = math.sqrt(meta['box_size'][i].item())
        else:
            raise ValueError('Number of landmarks is wrong')
        errorsum = 0
        for j in range(L):
            if pts_pred[j,0] != 0 and pts_pred[j,1] != 0:
                errorsum += np.sum(np.linalg.norm(pts_pred[j, ]-pts_gt[j, ]))
        rmse[i] = errorsum / (interocular * L)

    return rmse

def compute_nme_point(preds, meta):

    targets = meta['pts']
    box_size = meta['box_size'][0].item()
    scale = meta['scale'][0].item()
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros((N,L))

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 5:
            interocular = math.sqrt(meta['box_size'][i].item())
            for j in range(L):
                errorsum = 0
                if pts_gt[j,0] > 0 and pts_gt[j,1] > 0:
                    errorsum = np.sum(np.linalg.norm(pts_pred[j, ]-pts_gt[j, ]))
                rmse[i,j] = errorsum / interocular
            continue
        elif L == 9:
            interocular = math.sqrt(meta['box_size'][i].item())
        else:
            raise ValueError('Number of landmarks is wrong')
        errorsum = 0
        for j in range(L):
            if pts_pred[j,0] != 0 and pts_pred[j,1] != 0:
                errorsum = np.sum(np.linalg.norm(pts_pred[j, ]-pts_gt[j, ]))
            rmse[i,j] = errorsum / (interocular)

    return rmse

def compute_shift_pre(preds1, preds2, meta):

    scale = meta['scale'][0].item()
    preds1 = preds1.numpy() # human
    preds2 = preds2.numpy()
 
    N = preds2.shape[0]
    L = preds2.shape[1]
    rmse = np.zeros(N)

    for i in range(N): 
        pts_pred, pts_gt = preds1[i, ], preds2[i, ]
        interocular = math.sqrt(meta['box_size'][i].item())
        errorsum = 0
        errorsum += np.sum(np.linalg.norm(pts_pred[6, ]-pts_gt[0, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[8, ]-pts_gt[1, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[9, ]-pts_gt[2, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[11, ]-pts_gt[3, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[13, ]-pts_gt[4, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[15, ]-pts_gt[5, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[17, ]-pts_gt[6, ]))
        rmse[i] = errorsum / (interocular * 7)

    return rmse

def compute_shift_pre_data(preds1, preds2, scale):

    preds1 = np.array(preds1)
    preds2 = np.array(preds2)
    N = preds2.shape[0]
    L = preds2.shape[1]
    rmse = np.zeros(N)

    for i in range(N): 
        pts_pred, pts_gt = preds1[i, ], preds2[i, ]
        interocular = np.linalg.norm(math.sqrt(scale*200))
        errorsum = 0
        errorsum += np.sum(np.linalg.norm(pts_pred[6, ]-pts_gt[0, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[8, ]-pts_gt[1, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[9, ]-pts_gt[2, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[11, ]-pts_gt[3, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[13, ]-pts_gt[4, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[15, ]-pts_gt[5, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[17, ]-pts_gt[6, ]))
        rmse[i] = errorsum / (interocular * 7)

    return rmse

def compute_shift_data(preds1, preds2, scale):
    preds1 = np.array(preds1)
    preds2 = np.array(preds2)
    N = preds2.shape[0]
    L = preds2.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds1[i, ], preds2[i, ]
        interocular = np.linalg.norm(math.sqrt(scale*200))
        errorsum = 0
        errorsum += np.sum(np.linalg.norm(pts_pred[0, ]-pts_gt[3, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[1, ]-pts_gt[2, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[2, ]-pts_gt[1, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[3, ]-pts_gt[0, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[4, ]-pts_gt[4, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[5, ]-pts_gt[6, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[6, ]-pts_gt[5, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[7, ]-pts_gt[7, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[8, ]-pts_gt[8, ]))
        rmse[i] = errorsum / (interocular * L)

    return rmse




def compute_shift(preds1, preds2, scale):
    preds1 = np.array(preds1)
    preds2 = np.array(preds2)
    N = preds2.shape[0]
    L = preds2.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds1[i, ], preds2[i, ]
        pts_gt[:,0] = meta['h'][i].item() - pts_gt[:,0]
        interocular = math.sqrt(meta['box_size'][i].item())
        errorsum = 0
        errorsum += np.sum(np.linalg.norm(pts_pred[0, ]-pts_gt[3, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[1, ]-pts_gt[2, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[2, ]-pts_gt[1, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[3, ]-pts_gt[0, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[4, ]-pts_gt[4, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[5, ]-pts_gt[6, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[6, ]-pts_gt[5, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[7, ]-pts_gt[7, ]))
        errorsum += np.sum(np.linalg.norm(pts_pred[8, ]-pts_gt[8, ]))
        rmse[i] = errorsum / (interocular * L)

    return rmse






def decode_preds(output, center, scale, res):
    
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds
