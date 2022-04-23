import argparse
import logging
import math
import os
import random
import time
import csv
import cv2
from numpy.lib.function_base import rot90
import scipy.ndimage
import json

from hrnet.lib.config import config,update_config
from hrnet import hrnet
from hrnet.utils import utils
from hrnet.core.evaluation import compute_nme,compute_nme_human, compute_nme_point, compute_shift, compute_shift_pre,decode_preds
from hrnet.utils.transforms import fliplr_joints, crop, generate_target_inverse, transform_pixel


import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.cuda import amp
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hrnet.data.data import DATASET_GETTERS
from hrnet.data.animalweb import read_pts

from utils import (AverageMeter, accuracy, create_loss_fn,
                   model_load_state_dict, reduce_tensor, save_checkpoint)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./checkpoint3', type=str, help='save path')
parser.add_argument('--dataset', default='animalweb', type=str,
                    choices=['cifar10', 'cifar100','animalweb'], help='dataset name')
parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
# parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=125, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=512, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=1e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--vis', action='store_true',
                    help='output the img of test set')




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < 50:
            return 0.0

        if current_step <num_wait_steps:
            return float(current_step) / float(max(1, num_wait_steps))

        progress = float(current_step - num_wait_steps) / \
            float(max(1, num_training_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def inference(args, data_loader,model,criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    
    
    csvfile = open(f'test_result/test_nme_{args.name}.csv', 'w',newline='')
    print(f'test_result/test_nme_{args.name}.csv')
    csv_write = csv.writer(csvfile)
    csv_head = ["img", "nme"]
    csv_write.writerow(csv_head)
    test_iter = tqdm(data_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inp = inp.to(args.device)
            target = target.to(args.device)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])            
            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            
            nme_avg = np.sum(nme_temp)/preds.size(0)
            
            
            for n in range(score_map.size(0)):
                csv_write.writerow([meta['image'][n],nme_avg])
                predictions[meta['index'][n], :, :] = preds[n, :, :]
            

            test_iter.set_description(
                f"Test Iter: {i+1:3}/{len(data_loader):3}. Data: {data_time.avg:.2f}s. "
                f"img: {meta['image']} , NME: {nme_avg:.2f}.")

            batch_time.update(time.time() - end)
            end = time.time()
    print('csv file written finish')
    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    print(msg)
    logger.info(msg)
    csvfile.close()
    return nme, predictions



def main():
    args = parser.parse_args()
    update_config(config, args.cfg)
    args.nme = 1e6
    if args.local_rank != -1:
        args.gpu = args.local_rank
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
    else:
        args.gpu = 0
        args.world_size = 1

    args.device = torch.device('cuda', args.gpu)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"results/{args.name}")
        wandb.init(name=args.name, project='MPL', config=args)

    if args.seed is not None:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    if args.dataset == "cifar10":
        depth, widen_factor = 28, 2
    elif args.dataset == 'cifar100':
        depth, widen_factor = 28, 8

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()


    teacher_model = hrnet.get_face_alignment_net(config)
    student_model = hrnet.get_face_alignment_net(config)
    if args.local_rank == 0:
        torch.distributed.barrier()

    # logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")

    teacher_model.to(args.device)
    student_model.to(args.device)
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    t_optimizer = utils.get_optimizer(config,teacher_model)
    s_optimizer = utils.get_optimizer(config,student_model)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            if not (args.evaluate or args.finetune or args.vis):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                model_load_state_dict(student_model, checkpoint['student_state_dict'])
            else:
                model_load_state_dict(student_model, checkpoint['student_state_dict'])#

            logger.info(f"=> loaded checkpoint '{args.resume}'")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)



    if args.evaluate:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        inference(args, test_loader, student_model, criterion)
        return

    teacher_model.zero_grad()
    student_model.zero_grad()

    return


if __name__ == '__main__':
    main()
