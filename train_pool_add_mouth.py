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
from hrnet.core.evaluation import compute_nme, compute_shift, compute_shift_pre,decode_preds
from hrnet.utils.transforms import fliplr_joints, crop, generate_target_inverse, transform_pixel

from hrnet.data.animalwebV4 import AnimalWeb, dataloader_gen


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

from hrnet.data.animalweb import read_pts

# from models import ModelEMA, WideResNet
from utils import (AverageMeter, accuracy, create_loss_fn,
                   model_load_state_dict, reduce_tensor, save_checkpoint)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
parser.add_argument('--name', type=str, required=True, help='experiment name')

parser.add_argument('--ffptsfile', help='flipptsfile', required=True, type=str)
parser.add_argument('--fptsfile', help='fptsfile', required=True, type=str)
parser.add_argument('--hptsfile', help='hptsfile', required=True, type=str)
parser.add_argument('--heightfile', help='heightfile', required=True, type=str)
parser.add_argument('--l-threshold', default=0.18, type=float, help='threshold')
parser.add_argument('--f-threshold', default=3, type=float, help='threshold')       

parser.add_argument('--update-steps', default=100, type=int, help='number of total steps to run')


parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./checkpoint3', type=str, help='save path')
parser.add_argument('--dataset', default='animalweb', type=str,
                    choices=['cifar10', 'cifar100','animalweb'], help='dataset name')
parser.add_argument('--total-steps', default=5000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=500, type=int, help='number of eval steps to run')
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

parser.add_argument("--amp", action="store_true",default=False, help="use 16-bit (mixed) precision")
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

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = torch.cat([get_rot_mat(t)[None, ...].type(dtype) for t in theta])
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x

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


def train_loop(cfg,args, labeled_loader, unlabeled_loader, test_loader,
               teacher_model, student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}")
    logger.info(f"   Total steps = {args.total_steps}")

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    loader = dataloader_gen(cfg, args)
    pseudo_dict = {}

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            images_l, targets,_ = labeled_iter.next()
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            images_l, targets,_ = labeled_iter.next()

        try:
            images_uw, _,meta, pseudo = unlabeled_iter.next()
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            images_uw, _,meta, pseudo = unlabeled_iter.next()

        data_time.update(time.time() - end)

        images_l = images_l.to(args.device)
        images_uw = images_uw.to(args.device)
        targets = targets.to(args.device)
        with amp.autocast(enabled=args.amp):
            batch_size = images_l.shape[0]
            t_images = torch.cat((images_l, images_uw))
            t_logits = teacher_model(t_images)
            t_logits = t_logits.cuda(non_blocking = True)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw = t_logits[batch_size:]

            t_loss_l = criterion(t_logits_l, targets) + criterion(t_logits_l[5:], targets[5:])

            pseudo_label = t_logits.detach()
            s_images = torch.cat((images_l, images_uw))
            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_uw = s_logits[batch_size:]

            s_loss_l_old = criterion(s_logits_l.detach(),targets)
            s_loss = criterion(s_logits, pseudo_label)

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()


        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(images_l)
            s_loss_l_new = criterion(s_logits_l.detach(), targets)
            dot_product = s_loss_l_old - s_loss_l_new
            
            maxidx = t_logits_uw.detach().view(t_logits_uw.size(0),t_logits_uw.size(1),-1).argmax(2)
            indices = torch.cat(((maxidx / t_logits_uw.size(2)).view(t_logits_uw.size(0), t_logits_uw.size(1),1).int(),\
                (maxidx % t_logits_uw.size(3)).view(t_logits_uw.size(0), t_logits_uw.size(1),1).int()), dim=2)
            hard_pseudo_label = np.zeros((t_logits_uw.size(0),t_logits_uw.size(1), t_logits_uw.size(2), t_logits_uw.size(3)))
            
            for batch in range(t_logits_uw.size(0)):
                for i in range(t_logits_uw.size(1)):
                    if indices[batch,i, 1] > 0:
                        hard_pseudo_label[batch,i] = generate_target_inverse(hard_pseudo_label[batch,i], indices[batch,i], 1)
            hard_pseudo_label = torch.from_numpy(hard_pseudo_label)
            hard_pseudo_label = hard_pseudo_label.to(torch.float32)
            hard_pseudo_label = hard_pseudo_label.to(args.device)

            t_loss_mpl = dot_product * criterion(t_logits_uw,hard_pseudo_label)
            t_loss = t_loss_l + t_loss_mpl
            
            
        
        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            # mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_mpl.update(t_loss_mpl.item())

        ###################### Label Pool###################################
        preds = decode_preds(s_logits_uw.data.cpu(), meta['center'], meta['scale'], [64, 64])
        
        for i,m in enumerate(pseudo):
            if m == 1:
                pseudo_dict[meta['image'][i]] = preds.numpy()[i]
        
        ##################################################################
        if (step+1) %args.update_steps == 0:
            labeled_loader = loader.run('label',pseudo_dict=pseudo_dict)

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. "
            f"Pseudo_Label: {len(pseudo_dict)}")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)
            wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step//args.eval_step
        if (step+1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                wandb.log({"train/1.s_loss": s_losses.avg,
                           "train/2.t_loss": t_losses.avg,
                           "train/3.t_labeled": t_losses_l.avg,
                           "train/5.t_mpl": t_losses_mpl.avg,
                           "train/6.mask": mean_mask.avg})

                test_model = student_model
                test_loss, nme = evaluate(args, test_loader, test_model, criterion)

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/nme", nme, args.num_eval)
                wandb.log({"test/loss": test_loss,
                           "test/nme": nme})

                is_best = nme < args.nme
                if is_best:
                    args.nme = nme

                logger.info(f"nme: {nme:.5f}")
                logger.info(f"Best nme: {args.nme:.5f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'best_nme': args.nme,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)



def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0

    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets,meta) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            outputs = outputs.data.cpu()
            preds = decode_preds(outputs, meta['center'], meta['scale'], [64, 64])

            nme_temp = compute_nme(preds,meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            
            nme_avg = np.sum(nme_temp)/preds.size(0)

            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"nme: {nme_avg:.2f}.")

        nme = nme_batch_sum / nme_count
        failure_008_rate = count_failure_008 / nme_count
        failure_010_rate = count_failure_010 / nme_count
        test_iter.close()
        return losses.avg, nme

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
    
    
    csvfile = open(f'/home/cseadmin/hsc/hrnet/HRNet/tools/species/test_nme_{args.name}.csv', 'w',newline='')
    print(f'test_result/test_{args.name}.csv')
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
            csv_write.writerow([meta['image'][0],nme_avg])
            
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
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
    logger.info(msg)
    csvfile.close()
    return nme, predictions


def vis(args,config, data_loader, model):
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))
    model.eval()
    total = len(data_loader)
    test_iter = tqdm(data_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):  
            inp = inp.to(args.device)
            output = model(inp)
            score_map = output
            score_map = score_map.data.cpu()
            # vis = output[:,9:,:,:]
            
            preds = decode_preds(score_map, meta['center'], meta['scale'], config.MODEL.HEATMAP_SIZE)
            print(meta['image'][0])
            # jointv = vis.to("cpu").numpy()[0]
            # joint_vis = []
            # for j in range(jointv.shape[0]):
            #     joint_vis.append(np.mean(jointv[j,]).item())
            nme_temp = compute_nme(preds, meta)
            img = cv2.imread(os.path.join(config.DATASET.ROOT,meta['image'][0]))
            pts = read_pts(os.path.join(config.DATASET.ROOT,
                                  meta['image'][0]).replace('jpg','pts'))
            preds = preds.tolist()[0]
            for pt in pts:
                if pt[0] != 0 and pt[1] != 0:
                    cv2.circle(img,(int(pt[0]),int(pt[1])),3,(0,0,255),-1)
            color = [   (255,0,0),
                     (255,128,0),
                     (255,255,0),
                     (0,255,0),
                     (0,255,255),
                     (0,0,255),
                     (128,0,255),
                     (128,0,128),
                     (255,255,255)
                    ]
            for idx,pt in enumerate(preds):
                # if joint_vis[idx] < 0.5:
                #     cv2.circle(img,(int(pt[0]),int(pt[1])),5,color[idx],-1)
                # else:
                    cv2.circle(img,(int(pt[0]),int(pt[1])),5,color[idx])
                
                    
            
            cv2.imwrite('image_result/model_v2/'+meta['image'][0],img)
        
            print('{}/{}==>'.format(i,total)+meta['image'][0],' nme = ',nme_temp)


def finetune(args, train_loader, test_loader, model, criterion):
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        train_loader.dataset,
        sampler=train_sampler(train_loader.dataset),
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        pin_memory=True)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.finetune_lr,
                          momentum=args.finetune_momentum,
                          weight_decay=args.finetune_weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader)*args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch+624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets,_) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                model.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.world_size > 1:
                loss = reduce_tensor(loss.detach(), args.world_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss, nme = evaluate(args, test_loader, model, criterion)
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
            args.writer.add_scalar("finetune/nme", nme, epoch)
            wandb.log({"finetune/train_loss": losses.avg,
                       "finetune/test_loss": test_loss,
                       "finetune/nme": nme,
                       })

            is_best = nme < args.nme
            if is_best:
                args.nme = nme

            logger.info(f"nme: {nme:.2f}")
            logger.info(f"Best nme: {args.nme:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'nme': args.nme,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("result/finetune_acc@1", args.best_top1)
            wandb.log({"result/fintune_acc@1": args.best_top1})
    return

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

    labeled_dataset, unlabeled_dataset, test_dataset = AnimalWeb(config, args, 'label'), AnimalWeb(config, args, 'unlabel'), AnimalWeb(config, args, 'test')
    exit()
    if args.local_rank == 0:
        torch.distributed.barrier()

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=4*config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    exit(-1)

    teacher_model = hrnet.get_face_alignment_net(config)
    student_model = hrnet.get_face_alignment_net(config)

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")

    teacher_model.to(args.device)
    student_model.to(args.device)
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    no_decay = ['bn']
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
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         logger.info(f"=> loading checkpoint '{args.resume}'")
    #         loc = f'cuda:{args.gpu}'
    #         checkpoint = torch.load(args.resume, map_location=loc)
    #         # args.nme = checkpoint['best_nme']
    #         if not (args.evaluate or args.finetune or args.vis):
    #             args.start_step = checkpoint['step']
    #             t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
    #             s_optimizer.load_state_dict(checkpoint['student_optimizer'])
    #             t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
    #             s_scheduler.load_state_dict(checkpoint['student_scheduler'])
    #             t_scaler.load_state_dict(checkpoint['teacher_scaler'])
    #             s_scaler.load_state_dict(checkpoint['student_scaler'])
    #             model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
    #             model_load_state_dict(student_model, checkpoint['student_state_dict'])
    #             # if avg_student_model is not None:
    #             #     model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

    #         else:
    #             model_load_state_dict(student_model, checkpoint['state_dict'])#

    #         logger.info(f"=> loaded checkpoint '{args.resume}'")
    #     else:
    #         logger.info(f"=> no checkpoint found at '{args.resume}'")
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            
            model_load_state_dict(student_model, checkpoint['state_dict'])
            model_load_state_dict(teacher_model, checkpoint['state_dict'])
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

    if args.vis:
        vis(args, config,test_loader,student_model)
        return

    if args.finetune:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
        del s_scaler, s_scheduler, s_optimizer
        finetune(args, labeled_loader, test_loader, student_model, criterion)
        return

    if args.evaluate:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        inference(args, test_loader, student_model, criterion)
        return

    teacher_model.zero_grad()
    student_model.zero_grad()

    train_loop(config,args, labeled_loader, unlabeled_loader, test_loader,
               teacher_model, student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler)
    return


if __name__ == '__main__':
    main()
