import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import matplotlib

from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch

import random


import sys

import argparse
from torch.backends import cudnn

sys.path.append("/home/yerinyoon/code/anonymousNet/data/celeba/")

from model import *
from data_extract import *

import wandb

parser = argparse.ArgumentParser()
def str2bool(v):
    return v.lower() in ('true')

def main(config):

    wandb.init(project="starGAN")
    wandb.config.update(config)


    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
         get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)
    

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, configi, wandb)
    wandb.watch(solver)
    if config.mode == 'train':

        if config.dataset in ['CelebA', 'RaFD']:
            solver.train(config)
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()

# Model configuration.
parser.add_argument('--c_dim', type=int, default=17, help='dimension of domain labels (1st dataset)')
parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
parser.add_argument('--image_size', type=int, default=128, help='image resolution')
parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

# Training configuration.
parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--num_iters', type=int, default=400000, help='number of total iterations for training D')
parser.add_argument('--num_iters_decay', type=int, default=50000, help='number of iterations for decaying lr')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=220000, help='resume training from this step')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                    default=['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Big_Lips',
                            'Big_Nose','Black_Hair','Bushy_Eyebrows','Chubby','Double_Chin',
                            'High_Cheekbones','Mustache','Narrow_Eyes','Oval_Face','Pale_Skin',
                            'Pointy_Nose','Sideburns','Straight_Hair'])
                    
                    # Bangs, Black_Hair, BushEyebrows
'''parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                    default=['Arched_Eyebrows', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Chubby', 'Double_Chin',
'High_Cheekbones', 'Narrow_Eyes', 'Oval_Face', 'Pointy_Nose', 'Sideburns'])
'''
parser.add_argument('--service_attrs', type=list, nargs='+', help='selected attributes for the CelebA dataset',
                    default=["Narrow_Eyes", "Pointy_Nose", "Gray_Hair", "Straight_Hair", "Bushy_Eyebrows", "PaleSkin"])
# Test configuration.
parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

# Miscellaneous.
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
parser.add_argument('--use_tensorboard', type=str2bool, default=True)

# Directories.
parser.add_argument('--celeba_image_dir', type=str, default="/home/yerinyoon/data/zip/data/celeba/img_align_celeba")
parser.add_argument('--attr_path', type=str, default="/home/yerinyoon/data/zip/data/celeba/list_attr_celeba.txt")
# parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
parser.add_argument('--log_dir', type=str, default='../')
parser.add_argument('--model_save_dir', type=str, default='/home/yerinyoon/code/anonymousNet/sevice_model_save_point')
parser.add_argument('--sample_dir', type=str, default='/home/yerinyoon/code/anonymousNet/service_model_save_point')
parser.add_argument('--result_dir', type=str, default='/home/yerinyoon/code/anonymousNet/service_model_save_point')

parser.add_argument('--service_model_save_dir', type=str, default='/home/yerinyoon/code/anonymousNet/service_model_save_point')

# Step size.
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=10000)
parser.add_argument('--lr_update_step', type=int, default=1000)

#model save
parser.add_argument("--output-prefix", default="model")
parser.add_argument('--resume-from', default="/home/yerinyoon/code/anonymousNet/service_model_save_point")
parser.add_argument('--input-dim', type=int)

parser.add_argument("--gpu_id", default=0)

config = parser.parse_args(args=[])
# print(config)


main(config)
