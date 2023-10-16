#!/usr/bin/env python
from pprint import pprint
from pathlib import Path
#from libpressio import PressioCompressor
import sys
sys.path.append("../")

import argparse
import os
from distutils.util import strtobool
from typing import Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from torchvision import datasets

import matplotlib.pyplot as plt

from torch import Tensor
from typing import Type

from utils.utils import TrainingParams
from compressor.compress_entry import compress, decompress, get_lhs_rhs_decompress
from config import PARAMS
from model import CloudMaskNet
from data_utils import get_data_generator

TRAIN_SIZE = PARAMS.batch_size
TEST_SIZE = PARAMS.batch_size
CF = PARAMS.cf

RPIX = PARAMS.rpix
CPIX = PARAMS.cpix
BD = PARAMS.BD
RBLKS = PARAMS.rblks
CBLKS = PARAMS.cblks

IS_BASELINE_NETWORK = PARAMS.is_base

if IS_BASELINE_NETWORK:
    MODEL_NAME = "torch_base_cloudmask"
else:
    MODEL_NAME = "torch_matmul_cloudmask_cf"+str(CF)

COMPRESSOR_PATH = "/home/mkshah5/SZ/build/bin/sz"

DATA_DIR = "/home/shahm/sciml_bench/datasets/cloud_slstr_ds1"

# Dependent on the number of channels
def full_comp(x, err=2.7e-3):
    fshape = str(TRAIN_SIZE)+" "+str(PARAMS.nchannels)+" "+str(RPIX)+" "+str(CPIX)
    n_x = x.numpy().astype(np.float32)
    n_x.tofile('tmpb.bin')
    c_command = COMPRESSOR_PATH+" -z -f -M ABS -A "+str(err)+" -i tmpb.bin -4 "+fshape
    d_command = COMPRESSOR_PATH+" -x -f -s tmpb.bin.sz -4 "+fshape+" -i tmpb.bin"
    os.system(c_command)
    cr = (os.stat('tmpb.bin').st_size)/os.stat('tmpb.bin.sz').st_size
    print("CR: "+str(cr))
    os.system(d_command)
    decomp = np.fromfile('tmpb.bin.sz.out',dtype=np.float32)
    decomp = np.reshape(decomp, (TRAIN_SIZE,PARAMS.nchannels,RPIX,CPIX))
    return torch.from_numpy(decomp)


class CloudMaskCompress(nn.Module):
    def __init__(self):
        super(CloudMaskCompress, self).__init__()

        self.internal_model = CloudMaskNet((RPIX,CPIX,PARAMS.nchannels))

    # assume bs > 1
    def forward(self, x):

        out = self.internal_model(x)
        return out
    
class CloudMaskBase(nn.Module):
    def __init__(self):
        super(CloudMaskBase, self).__init__()
        self.internal_model = CloudMaskNet((RPIX,CPIX,PARAMS.nchannels))

    # assume bs > 1
    def forward(self, x):
        out = self.internal_model(x)
        return out

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--device',type=int,default=0)

def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument('--train-torch', action='store_true', help='train FP32 PyTorch version of application')
    parser.add_argument('-n', '--num-iterations', type=int, default=100, help='Number of iterations to run the pef for')
    parser.add_argument('-e', '--num-epochs', type=int, default=50)
    parser.add_argument('--log-path', type=str, default='checkpoints')
    parser.add_argument('--dump-interval', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--measure-train-performance', action='store_true')
    parser.add_argument('--acc-test', action='store_true', help='Option for accuracy guard test in RDU regression.')
    parser.add_argument('--data-dir',
                        type=str,
                        default='mnist_data',
                        help="The folder to download the MNIST dataset to.")


def train(args: argparse.Namespace, model: nn.Module, optimizer,device) -> None:
    train_loader, test_loader = get_data_generator(DATA_DIR)

    loss_function = nn.BCELoss()
    # Train the model
    model = model.cuda()
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            
            labels = labels.cuda()
            if not IS_BASELINE_NETWORK:
                images = full_comp(images)
            
            
            images = torch.mul(images, 255)
            images = images.cuda()
            run_start = time.time()
                    
            outputs = model(images)
            
            loss = loss_function(outputs,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            run_end = time.time()
            avg_loss += loss.mean()
            run_time = run_end-run_start
            print("===Timing===")
            print("Step Run Time (s): "+str(run_time))
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1, total_step,
                                                                     avg_loss / (i + 1)))
        test_acc = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for images, labels in test_loader:
            
                if not IS_BASELINE_NETWORK:
                    images = full_comp(images)
                images = torch.mul(images, 255)
                images.to(device)
                outputs = model(images)
                loss = loss_function(outputs,labels)
                
                total_loss += loss.mean()


            test_acc = 100.0 * correct / total
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))
    
    torch.save(model, MODEL_NAME+".pt")

def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_run_args(parser)
    args = parser.parse_args()
    command = "train"
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if command == "test":
        model = torch.load(MODEL_NAME+".pt").to(device)
    else:
        if IS_BASELINE_NETWORK:
            model = CloudMaskBase().to(device)
        else:
            model = CloudMaskCompress().to(device)

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    train(args, model, optimizer,device)


if __name__ == '__main__':
    main()
