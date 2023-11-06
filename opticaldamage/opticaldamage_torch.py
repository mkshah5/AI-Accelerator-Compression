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
from model import OpticalDamageNet
from data_utils import get_data_generator

PARAMS = None
TRAIN_SIZE = None
TEST_SIZE = None
CF = None
RPIX = None
CPIX = None
BD = None
RBLKS = None
CBLKS = None
IS_BASELINE_NETWORK = None
MODEL_NAME = None
COMPRESSOR = "dct"

BENCHMARK_NAME = "opticaldamage"
VERSION = "torch"

DATA_DIR = "/home/mkshah5/sciml_bench/datasets/optical_damage_ds1"

COMPRESSOR_PATH = "/home/mkshah5/SZ/build/bin/sz"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# Dependent on the number of channels
def sz_comp(x, err=2.5e-4):
    fshape = str(TRAIN_SIZE)+" "+str(RPIX)+" "+str(CPIX)
    n_x = x.numpy().astype(np.float32)
    n_x.tofile('tmpb.bin')
    c_command = COMPRESSOR_PATH+" -z -f -M ABS -A "+str(err)+" -i tmpb.bin -3 "+fshape
    d_command = COMPRESSOR_PATH+" -x -f -s tmpb.bin.sz -3 "+fshape+" -i tmpb.bin"
    os.system(c_command)
    cr = (os.stat('tmpb.bin').st_size)/os.stat('tmpb.bin.sz').st_size
    print("CR: "+str(cr))
    os.system(d_command)
    decomp = np.fromfile('tmpb.bin.sz.out',dtype=np.float32)
    decomp = np.reshape(decomp, (TRAIN_SIZE,PARAMS.nchannels,RPIX,CPIX))
    return torch.from_numpy(decomp)

def dct_comp(x):
    out = compress(torch.squeeze(x),PARAMS)
    out = torch.reshape(out, (TRAIN_SIZE, PARAMS.nchannels, out.shape[1], out.shape[2]))
    
    lhs, rhs = get_lhs_rhs_decompress(PARAMS)
    
    o = decompress(out,lhs,rhs)
    o = torch.reshape(o, (TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX))

    return o.to(torch.float32)

def full_comp(x):
    if COMPRESSOR=="dct":
        return dct_comp(torch.mul(x,255))
    elif COMPRESSOR=="sz":
        return torch.mul(sz_comp(x),255)

class OpticalDamageCompress(nn.Module):
    def __init__(self):
        super(OpticalDamageCompress, self).__init__()

        self.internal_model = OpticalDamageNet((RPIX, CPIX, PARAMS.nchannels))

    # assume bs > 1
    def forward(self, x):

        out = self.internal_model(x)
        return out
    
class OpticalDamageBase(nn.Module):
    def __init__(self):
        super(OpticalDamageBase, self).__init__()
        self.internal_model = OpticalDamageNet((RPIX, CPIX, PARAMS.nchannels))

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
    parser.add_argument('--config_path', type=str, default='./config-ch4.txt')
    parser.add_argument('--compressor', type=str,default='dct')

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
    train_loader = get_data_generator(Path(DATA_DIR), TRAIN_SIZE, is_inference=False)
    test_loader = get_data_generator(Path(DATA_DIR), TRAIN_SIZE, is_inference=True)
    loss_function = nn.MSELoss()

    #torch.set_default_device('cuda:1')
    # Train the model
    model = model.cuda()
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (images) in enumerate(train_loader):
            gt = images.to(torch.float32)
            gt = gt.cuda()
            if not IS_BASELINE_NETWORK:
                images = full_comp(images)
            
            images = images.to(torch.float32)
            images = images.cuda()
            run_start = time.time()
                    
            outputs = model(images)
            
            loss = loss_function(outputs,gt)

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
            for i,(images) in enumerate(test_loader):
                gt = images.to(torch.float32)
                images = images.to(torch.float32)
                if not IS_BASELINE_NETWORK:
                    images = full_comp(images)
                images.to(device)
                images = images.cuda()
                gt = gt.cuda()
                outputs = model(images)
                loss = loss_function(outputs,gt)
                
                total_loss += loss.mean()


            #test_acc = 100.0 * correct / total
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))
    
    torch.save(model, MODEL_NAME+".pt")

def main():
    global PARAMS, TRAIN_SIZE, TEST_SIZE, CF, RPIX, CPIX, BD, RBLKS, CBLKS, IS_BASELINE_NETWORK, MODEL_NAME, COMPRESSOR

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_run_args(parser)
    args = parser.parse_args()
    PARAMS = TrainingParams(args.config_path)
    TRAIN_SIZE = PARAMS.batch_size
    TEST_SIZE = PARAMS.batch_size
    CF = PARAMS.cf

    RPIX = PARAMS.rpix
    CPIX = PARAMS.cpix
    BD = PARAMS.BD
    RBLKS = PARAMS.rblks
    CBLKS = PARAMS.cblks

    IS_BASELINE_NETWORK = PARAMS.is_base
    COMPRESSOR = args.compressor


    if IS_BASELINE_NETWORK:
        MODEL_NAME = VERSION+"_base_"+BENCHMARK_NAME
    else:
        MODEL_NAME = VERSION+"_matmul_"+BENCHMARK_NAME+"_cf"+str(CF)
    command = "train"
    device = "cuda:1"
    device = torch.device(device)
    torch.cuda.set_device(device)
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    
    if command == "test":
        model = torch.load(MODEL_NAME+".pt").to(device)
    else:
        if IS_BASELINE_NETWORK:
            model = OpticalDamageBase().to(device)
        else:
            model = OpticalDamageCompress().to(device)

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    print("Num params: "+str(count_parameters(model)))
    train(args, model, optimizer,device)


if __name__ == '__main__':
    main()
