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
from compressor.compress_entry_gc import compress, decompress, get_lhs_rhs_decompress, get_idx_array
from utils.gpu_stats import GPUStats

# from config import PARAMS
from model import EMDenoiseNet
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

BENCHMARK_NAME = "emdenoise"
VERSION = "torchdiagonal"

COMPRESSOR_PATH = "/home/mkshah5/SZ/build/bin/sz"

DATA_DIR = "/home/mkshah5/sciml_bench/datasets/em_graphene_sim"

# Dependent on the number of channels
def sz_comp(x, err=9.6e-4):
    fshape = str(TRAIN_SIZE)+" "+str(PARAMS.nchannels)+" "+str(RPIX)+" "+str(CPIX)
    print(fshape)
    print(x.shape)
    n_x = x.numpy().astype(np.float32)
    n_x.tofile('tmpb.bin')
    c_command = COMPRESSOR_PATH+" -z -f -M ABS -A "+str(err)+" -i tmpb.bin -4 "+fshape
    d_command = COMPRESSOR_PATH+" -x -f -s tmpb.bin.sz -3 32 256 256 -i tmpb.bin"
    os.system(c_command)
    cr = (os.stat('tmpb.bin').st_size)/os.stat('tmpb.bin.sz').st_size
    print("CR: "+str(cr))
    os.system(d_command)
    decomp = np.fromfile('tmpb.bin.sz.out',dtype=np.float32)
    decomp = np.reshape(decomp, (TRAIN_SIZE,PARAMS.nchannels,RPIX,CPIX))
    return torch.from_numpy(decomp), os.stat('tmpb.bin.sz').st_size

# def dct_comp(x):
#     out = compress(torch.squeeze(x),PARAMS)
#     out = torch.reshape(out, (TRAIN_SIZE, PARAMS.nchannels, out.shape[1], out.shape[2]))
    
#     lhs, rhs = get_lhs_rhs_decompress(PARAMS)
#     lhs = torch.from_numpy(lhs).to(torch.bfloat16)
#     rhs = torch.from_numpy(rhs).to(torch.bfloat16)

#     o = decompress(out,lhs,rhs)
#     o = torch.reshape(o, (TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX))

#     return o.to(torch.float32), out.element_size() * out.nelement()

def dct_comp(x):
    out = compress(torch.squeeze(x),PARAMS)
    out = torch.reshape(out, (TRAIN_SIZE, PARAMS.nchannels, -1))
    
    lhs, rhs = get_lhs_rhs_decompress(PARAMS)
    lhs = torch.from_numpy(lhs).to(torch.float32)
    rhs = torch.from_numpy(rhs).to(torch.float32)
    idx = torch.as_tensor(get_idx_array(PARAMS)).to(torch.int64)
    cf_nrows = int(RBLKS*CF)
    cf_ncols = int(CBLKS*CF)

    # o = decompress(out,lhs,rhs)
    o = decompress(torch.squeeze(out[:,0,:]), lhs,rhs, idx, cf_nrows, cf_ncols)
    o = torch.reshape(o, (TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX))

    return o.to(torch.float32), out.element_size() * out.nelement()

def full_comp(x):
    if COMPRESSOR=="dct":
        c_res, size = dct_comp(torch.mul(x,255).to(torch.float32))
        return c_res, size
    elif COMPRESSOR=="sz":
        c_res, size = sz_comp(x.to(torch.float32))
        return torch.mul(c_res,255),size

class EMDenoiseCompress(nn.Module):
    def __init__(self):
        super(EMDenoiseCompress, self).__init__()

        self.internal_model = EMDenoiseNet()

    # assume bs > 1
    def forward(self, x):

        out = self.internal_model(x)
        return out
    
class EMDenoiseBase(nn.Module):
    def __init__(self):
        super(EMDenoiseBase, self).__init__()
        self.internal_model = EMDenoiseNet()

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
    stats = GPUStats(MODEL_NAME)
    train_loader = get_data_generator(Path(DATA_DIR) / 'train', TRAIN_SIZE, is_inference=False)
    test_loader = get_data_generator(Path(DATA_DIR) / 'inference', TRAIN_SIZE, is_inference=True)
    loss_function = nn.MSELoss()
    # Train the model
    model = model.cuda()
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        avg_loss = 0
        avg_step_time = 0.0
        avg_osize = 0.0
        avg_csize = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = torch.permute(images, (0, 3, 1, 2))
            labels = torch.permute(labels, (0, 3, 1, 2))
            
            avg_osize += images.element_size() * images.nelement()
            if not IS_BASELINE_NETWORK:
                images,size = full_comp(images)
                avg_csize += size
            
            run_start = time.time()
            labels = labels.cuda()
            images = images.cuda()
            outputs = model(images)
            
            loss = loss_function(outputs,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            run_end = time.time()
            avg_loss += loss.mean()
            run_time = run_end-run_start
            avg_step_time += run_time
            print("===Timing===")
            print("Step Run Time (s): "+str(run_time))
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1, total_step,
                                                                     avg_loss / (i + 1)))
        stats.add_stat("train_loss", avg_loss.item()/total_step)
        stats.add_stat("avg_step_time", avg_step_time/total_step)
        stats.add_stat("original_size_bytes", avg_osize/total_step)
        stats.add_stat("compressed_size_bytes", avg_csize/total_step)
        
        test_acc = 0.0
        test_steps = len(test_loader)
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for images, labels in test_loader:
                
                #images = torch.permute(images, (0, 3, 1, 2))
                #labels = torch.permute(labels, (0, 3, 1, 2))
                #print(images.shape)
                #print(labels.shape)
                if not IS_BASELINE_NETWORK:
                    images,size = full_comp(images)
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                loss = loss_function(outputs,labels)
                
                total_loss += loss.mean()


            #test_acc = 100.0 * correct / total
            stats.add_stat("test_loss", total_loss.item()/test_steps)
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))
        stats.register_epoch_row_and_update()
    stats.save_df()
    torch.save(model, MODEL_NAME+".pt")

def main():
    global COMPRESSOR,PARAMS, TRAIN_SIZE, TEST_SIZE, CF, RPIX, CPIX, BD, RBLKS, CBLKS, IS_BASELINE_NETWORK, MODEL_NAME

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
    COMPRESSOR = args.compressor

    IS_BASELINE_NETWORK = PARAMS.is_base

    if IS_BASELINE_NETWORK:
        MODEL_NAME = VERSION+"_base_"+BENCHMARK_NAME
    else:
        MODEL_NAME = VERSION+"_matmul_"+BENCHMARK_NAME+"_cf"+str(CF)
    command = "train"
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if command == "test":
        model = torch.load(MODEL_NAME+".pt").to(device)
    else:
        if IS_BASELINE_NETWORK:
            model = EMDenoiseBase().to(device)
        else:
            model = EMDenoiseCompress().to(device)

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    train(args, model, optimizer,device)


if __name__ == '__main__':
    main()
