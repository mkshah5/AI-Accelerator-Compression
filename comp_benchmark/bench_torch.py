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
import pandas as pd

from torch import Tensor
from typing import Type

from utils.utils import TrainingParams
from compressor.compress_entry import compress, decompress, get_lhs_rhs_decompress
from utils.gpu_stats import GPUStats


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

BENCHMARK_NAME = "standalone"
VERSION = "torch"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dct_compress(x):
    r = compress(torch.squeeze(x[:,0,:,:]), PARAMS)
    g = compress(torch.squeeze(x[:,1,:,:]), PARAMS)
    b = compress(torch.squeeze(x[:,2,:,:]), PARAMS)
    x = torch.stack((r,g,b),1)
    return x

def dct_decompress(x, lhs, rhs):
    r = decompress(torch.squeeze(x[:,0,:,:]), lhs,rhs)
    g = decompress(torch.squeeze(x[:,1,:,:]), lhs,rhs)
    b = decompress(torch.squeeze(x[:,2,:,:]), lhs,rhs)
    o = torch.stack((r,g,b),1)

    return o.to(torch.float32)


class CompressorModel(nn.Module):
    def __init__(self):
        super(CompressorModel, self).__init__()

        lhs, rhs = get_lhs_rhs_decompress(PARAMS)
        self.lhs = torch.from_numpy(lhs).to(torch.bfloat16)
        self.rhs = torch.from_numpy(rhs).to(torch.bfloat16)
        self.comp_op = dct_decompress
    # assume bs > 1
    def forward(self, x):

        out = self.comp_op(x, self.lhs, self.rhs)
        return out
    

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--config_path', type=str, default='./config-ch4.txt')
    parser.add_argument('--compressor', type=str,default='dct')
    parser.add_argument('--num_iterations', type=int,default=10)



def main():
    global COMPRESSOR, PARAMS, TRAIN_SIZE, TEST_SIZE, CF, RPIX, CPIX, BD, RBLKS, CBLKS, IS_BASELINE_NETWORK, MODEL_NAME, GPUSTATS

    parser = argparse.ArgumentParser()
    add_common_args(parser)
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
        MODEL_NAME = VERSION+"_"+COMPRESSOR+"_"+BENCHMARK_NAME+"_cf"+str(CF)
    command = "train"
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = CompressorModel().to(device)

    inputs = torch.rand(TRAIN_SIZE, 3, RPIX, CPIX)
    inputs = dct_compress(inputs)
    for i in range(args.num_iterations):
        s1 = time.time()
        inputs = inputs.to(device)
        out = model(inputs)
        out = out.to('cpu')
        s2 = time.time()
        print("Step: "+str(i)+", Time(s): "+str(s2-s1))
        inputs = inputs.to('cpu')



if __name__ == '__main__':
    main()
