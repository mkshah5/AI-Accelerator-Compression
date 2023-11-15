import argparse
import os
from distutils.util import strtobool
from typing import Tuple
import time
import sys
from pathlib import Path
sys.path.append("../")

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

import poptorch

from utils.utils import TrainingParams
from compressor.compress_entry_f32 import compress, decompress, get_lhs_rhs_decompress
from data_utils_graphcore import get_data_generator

from model_gc import CloudMaskNet

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

DATA_DIR = "/home/shahm/sciml_bench/datasets/cloud_slstr_ds1"
BENCHMARK_NAME = "cloudmask"
VERSION = "graphcore"

# Dependent on the number of channels
def full_comp(x):
    s = x.shape
    r = compress(torch.reshape(torch.squeeze(x[:,0,:,:], (TRAIN_SIZE,s[2],s[3]))),PARAMS)
    g = compress(torch.reshape(torch.squeeze(x[:,1,:,:], (TRAIN_SIZE,s[2],s[3]))),PARAMS)
    b = compress(torch.reshape(torch.squeeze(x[:,2,:,:], (TRAIN_SIZE,s[2],s[3]))),PARAMS)
    r1 = compress(torch.reshape(torch.squeeze(x[:,3,:,:], (TRAIN_SIZE,s[2],s[3]))),PARAMS)
    g1 = compress(torch.reshape(torch.squeeze(x[:,4,:,:], (TRAIN_SIZE,s[2],s[3]))),PARAMS)
    b1 = compress(torch.reshape(torch.squeeze(x[:,5,:,:], (TRAIN_SIZE,s[2],s[3]))),PARAMS)
    r2 = compress(torch.reshape(torch.squeeze(x[:,6,:,:], (TRAIN_SIZE,s[2],s[3]))),PARAMS)
    g2 = compress(torch.reshape(torch.squeeze(x[:,7,:,:], (TRAIN_SIZE,s[2],s[3]))),PARAMS)
    b2 = compress(torch.reshape(torch.squeeze(x[:,8,:,:], (TRAIN_SIZE,s[2],s[3]))),PARAMS)

    out = torch.stack((r,g,b,r1,g1,b1,r2,g2,b2),1)
    return out


class CloudMaskCompress(nn.Module):
    def __init__(self):
        super(CloudMaskCompress, self).__init__()

        self.criterion = nn.BCELoss()
        self.decompress = decompress

        self.internal_model = CloudMaskNet((RPIX,CPIX,PARAMS.nchannels))
    # assume bs > 1
    def forward(self, x, lhs, rhs, gt=None):
        s = x.shape
        r = decompress(torch.reshape(torch.squeeze(x[:,0,:,:], (TRAIN_SIZE,s[2],s[3]))), lhs,rhs)
        g = decompress(torch.reshape(torch.squeeze(x[:,1,:,:], (TRAIN_SIZE,s[2],s[3]))), lhs,rhs)
        b = decompress(torch.reshape(torch.squeeze(x[:,2,:,:], (TRAIN_SIZE,s[2],s[3]))), lhs,rhs)
        r1 = decompress(torch.reshape(torch.squeeze(x[:,3,:,:], (TRAIN_SIZE,s[2],s[3]))), lhs,rhs)
        g1 = decompress(torch.reshape(torch.squeeze(x[:,4,:,:], (TRAIN_SIZE,s[2],s[3]))), lhs,rhs)
        b1 = decompress(torch.reshape(torch.squeeze(x[:,5,:,:], (TRAIN_SIZE,s[2],s[3]))), lhs,rhs)
        r2 = decompress(torch.reshape(torch.squeeze(x[:,6,:,:], (TRAIN_SIZE,s[2],s[3]))), lhs,rhs)
        g2 = decompress(torch.reshape(torch.squeeze(x[:,7,:,:], (TRAIN_SIZE,s[2],s[3]))), lhs,rhs)
        b2 = decompress(torch.reshape(torch.squeeze(x[:,8,:,:], (TRAIN_SIZE,s[2],s[3]))), lhs,rhs)
        o = torch.stack((r,g,b,r1,g1,b1,r2,g2,b2),1)

        o = torch.reshape(o, (TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX))

        out = self.internal_model(o)
        if self.training:
            loss = self.criterion(out, gt)
            return out, loss
        return out
    
class CloudMaskBase(nn.Module):
    def __init__(self):
        super(CloudMaskBase, self).__init__()
        self.criterion = nn.BCELoss()
        self.internal_model = CloudMaskNet((RPIX,CPIX,PARAMS.nchannels))

    # assume bs > 1
    def forward(self, x, gt=None):
        out = self.internal_model(x)
        if self.training:
            loss = self.criterion(out, gt)
            return out, loss
        return out

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--config_path', type=str, default='./config-ch4.txt')
    parser.add_argument('--command',type=str,default='train')

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


def train(args: argparse.Namespace, model: nn.Module, optimizer: poptorch.optim.SGD) -> None:
    
    opts_t = poptorch.Options()
    opts_t.deviceIterations(1)
    opts_t.setAvailableMemoryProportion({"IPU0": 0.1});
    opts_t.Precision.setPartialsType(torch.half)
    opts_v = poptorch.Options()
    opts_v.deviceIterations(1)

    opts_v.setAvailableMemoryProportion({"IPU0": 0.1});

    opts_v.Precision.setPartialsType(torch.half)

    train_loader, test_loader = get_data_generator(DATA_DIR,PARAMS,opts_t,opts_v)

    total_step = len(train_loader)
    
    h_l = nn.BCELoss()
    
    lhs, rhs = get_lhs_rhs_decompress(PARAMS)        
    lhs = torch.as_tensor(lhs).to(torch.float32)
    rhs = torch.as_tensor(rhs).to(torch.float32)
    
    poptorch_model = poptorch.trainingModel(model,options=opts_t,optimizer=optimizer)
    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            #print(images.shape)
            
            images = torch.mul(images, 255).to(torch.float32)
            labels = labels.to(torch.float32)
            
            if not IS_BASELINE_NETWORK:
                images = full_comp(images)
            print(images.shape)
            run_start = time.time()
            outputs,loss = poptorch_model(images,lhs,rhs,labels)
            run_end = time.time()
            avg_loss += loss.mean()
            run_time = run_end-run_start
            print("===Timing===")
            print("Step Run Time (s): "+str(run_time))
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1, total_step,
                                                                     avg_loss / (i + 1)))
        poptorch_model.detachFromDevice()
        torch.save(model.state_dict(), MODEL_NAME+".pt")
        model = model.eval()
        poptorch_model_inf = poptorch.inferenceModel(model, options=opts_v)
        test_acc = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for images, labels in test_loader:
                
                labels = labels.to(torch.float32)
                images = torch.mul(images, 255)
                if not IS_BASELINE_NETWORK:
                    images = full_comp(images)

                outputs = poptorch_model_inf(images)

                loss = h_l(outputs,labels)
                total_loss += loss.mean()

            test_acc = 0
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))
        poptorch_model_inf.detachFromDevice()
    torch.save(model, MODEL_NAME+".pt")

def main():
    global PARAMS, TRAIN_SIZE, TEST_SIZE, CF, RPIX, CPIX, BD, RBLKS, CBLKS, IS_BASELINE_NETWORK, MODEL_NAME

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

    if IS_BASELINE_NETWORK:
        MODEL_NAME = VERSION+"_base_"+BENCHMARK_NAME
    else:
        MODEL_NAME = VERSION+"_matmul_"+BENCHMARK_NAME+"_cf"+str(CF)
    
    if args.command == "test":
        model = torch.load(MODEL_NAME+".pt")
    else:
        if IS_BASELINE_NETWORK:
            model = EMDenoiseBase()
        else:
            model = EMDenoiseCompress()

    print(args)

    optimizer = poptorch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(args, model, optimizer)


if __name__ == '__main__':
    main()
