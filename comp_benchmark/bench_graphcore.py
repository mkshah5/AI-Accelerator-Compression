import argparse
import os
from distutils.util import strtobool
from typing import Tuple
import time
import sys
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

#import matplotlib.pyplot as plt

from torch import Tensor
from typing import Type

import poptorch

from utils.utils import TrainingParams
from compressor.compress_entry_f32 import compress, decompress, get_lhs_rhs_decompress

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

BENCHMARK_NAME = "standalone"
VERSION = "graphcore"

# Dependent on the number of channels
def full_comp(x):
    r = compress(torch.squeeze(x[:,0,:,:]), PARAMS)
    g = compress(torch.squeeze(x[:,1,:,:]), PARAMS)
    b = compress(torch.squeeze(x[:,2,:,:]), PARAMS)
    out = torch.stack((r,g,b),1)

    return out


class CompressorModel(nn.Module):
    def __init__(self):
        super(CompressorModel, self).__init__()

        self.decompress = decompress
        lhs, rhs = get_lhs_rhs_decompress(PARAMS)        
        self.lhs = torch.as_tensor(lhs).to(torch.float32)
        self.rhs = torch.as_tensor(rhs).to(torch.float32)

    # assume bs > 1
    def forward(self, x):
        r = decompress(torch.squeeze(x[:,0,:,:]), self.lhs.ipu(),self.rhs.ipu())
        g = decompress(torch.squeeze(x[:,1,:,:]), self.lhs.ipu(),self.rhs.ipu())
        b = decompress(torch.squeeze(x[:,2,:,:]), self.lhs.ipu(),self.rhs.ipu())
        out = torch.stack((r,g,b),1)

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


def prepare_fulldata(args: argparse.Namespace, opts_t) -> Tuple[torch.utils.data.DataLoader]:

    features = torch.randn([1000, PARAMS.nchannels, RPIX, CPIX])
    dataset = torch.utils.data.TensorDataset(features)
    # Create data loaders.

    train_loader = poptorch.DataLoader(
        options = opts_t,
        dataset=dataset, 
        batch_size=TRAIN_SIZE,
        shuffle=True,
        drop_last=True
    )


    return train_loader


def train(args: argparse.Namespace, model: nn.Module) -> None:
    
    opts_t = poptorch.Options()
    opts_t.deviceIterations(1)
    opts_t.setAvailableMemoryProportion({"IPU0": 0.25});
    opts_t.Precision.setPartialsType(torch.half)

    train_loader = prepare_fulldata(args,opts_t)

    model = model.eval()
    poptorch_model_inf = poptorch.inferenceModel(model, options=opts_t)
    with torch.no_grad():
        i = 0
        for images in train_loader:
            #print(images)
            images = torch.mul(images[0], 255)
        
            if not IS_BASELINE_NETWORK:
                images = full_comp(images)
            s1 = time.time()
            outputs = poptorch_model_inf(images)
            s2 = time.time()
            print("Step: "+str(i)+", Time(s): "+str(s2-s1))

            i+=1


    poptorch_model_inf.detachFromDevice()

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


    MODEL_NAME = VERSION+"_dct_"+BENCHMARK_NAME+"_cf"+str(CF)
    
    model = CompressorModel()

    print(args)


    train(args, model)


if __name__ == '__main__':
    main()
