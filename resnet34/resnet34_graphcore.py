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

import matplotlib.pyplot as plt

from torch import Tensor
from typing import Type

import poptorch

from utils.utils import TrainingParams
from compressor.compress_entry import compress, decompress, get_lhs_rhs_decompress

from model import ResNet34

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

BENCHMARK_NAME = "resnet34"
VERSION = "graphcore"

# Dependent on the number of channels
def full_comp(x):
    r = compress(torch.squeeze(x[:,0,:,:]), PARAMS)
    g = compress(torch.squeeze(x[:,1,:,:]), PARAMS)
    b = compress(torch.squeeze(x[:,2,:,:]), PARAMS)
    out = torch.stack((r,g,b),1)

    return out


class ResNetCompress(nn.Module):
    def __init__(self):
        super(ResNetCompress, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.decompress = decompress
        lhs, rhs = get_lhs_rhs_decompress(PARAMS)        
        self.lhs = torch.as_tensor(lhs).to(torch.bfloat16)
        self.rhs = torch.as_tensor(rhs).to(torch.bfloat16)

        self.internal_model = ResNet34()
    # assume bs > 1
    def forward(self, x, labels=None):
        r = decompress(torch.squeeze(x[:,0,:,:]), self.lhs,self.rhs)
        g = decompress(torch.squeeze(x[:,1,:,:]), self.lhs,self.rhs)
        b = decompress(torch.squeeze(x[:,2,:,:]), self.lhs,self.rhs)
        out = torch.stack((r,g,b),1)

        out = self.internal_model(out)
        if self.training:
            loss = self.criterion(out, labels)
            return out, loss
        return out
    
class ResNetBase(nn.Module):
    def __init__(self):
        super(ResNetBase, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.internal_model = ResNet34()

    # assume bs > 1
    def forward(self, x, labels=None):
        out = self.internal_model(x)
        if self.training:
            loss = self.criterion(out, labels)
            return out, loss
        return out

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--config_path', type=str, default='./config-ch4.txt')

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


def prepare_fulldata(args: argparse.Namespace, opts_t, opts_v) -> Tuple[torch.utils.data.DataLoader]:

    dataset_train = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )
    # CIFAR10 validation dataset.
    dataset_valid = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # Create data loaders.

    train_loader = poptorch.DataLoader(
        options = opts_t,
        dataset=dataset_train, 
        batch_size=TRAIN_SIZE,
        shuffle=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        options=opts_v,
        dataset=dataset_valid, 
        batch_size=TEST_SIZE,
        shuffle=False,
        drop_last=True
    )

    return train_loader, valid_loader


def train(args: argparse.Namespace, model: nn.Module, optimizer: poptorch.optim.SGD) -> None:
    
    opts_t = poptorch.Options()
    opts_t.deviceIterations(10)

    opts_v = poptorch.Options()
    opts_v.deviceIterations(10)

    train_loader, test_loader = prepare_fulldata(args,opts_t,opts_v)

    total_step = len(train_loader)
    
    h_l = nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs):
        poptorch_model = poptorch.trainingModel(model,options=opts_t,optimizer=optimizer)
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = torch.mul(images, 255)
            if not IS_BASELINE_NETWORK:
                images = full_comp(images)
            
            run_start = time.time()
            outputs,loss = poptorch_model(images,labels)
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
                images = torch.mul(images, 255)
            
                if not IS_BASELINE_NETWORK:
                    images = full_comp(images)

                outputs = poptorch_model_inf(images)

                loss = h_l(outputs,labels)
                total_loss += loss.mean()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            test_acc = 100.0 * correct / total
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
            model = ResNetBase()
        else:
            model = ResNetCompress()

    print(args.mapping)

    optimizer = poptorch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(args, model, optimizer)


if __name__ == '__main__':
    main()
