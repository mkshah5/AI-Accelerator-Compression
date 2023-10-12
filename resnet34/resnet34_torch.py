#!/usr/bin/env python
from pprint import pprint
from pathlib import Path
from libpressio import PressioCompressor

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
from resnet34_config import PARAMS
from model import ResNet34

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
    MODEL_NAME = "torch_base_resnet34"
else:
    MODEL_NAME = "torch_matmul_resnet34_cf"+str(CF)

MODEL = ResNet34(TRAIN_SIZE)

# uncompressed_data = np.fromfile(dataset_path, dtype=np.float32)
# uncompressed_data = uncompressed_data.reshape(500, 500, 100)
# decompressed_data = uncompressed_data.copy()

# load and configure the compressor
compressor = PressioCompressor.from_config({
    "compressor_id": "sz",
    "compressor_config": {
        "sz:error_bound_mode_str": "abs",
        "sz:abs_err_bound": 1e-6,
        "sz:metric": "size"
        }
    })


# preform compression and decompression
# compressed = compressor.encode(uncompressed_data)
# decompressed = compressor.decode(compressed, decompressed_data)

# # print out some metrics collected during compression
# pprint(compressor.get_metrics())

# Dependent on the number of channels
def full_comp(x):
    n_x = x.numpy()
    decomp = n_x.copy()
    out = compressor.encode(n_x)
    decomp = compressor.decode(out)
    return decomp


class ResNetCompress(nn.Module):
    def __init__(self):
        super(ResNetCompress, self).__init__()

        self.decompress = decompress
        lhs, rhs = get_lhs_rhs_decompress(PARAMS)        
        self.lhs = torch.as_tensor(lhs).to(torch.bfloat16)
        self.rhs = torch.as_tensor(rhs).to(torch.bfloat16)

    # assume bs > 1
    def forward(self, x, labels):

        out = MODEL(x)
        return out
    
class ResNetBase(nn.Module):
    def __init__(self):
        super(ResNetBase, self).__init__()

    # assume bs > 1
    def forward(self, x, labels):
        out = MODEL(x)
        return out

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)


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


def prepare_fulldata(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader]:

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
    train_loader = DataLoader(
        dataset_train, 
        batch_size=TRAIN_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid, 
        batch_size=TEST_SIZE,
        shuffle=False
    )

    return train_loader, valid_loader


def train(args: argparse.Namespace, model: nn.Module, optimizer) -> None:
    train_loader, test_loader = prepare_fulldata(args)
    loss_function = nn.CrossEntropyLoss()
    # Train the model
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = torch.mul(images, 255)
            
            if not IS_BASELINE_NETWORK:
                images = full_comp(images)

            
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
                images = torch.mul(images, 255)
            
                if not IS_BASELINE_NETWORK:
                    images = full_comp(images)

                outputs = model(images)
                loss = loss_function(outputs,labels)
                
                total_loss += loss.mean()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            test_acc = 100.0 * correct / total
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))
    
    torch.save(model, MODEL_NAME+".pt")

def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_run_args(parser)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.command == "test":
        model = torch.load(MODEL_NAME+".pt").to(device)
    else:
        if IS_BASELINE_NETWORK:
            model = ResNetBase().to(device)
        else:
            model = ResNetCompress().to(device)

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    train(args, model, optimizer)


if __name__ == '__main__':
    main()
