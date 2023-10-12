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

import sambaflow.samba.utils as utils
from sambaflow import samba
from sambaflow.samba.env import use_mock_samba_runtime
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.benchmark_acc import AccuracyReport
from sambaflow.samba.utils.common import common_app_driver
from sambaflow.samba.utils.pef_utils import get_pefmeta

from utils.utils import TrainingParams
from compressor.compress_entry import compress, decompress, get_lhs_rhs_decompress
from resnet34_config import PARAMS
from model import ResNet34

MOCK_SAMBA_RUNTIME = use_mock_samba_runtime()
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
    MODEL_NAME = "base_resnet34"
else:
    MODEL_NAME = "matmul_resnet34_cf"+str(CF)

MODEL = ResNet34(TRAIN_SIZE)

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
        self.lhs = samba.from_torch_tensor(torch.as_tensor(lhs).to(torch.bfloat16),name='lhs')
        self.rhs = samba.from_torch_tensor(torch.as_tensor(rhs).to(torch.bfloat16),name='rhs')

    # assume bs > 1
    def forward(self, x, labels):
        r = decompress(torch.squeeze(x[:,0,:,:]), self.lhs,self.rhs)
        g = decompress(torch.squeeze(x[:,1,:,:]), self.lhs,self.rhs)
        b = decompress(torch.squeeze(x[:,2,:,:]), self.lhs,self.rhs)
        out = torch.stack((r,g,b),1)

        out = MODEL(out)
        loss = self.criterion(out, labels)
        return out, loss
    
class ResNetBase(nn.Module):
    def __init__(self):
        super(ResNetBase, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    # assume bs > 1
    def forward(self, x, labels):
        out = MODEL(x)
        loss = self.criterion(out, labels)
        return out, loss

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


def get_inputs_compress(args: argparse.Namespace) -> Tuple[samba.SambaTensor, samba.SambaTensor]:
    images = torch.randn(TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX)
    images = full_comp(images)
    images = samba.from_torch_tensor(images, name='image', batch_dim=0)
    labels = samba.randint(args.num_classes, (TRAIN_SIZE, ), name='label', batch_dim=0)
        
    return (images, labels)

def get_inputs_base(args: argparse.Namespace) -> Tuple[samba.SambaTensor, samba.SambaTensor]:
    images = torch.randn(TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX)
    images = samba.from_torch_tensor(images, name='image', batch_dim=0)
    labels = samba.randint(args.num_classes, (TRAIN_SIZE, ), name='label', batch_dim=0)
        
    return (images, labels)



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


def train(args: argparse.Namespace, model: nn.Module, optimizer: samba.optim.SGD) -> None:
    train_loader, test_loader = prepare_fulldata(args)

    # Train the model
    total_step = len(train_loader)
    hyperparam_dict = {"lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay}
    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = torch.mul(images, 255)
            
            if not IS_BASELINE_NETWORK:
                images = full_comp(images)

            samba.session.start_runtime_profile()
            
            run_start = time.time()
            sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
            
            sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)
                    
            outputs, loss = samba.session.run(input_tensors=[sn_images, sn_labels],
                                            output_tensors=model.output_tensors,
                                            hyperparam_dict=hyperparam_dict,
                                            data_parallel=args.data_parallel,
                                            reduce_on_rdu=args.reduce_on_rdu)
            
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            
            run_end = time.time()
            samba.session.end_runtime_profile(MODEL_NAME+'.log')
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

                sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
            
                sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)
                    
                outputs, loss = samba.session.run(input_tensors=[sn_images, sn_labels],
                                                output_tensors=model.output_tensors,
                                                section_types=["FWD"])
            

                loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)

                total_loss += loss.mean()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            test_acc = 100.0 * correct / total
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))
    samba.session.to_cpu(model)
    
    torch.save(model, MODEL_NAME+".pt")

def main():
    args = parse_app_args(dev_mode=True,
                          common_parser_fn=add_common_args,
                          test_parser_fn=add_run_args,
                          run_parser_fn=add_run_args)
    utils.set_seed(256)
    if IS_BASELINE_NETWORK:
        inputs = get_inputs_base(args)
    else:
        inputs = get_inputs_compress(args)

    if args.command == "test":
        model = torch.load(MODEL_NAME+".pt")
    else:
        if IS_BASELINE_NETWORK:
            model = ResNetBase()
        else:
            model = ResNetCompress()
    samba.from_torch_model_(model)
    print(inputs)
    print(args.mapping)

    optimizer = samba.optim.SGD(model.parameters(), lr=args.lr) if not args.inference else None
    if args.command == "compile":
        samba.session.compile(model,
                              inputs,
                              optimizer,
                              name=MODEL_NAME,
                              app_dir=utils.get_file_dir(__file__),
                              squeeze_bs_dim=False,
                              config_dict=vars(args),
                              pef_metadata=get_pefmeta(args, model))

    elif args.command == "test":
        #Test Lenet
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        outputs = model.output_tensors
        # run_test(args, model, outputs)
    elif args.command == "run":
        #Run Lenet
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        train(args, model, optimizer)


if __name__ == '__main__':
    main()
