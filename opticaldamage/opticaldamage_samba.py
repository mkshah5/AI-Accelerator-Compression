import argparse
import os
from distutils.util import strtobool
from typing import Tuple
import time
from pathlib import Path
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

import sambaflow.samba.utils as utils
from sambaflow import samba
from sambaflow.samba.env import use_mock_samba_runtime
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.benchmark_acc import AccuracyReport
from sambaflow.samba.utils.common import common_app_driver
from sambaflow.samba.utils.pef_utils import get_pefmeta

from utils.utils import TrainingParams
from compressor.compress_entry import compress, decompress, get_lhs_rhs_decompress
from config import PARAMS
from model import OpticalDamageNet
from data_utils import get_data_generator

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

SAMPLE_SHAPE = (RPIX, CPIX, PARAMS.nchannels)

if IS_BASELINE_NETWORK:
    MODEL_NAME = "base_opticaldamage"
else:
    MODEL_NAME = "matmul_opticaldamage_cf"+str(CF)

DATA_DIR = "/home/shahm/sciml_bench/datasets/optical_damage_ds1"


# Dependent on the number of channels
def full_comp(x):
    out = compress(torch.squeeze(x),PARAMS)
    out = torch.reshape(out, (TRAIN_SIZE, PARAMS.nchannels, out.shape[1], out.shape[2]))

    return out


class OpticalDamageCompress(nn.Module):
    def __init__(self):
        super(OpticalDamageCompress, self).__init__()

        self.criterion = nn.MSELoss()
        self.decompress = decompress
        lhs, rhs = get_lhs_rhs_decompress(PARAMS)        
        self.lhs = samba.from_torch_tensor(torch.as_tensor(lhs).to(torch.bfloat16),name='lhs')
        self.rhs = samba.from_torch_tensor(torch.as_tensor(rhs).to(torch.bfloat16),name='rhs')

        self.internal_model = OpticalDamageNet(SAMPLE_SHAPE)
    # assume bs > 1
    def forward(self, x, gt):
        o = decompress(x,self.lhs,self.rhs)
        o = torch.reshape(o, (TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX))
        
        out = self.internal_model(o)
        loss = self.criterion(out, gt)
        return out, loss
    
class OpticalDamageBase(nn.Module):
    def __init__(self):
        super(OpticalDamageBase, self).__init__()
        self.criterion = nn.MSELoss()
        self.internal_model = OpticalDamageNet(SAMPLE_SHAPE)

    # assume bs > 1
    def forward(self, x, gt):
        out = self.internal_model(x)
        loss = self.criterion(out, gt)
        return out, loss

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.01)
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
    gt = samba.from_torch_tensor(torch.randn(TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX), name='gt', batch_dim=0)
    
      
    return (images, gt)

def get_inputs_base(args: argparse.Namespace) -> Tuple[samba.SambaTensor, samba.SambaTensor]:
    images = torch.randn(TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX)
    images = samba.from_torch_tensor(images, name='image', batch_dim=0)
    gt = samba.from_torch_tensor(torch.randn(TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX), name='gt', batch_dim=0)
        
    return (images, gt)


def train(args: argparse.Namespace, model: nn.Module, optimizer: samba.optim.SGD) -> None:
    train_loader = get_data_generator(Path(DATA_DIR), TRAIN_SIZE, is_inference=False)
    test_loader = get_data_generator(Path(DATA_DIR), TRAIN_SIZE, is_inference=True)

    # Train the model
    total_step = len(train_loader)
    hyperparam_dict = {"lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay}
    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (images) in enumerate(train_loader):
            gt = images.to(torch.float32)
            images = torch.mul(images, 255)
            if not IS_BASELINE_NETWORK:
                images = full_comp(images)

            samba.session.start_runtime_profile()
            
            run_start = time.time()
            sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
            
            sn_labels = samba.from_torch_tensor(gt, name='gt', batch_dim=0)
                    
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
            for images, gt in test_loader:
                gt = images.to(torch.float32)
                images = torch.mul(images, 255)
                if not IS_BASELINE_NETWORK:
                    images = full_comp(images)

                sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
            
                sn_labels = samba.from_torch_tensor(gt, name='gt', batch_dim=0)
                    
                outputs, loss = samba.session.run(input_tensors=[sn_images, sn_labels],
                                                output_tensors=model.output_tensors,
                                                section_types=["FWD"])
            

                loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)

                total_loss += loss.mean()


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
            model = OpticalDamageBase()
        else:
            model = OpticalDamageCompress()
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
