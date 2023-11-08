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
# from config import PARAMS
from model import CloudMaskNet
from data_utils import get_data_generator

MOCK_SAMBA_RUNTIME = use_mock_samba_runtime()
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
VERSION = "samba"

# Dependent on the number of channels
def full_comp(x):
    r = compress(torch.squeeze(x[:,0,:,:]),PARAMS)
    g = compress(torch.squeeze(x[:,1,:,:]),PARAMS)
    b = compress(torch.squeeze(x[:,2,:,:]),PARAMS)
    r1 = compress(torch.squeeze(x[:,3,:,:]),PARAMS)
    g1 = compress(torch.squeeze(x[:,4,:,:]),PARAMS)
    b1 = compress(torch.squeeze(x[:,5,:,:]),PARAMS)
    r2 = compress(torch.squeeze(x[:,6,:,:]),PARAMS)
    g2 = compress(torch.squeeze(x[:,7,:,:]),PARAMS)
    b2 = compress(torch.squeeze(x[:,8,:,:]),PARAMS)

    out = torch.stack((r,g,b,r1,g1,b1,r2,g2,b2),1)
    return out


class CloudMaskCompress(nn.Module):
    def __init__(self):
        super(CloudMaskCompress, self).__init__()

        self.criterion = nn.BCELoss()
        self.decompress = decompress
        lhs, rhs = get_lhs_rhs_decompress(PARAMS)        
        self.lhs = samba.from_torch_tensor(torch.as_tensor(lhs).to(torch.bfloat16),name='lhs')
        self.rhs = samba.from_torch_tensor(torch.as_tensor(rhs).to(torch.bfloat16),name='rhs')

        self.internal_model = CloudMaskNet((RPIX,CPIX,PARAMS.nchannels))
    # assume bs > 1
    def forward(self, x, gt):
        r = decompress(torch.squeeze(x[:,0,:,:]), self.lhs,self.rhs)
        g = decompress(torch.squeeze(x[:,1,:,:]), self.lhs,self.rhs)
        b = decompress(torch.squeeze(x[:,2,:,:]), self.lhs,self.rhs)
        r1 = decompress(torch.squeeze(x[:,3,:,:]), self.lhs,self.rhs)
        g1 = decompress(torch.squeeze(x[:,4,:,:]), self.lhs,self.rhs)
        b1 = decompress(torch.squeeze(x[:,5,:,:]), self.lhs,self.rhs)
        r2 = decompress(torch.squeeze(x[:,6,:,:]), self.lhs,self.rhs)
        g2 = decompress(torch.squeeze(x[:,7,:,:]), self.lhs,self.rhs)
        b2 = decompress(torch.squeeze(x[:,8,:,:]), self.lhs,self.rhs)
        o = torch.stack((r,g,b,r1,g1,b1,r2,g2,b2),1)

        o = torch.reshape(o, (TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX))
        
        out = self.internal_model(o)
        loss = self.criterion(out, gt)
        return out, loss
    
class CloudMaskBase(nn.Module):
    def __init__(self):
        super(CloudMaskBase, self).__init__()
        self.criterion = nn.BCELoss()
        self.internal_model = CloudMaskNet((RPIX,CPIX,PARAMS.nchannels))

    # assume bs > 1
    def forward(self, x, gt):
        out = self.internal_model(x)
        loss = self.criterion(out, gt)
        return out, loss

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.00001)
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


def get_inputs_compress(args: argparse.Namespace) -> Tuple[samba.SambaTensor, samba.SambaTensor]:
    images = torch.randn(TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX)
    images = full_comp(images)
    images = samba.from_torch_tensor(images, name='image', batch_dim=0)
    gt = samba.from_torch_tensor(torch.randn(TRAIN_SIZE, 1, RPIX, CPIX), name='gt', batch_dim=0)
    
      
    return (images, gt)

def get_inputs_base(args: argparse.Namespace) -> Tuple[samba.SambaTensor, samba.SambaTensor]:
    images = torch.randn(TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX)
    images = samba.from_torch_tensor(images, name='image', batch_dim=0)
    gt = samba.from_torch_tensor(torch.randn(TRAIN_SIZE, 1, RPIX, CPIX), name='gt', batch_dim=0)
        
    return (images, gt)


def train(args: argparse.Namespace, model: nn.Module, optimizer: samba.optim.SGD) -> None:
    train_loader, test_loader = get_data_generator(DATA_DIR,PARAMS)

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
            
            sn_labels = samba.from_torch_tensor(labels, name='gt', batch_dim=0)
                    
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
            
                sn_labels = samba.from_torch_tensor(labels, name='gt', batch_dim=0)
                    
                outputs, loss = samba.session.run(input_tensors=[sn_images, sn_labels],
                                                output_tensors=model.output_tensors,
                                                section_types=["FWD"])
            

                loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)

                total_loss += loss.mean()


            #test_acc = 100.0 * correct / total
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))
    samba.session.to_cpu(model)
    
    torch.save(model, MODEL_NAME+".pt")

def main():
    global PARAMS, TRAIN_SIZE, TEST_SIZE, CF, RPIX, CPIX, BD, RBLKS, CBLKS, IS_BASELINE_NETWORK, MODEL_NAME

    args = parse_app_args(dev_mode=True,
                          common_parser_fn=add_common_args,
                          test_parser_fn=add_run_args,
                          run_parser_fn=add_run_args)
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

    utils.set_seed(256)
    if IS_BASELINE_NETWORK:
        inputs = get_inputs_base(args)
    else:
        inputs = get_inputs_compress(args)

    if args.command == "test":
        model = torch.load(MODEL_NAME+".pt")
    else:
        if IS_BASELINE_NETWORK:
            model = CloudMaskBase()
        else:
            model = CloudMaskCompress()
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
