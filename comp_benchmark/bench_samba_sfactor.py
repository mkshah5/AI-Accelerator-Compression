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

import sambaflow.samba.utils as utils
from sambaflow import samba
from sambaflow.samba.env import use_mock_samba_runtime
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.benchmark_acc import AccuracyReport
from sambaflow.samba.utils.common import common_app_driver
from sambaflow.samba.utils.pef_utils import get_pefmeta

from utils.utils import TrainingParams
from compressor.compress_entry import compress, decompress, get_lhs_rhs_decompress
from compressor.compress_entry_highres import compress_sfactor_bfloat16, decompress_sfactor_bfloat16, get_lhs_rhs_decompress, get_new_params

# from config import PARAMS


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

BENCHMARK_NAME = "standalone"
VERSION = "samba"
SFACTOR = 2 

# Dependent on the number of channels
def full_comp(x):
    r = compress_sfactor_bfloat16(torch.squeeze(x[:,0,:,:]), PARAMS, SFACTOR)
    g = compress_sfactor_bfloat16(torch.squeeze(x[:,1,:,:]), PARAMS, SFACTOR)
    b = compress_sfactor_bfloat16(torch.squeeze(x[:,2,:,:]), PARAMS, SFACTOR)
    out = torch.stack((r,g,b),1)

    return out

class CompressorModel(nn.Module):
    def __init__(self):
        super(CompressorModel, self).__init__()


        self.lin = nn.Linear(3*32*32, 1)
        self.decompress = decompress_sfactor_bfloat16
        newparams = get_new_params(SFACTOR, PARAMS)
        lhs, rhs = get_lhs_rhs_decompress(newparams)        
        self.lhs = samba.from_torch_tensor(torch.as_tensor(lhs).to(torch.bfloat16),name='lhs')
        self.rhs = samba.from_torch_tensor(torch.as_tensor(rhs).to(torch.bfloat16),name='rhs')
        self.newrblks = int(newparams.rblks)
        self.newcblks = int(newparams.cblks)
    # assume bs > 1
    def forward(self, x):
        r = self.decompress(torch.squeeze(x[:,0,:,:]), self.lhs,self.rhs,self.newrblks,self.newcblks, RPIX, CPIX,CF)
        g = self.decompress(torch.squeeze(x[:,1,:,:]), self.lhs,self.rhs,self.newrblks,self.newcblks, RPIX, CPIX,CF)
        b = self.decompress(torch.squeeze(x[:,2,:,:]), self.lhs,self.rhs,self.newrblks,self.newcblks, RPIX, CPIX,CF)
        out = torch.stack((r,g,b),1)
        out = torch.reshape(out,(-1,3*32*32))
        return self.lin(out)


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--config_path', type=str, default='./config-ch4.txt')
    parser.add_argument('--compressor', type=str,default='dct')


def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument('-n', '--num-iterations', type=int, default=100, help='Number of iterations to run the pef for')



def get_inputs_compress(args: argparse.Namespace):
    images = torch.randn(TRAIN_SIZE, PARAMS.nchannels, RPIX, CPIX)
    images = full_comp(images)
    images = samba.from_torch_tensor(images, name='image', batch_dim=0)

        
    return (images)


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

    
    MODEL_NAME = VERSION+"_dct_"+BENCHMARK_NAME+"_cf"+str(CF)
    
    utils.set_seed(256)

    inputs = get_inputs_compress(args)
    
    model = CompressorModel()
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

    elif args.command == "run":
        #Run Lenet
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        for i in range(10):
            samba.session.start_runtime_profile()
            s1 = time.time()
            outputs = samba.session.run(input_tensors=[inputs],
                                            output_tensors=model.output_tensors,
                                            section_types=["FWD"])
            o = samba.to_torch(outputs[0])
            s2 = time.time()
            samba.session.end_runtime_profile(MODEL_NAME+'.log')
            print("Step: "+str(i)+", Time(s): "+str(s2-s1))
            print(o)


if __name__ == '__main__':
    main()
