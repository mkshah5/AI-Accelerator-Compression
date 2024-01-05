# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# isort: off
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append('..')
# isort: on
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelzoo.common.run_utils.cli_pytorch import get_params_from_args
from utils.utils import TrainingParams
from compressor.compress_entry_f32 import compress_on_device, decompress, get_lhs_rhs_decompress

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
VERSION = "cerebras"



class CompressorModel(nn.Module):
    def __init__(self,params):
        super(CompressorModel, self).__init__()
        cparams = params["cparams"]
        self.decompress = compress_on_device

    # assume bs > 1
    def forward(self, inputs):
        x,labels,lhs,rhs = inputs
        lhs = torch.squeeze(lhs[0,:,:])
        rhs = torch.squeeze(rhs[0,:,:])
        r = decompress(torch.squeeze(x[:,0,:,:]), lhs,rhs)
        g = decompress(torch.squeeze(x[:,1,:,:]), lhs,rhs)
        b = decompress(torch.squeeze(x[:,2,:,:]), lhs,rhs)
        out = torch.stack((r,g,b),1)

        return torch.sum(out)
    
def main():
    params = get_params_from_args()
    from cerebras_utils import set_defaults

    set_defaults(params)
    cfs = [4,7]

    from modelzoo.common.pytorch.run_utils import main
    from modelzoo.fc_mnist.pytorch.data import (
        get_train_dataloader,
    )
    ver = os.getenv('DCT_COMP_CONFIG')
    print(ver)
    params["cparams"] = TrainingParams(ver)



    main(params, CompressorModel, get_train_dataloader, get_train_dataloader)


if __name__ == '__main__':
    main()
