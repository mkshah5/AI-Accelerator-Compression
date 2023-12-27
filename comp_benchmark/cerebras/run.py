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
VERSION = "groq"

# Dependent on the number of channels
def full_comp(x):
    r = compress(torch.squeeze(x[:,0,:,:]), PARAMS)
    g = compress(torch.squeeze(x[:,1,:,:]), PARAMS)
    b = compress(torch.squeeze(x[:,2,:,:]), PARAMS)
    out = torch.stack((r,g,b),1)

    return out


class CompressorModel(nn.Module):
    def __init__(self,params):
        super(CompressorModel, self).__init__()
        cparams = params["cparams"]
        self.decompress = decompress
        lhs, rhs = get_lhs_rhs_decompress(cparams)        
        self.lhs = torch.as_tensor(lhs).to(torch.float32)
        self.rhs = torch.as_tensor(rhs).to(torch.float32)

    # assume bs > 1
    def forward(self, x):
        r = decompress(torch.squeeze(x[:,0,:,:]), self.lhs,self.rhs)
        g = decompress(torch.squeeze(x[:,1,:,:]), self.lhs,self.rhs)
        b = decompress(torch.squeeze(x[:,2,:,:]), self.lhs,self.rhs)
        out = torch.stack((r,g,b),1)

        return out
    
def main():
    params = get_params_from_args()
    from cerebras_utils import set_defaults

    set_defaults(params)
    params["cparams"] = TrainingParams("./config-ch4.txt")


    from modelzoo.common.pytorch.run_utils import main
    from modelzoo.fc_mnist.pytorch.data import (
        get_train_dataloader,
    )

    main(params, CompressorModel, get_train_dataloader, get_train_dataloader)


if __name__ == '__main__':
    main()
