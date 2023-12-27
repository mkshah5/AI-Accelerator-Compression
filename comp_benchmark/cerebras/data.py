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

import torch
from torchvision import datasets, transforms
import sys
sys.path.append('..')

import cerebras_pytorch as cstorch
import cerebras_pytorch.distributed as dist
from modelzoo.common.pytorch.input_utils import get_streaming_batch_size
from modelzoo.common.pytorch.utils import SampleGenerator
from utils.utils import TrainingParams

from compressor.compress_entry_f32 import compress, decompress, get_lhs_rhs_decompress

def full_comp(x, params):
    r = compress(torch.squeeze(x[:,0,:,:]), params)
    g = compress(torch.squeeze(x[:,1,:,:]), params)
    b = compress(torch.squeeze(x[:,2,:,:]), params)
    out = torch.stack((r,g,b),1)

    return out

def get_train_dataloader(params):
    """
    :param <dict> params: dict containing input parameters for creating dataset.
    Expects the following fields:

    - "data_dir" (string): path to the data files to use.
    - "batch_size" (int): batch size
    - "to_float16" (bool): whether to convert to float16 or not
    - "drop_last_batch" (bool): whether to drop the last batch or not
    """
    cparams = params["cparams"]
    input_params = params["train_input"]
    use_cs = cstorch.use_cs()

    batch_size = get_streaming_batch_size(cparams.batch_size)
    to_float16 = input_params.get("to_float16", True)
    use_bfloat16 = params["model"].get("use_bfloat16", False)
    shuffle = input_params["shuffle"]

    dtype = torch.float32
    if to_float16:
        if use_cs or torch.cuda.is_available():
            dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        else:
            print(
                f"Input dtype float16 is not supported with "
                f"vanilla PyTorch CPU workflow. Using float32 instead."
            )

    x = torch.randn(cparams.batch_size, cparams.nchannels, cparams.rpix, cparams.cpix, dtype=dtype)
    x = full_comp(x, cparams)

    num_streamers = dist.num_streamers() if dist.is_streamer() else 1
    train_loader = SampleGenerator(
        data=(
            x,
            torch.zeros(
                cparams.batch_size, dtype=torch.int32 if use_cs else torch.int64
            ),
        ),
        sample_count=60000 // batch_size // num_streamers,
    )
   
    return train_loader
