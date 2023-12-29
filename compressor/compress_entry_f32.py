import torch
import numpy as np

from compressor.dct_funcs import *
from utils.utils import TrainingParams

def get_lhs_rhs_compress(params: TrainingParams):
    T = generate_T(params)
    grown_dct = generate_grown_T(T, params)
    grown_dct_T = np.transpose(grown_dct)
    
    mask = generate_mask(params)
    mask_T = np.transpose(mask)

    rhs = np.matmul(grown_dct_T, mask_T)
    lhs = np.matmul(mask, grown_dct)

    return lhs, rhs

def get_lhs_rhs_decompress(params: TrainingParams):
    T = generate_T(params)
    grown_dct = generate_grown_T(T, params)
    grown_dct_T = np.transpose(grown_dct)
    
    mask = generate_mask(params)
    mask_T = np.transpose(mask)
    
    lhs = np.matmul(grown_dct_T, mask_T)
    rhs = np.matmul(mask, grown_dct)

    return lhs, rhs

def compress(A, params: TrainingParams):
    lhs, rhs = get_lhs_rhs_compress(params)

    rhs = torch.as_tensor(rhs).to(torch.float32)
    lhs = torch.as_tensor(lhs).to(torch.float32)
    A = torch.subtract(A,128)

    o = torch.add(torch.matmul(lhs, torch.matmul(A, rhs)), 0.5).to(torch.int32).to(torch.float32)

    return o

def compress_on_device(A, lhs, rhs):
    # Same compression algorithm, except require lhs and rhs arguments to be passed in such that
    # these arrays can be allocated properly in device memory
    A = torch.subtract(A,128)

    o = torch.add(torch.matmul(lhs, torch.matmul(A, rhs)), 0.5).to(torch.int32).to(torch.float32)

    return o
    
def decompress(A, lhs, rhs):
    return torch.add(torch.matmul(lhs, torch.matmul(A, rhs)),128).to(torch.float32)
