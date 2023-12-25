import torch
import numpy as np

from compressor.dct_funcs import *
#from compress_entry import *
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

def compress_sfactor1(A, params: TrainingParams):
    lhs, rhs = get_lhs_rhs_compress(params)

    rhs = torch.as_tensor(rhs).to(torch.float32)
    lhs = torch.as_tensor(lhs).to(torch.float32)
    A = torch.subtract(A,128)

    o = torch.add(torch.matmul(lhs, torch.matmul(A, rhs)), 0.5).to(torch.int32).to(torch.float32)

    return o

def get_new_params(sfactor, params: TrainingParams):
    newParams = TrainingParams('')
    newParams.batch_size = params.batch_size
    newParams.cf = params.cf
    newParams.rpix = int(params.rpix/sfactor)
    newParams.cpix = int(params.cpix/sfactor)
    newParams.nchannels = params.nchannels
    newParams.is_base = params.is_base

    newParams.rblks = newParams.rpix/params.BD
    newParams.cblks = newParams.cpix/params.BD
    newParams.BD = params.BD
    return newParams

def compress_sfactor(A, params: TrainingParams,sfactor=1):
    
    newparams = get_new_params(sfactor, params)

    lhs, rhs = get_lhs_rhs_compress(newparams)

    rhs = torch.as_tensor(rhs).to(torch.float32)
    lhs = torch.as_tensor(lhs).to(torch.float32)
    A = torch.reshape(A, (-1, newparams.rpix, newparams.cpix))
    A = torch.subtract(A,128)

    o = torch.add(torch.matmul(lhs, torch.matmul(A, rhs)), 0.5).to(torch.int32).to(torch.float32)
    o = torch.reshape(o, (newparams.batch_size, int(params.rblks*params.cf),int(params.cblks*params.cf)))
    return o

def compress_sfactor_2(A, params: TrainingParams):
    newparams = get_new_params(2, params)

    lhs, rhs = get_lhs_rhs_compress(newparams)

    rhs = torch.as_tensor(rhs).to(torch.float32)
    lhs = torch.as_tensor(lhs).to(torch.float32)

    A = torch.subtract(A,128)
    A1 = A[:,0:newparams.rpix, 0:newparams.cpix]
    A2 = A[:,0:newparams.rpix, newparams.cpix:-1]
    A3 = A[:,newparams.rpix:-1, 0:newparams.cpix]
    A4 = A[:,newparams.rpix:-1, newparams.cpix:-1]

    o1 = torch.add(torch.matmul(lhs, torch.matmul(A1, rhs)), 0.5).to(torch.int32).to(torch.float32)
    o2 = torch.add(torch.matmul(lhs, torch.matmul(A2, rhs)), 0.5).to(torch.int32).to(torch.float32)
    o3 = torch.add(torch.matmul(lhs, torch.matmul(A3, rhs)), 0.5).to(torch.int32).to(torch.float32)
    o4 = torch.add(torch.matmul(lhs, torch.matmul(A4, rhs)), 0.5).to(torch.int32).to(torch.float32)

    return torch.stack((o1,o2,o3,o4),1)

def compress_on_device(A, lhs, rhs):
    # Same compression algorithm, except require lhs and rhs arguments to be passed in such that
    # these arrays can be allocated properly in device memory
    A = torch.subtract(A,128)

    o = torch.add(torch.matmul(lhs, torch.matmul(A, rhs)), 0.5).to(torch.int32).to(torch.float32)

    return o
    
def decompress(A, lhs, rhs):
    return torch.add(torch.matmul(lhs, torch.matmul(A, rhs)),128).to(torch.float32)

def decompress_sfactor(A, lhs, rhs, newrblks, newcblks, rpix, cpix,cf):
    A = torch.reshape(A, (-1,int(newrblks*cf),int(newcblks*cf)))
    o = torch.add(torch.matmul(lhs, torch.matmul(A, rhs)),128).to(torch.float32)

    return torch.reshape(o, (-1, rpix, cpix))

#    return o

def decompress_sfactor_2(A, lhs, rhs, rpix, cpix):
    A1 = torch.squeeze(A[:,0,:,:])
    A2 = torch.squeeze(A[:,1,:,:])
    A3 = torch.squeeze(A[:,2,:,:])
    A4 = torch.squeeze(A[:,3,:,:])

    o1 = torch.add(torch.matmul(lhs, torch.matmul(A1, rhs)),128).to(torch.float32)
    o2 = torch.add(torch.matmul(lhs, torch.matmul(A2, rhs)),128).to(torch.float32)
    o3 = torch.add(torch.matmul(lhs, torch.matmul(A3, rhs)),128).to(torch.float32)
    o4 = torch.add(torch.matmul(lhs, torch.matmul(A4, rhs)),128).to(torch.float32)
    o = torch.cat((o1,o2,o3,o4),-1)
    return torch.reshape(o, (-1, rpix, cpix))
