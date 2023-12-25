import torch
import numpy as np

from compressor.dct_funcs import *
from utils.utils import TrainingParams

### Graphcore supports torch.scatter() and torch.gather(), which can
### improve compression ratio with little impact on accuracy

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

def get_idx_array(params: TrainingParams):
    rblks = params.rblks
    cblks = params.cblks
    cf = params.cf
    bs = params.batch_size
    ncols = cblks*cf
    idx = []

    for i in range(int(rblks)):
        blk_row_id = i
        for j in range(int(cblks)):
            blk_col_id = j
            b_start = cf*blk_row_id*ncols + cf*blk_col_id
            for k in range(int(cf)):
                t = cf-k
                for w in range(int(cf)):
                    if w < t:
                        x = b_start + k*ncols+w
                        idx.append(x)
    
    t_idx = []    
    for i in range(int(bs)):
        t_idx.extend(idx)
    idx = np.asarray(t_idx)
    idx = np.reshape(idx, (bs,-1))
    return idx

def apply_diagonal_fold(A, params: TrainingParams):
    ### retrieve index array for use in extracting values
    idx = torch.from_numpy(get_idx_array(params)).to(torch.int64)
    values = torch.gather(torch.reshape(A, (params.batch_size,-1)), 1, idx)
    return values
    
def apply_diagonal_unfold(values, idx, cf_nrows, cf_ncols):
    dst = torch.zeros((idx.shape[0], cf_nrows*cf_ncols))
    dst = torch.scatter(dst, 1, idx, values)
    return dst

def compress(A, params: TrainingParams):
    lhs, rhs = get_lhs_rhs_compress(params)

    rhs = torch.as_tensor(rhs).to(torch.float32)
    lhs = torch.as_tensor(lhs).to(torch.float32)
    A = torch.subtract(A,128)

    o = torch.add(torch.matmul(lhs, torch.matmul(A, rhs)), 0.5).to(torch.int32).to(torch.float32)
    o = apply_diagonal_fold(o,params)
    return o
    
def decompress(A, lhs, rhs,idx, cf_nrows, cf_ncols):
    ### Need idx array (computed at init time)
    A = apply_diagonal_unfold(A, idx, cf_nrows, cf_ncols)
    A = torch.reshape(A, (idx.shape[0], cf_nrows, cf_ncols))
    return torch.add(torch.matmul(lhs, torch.matmul(A, rhs)),128).to(torch.float32)
    
