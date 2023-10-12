import numpy as np
from utils.utils import TrainingParams

def generate_T(params: TrainingParams):
    BD = params.BD
    T = np.zeros((BD,BD))

    for i in range(BD):
        for j in range(BD):
            if i == 0:
                T[i,j] = 1/np.sqrt(BD)
            else:
                cos_term = ((2*j+1)*i*np.pi)/(2*BD)
                T[i,j] = np.sqrt(2/BD)*np.cos(cos_term)
    return T

def generate_grown_T(T, params: TrainingParams):
    RPIX = params.rpix
    CPIX = params.cpix
    RBLKS = params.rblks
    BD = params.BD

    grown_dct = np.zeros((RPIX,CPIX))
    
    for i in range(int(RBLKS)):
          grown_dct[i*BD:(i+1)*BD, i*BD:(i+1)*BD] = T

    return grown_dct

def generate_grown_T_t(T, params: TrainingParams):
    RPIX = params.rpix
    CPIX = params.cpix
    RBLKS = params.rblks
    BD = params.BD

    grown_dct = np.zeros((RPIX,CPIX))
    T = np.transpose(T)
    for i in range(int(RBLKS)):
          grown_dct[i*BD:(i+1)*BD, i*BD:(i+1)*BD] = T

    return grown_dct

def generate_mask(params: TrainingParams):
    RPIX = params.rpix
    CPIX = params.cpix
    RBLKS = params.rblks
    BD = params.BD
    CF = params.cf

    mask = np.zeros((int(CF*RBLKS),CPIX))
    identity = np.identity(CF)
    for i in range(int(RBLKS)):
        for j in range(CF):
            mask[i*CF:(i+1)*CF,i*BD:i*BD+CF] = identity
    
    return mask

