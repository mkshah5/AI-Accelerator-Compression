from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
#from config import PARAMS

#IMAGE_SHAPE = (PARAMS.batch_size, PARAMS.nchannels,PARAMS.rpix, PARAMS.cpix)
PARAMS_C = None

from typing import Union, List

def normalize(x):
    x = (x - np.min(x))  / (np.max(x) - np.min(x))
    x = np.where(np.isnan(x), np.zeros_like(x), x)
    return x

class CloudDataset(torch.utils.data.Dataset):
    """
    A generic zipped dataset loader for EMDenoiser
    """
    def __init__(self, file_paths: Union[Path, List[Path]]):
        self.paths = file_paths
        self.dataset_len = len(file_paths)

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        loaded_data = self.load_data(self.paths[index])
        img, msk = self._preprocess_images(loaded_data[0],loaded_data[1],loaded_data[2])
        return img, msk
    
    def load_data(self, path):
        #path = path.decode()

        with h5py.File(path, 'r') as handle:
            refs = handle['refs'][:]
            bts = handle['bts'][:]
            msk = handle['bayes'][:]

        bts = (bts - bts.mean()) / bts.std()
        refs = (refs - refs.mean()) / refs.std()
        img = np.concatenate([refs, bts], axis=-1)

        msk[msk > 0] = 1
        msk[msk == 0] = 0
        msk = msk.astype(float)

        return (img, msk, path)
    
    def _preprocess_images(self, img, msk, path):
        # Crop & convert to patches
        img = self._transform_image(img)
        img = normalize(img)
        msk = self._transform_image(msk)

        return torch.from_numpy(img).to(torch.float32), torch.from_numpy(msk).to(torch.float32)
        
    def _transform_image(self, img):
        img = img[0:800, 0:800]
        
        return img.transpose((2,0,1))



# Dataloader specific to this benchmark
def get_data_generator(dataset_dir: str, param):
    global PARAMS_C
    PARAMS_C = param
    data_paths = list(Path(dataset_dir).glob('**/S3A*.hdf'))
    train_paths, test_paths = train_test_split(data_paths, train_size=0.8, random_state=42)

    train_dataset = CloudDataset(train_paths)
    test_dataset = CloudDataset(test_paths)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=False,num_workers=1,drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=param.batch_size, shuffle=True,num_workers=1,drop_last=True)

    return train_data_loader, test_data_loader
