from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn, optim
import torch

IMAGE_SHAPE = (200, 200, 1)

def normalize(x):
    x = (x - np.min(x))  / (np.max(x) - np.min(x))
    x = np.where(np.isnan(x), np.zeros_like(x), x)
    return x

def load_images(file_path):
    # List all TIFF files in the directory
    file_names = list(Path(file_path).glob('*.TIFF'))

    images = np.zeros((len(file_names), *IMAGE_SHAPE))
    for index, file_name in enumerate(tqdm(file_names)):
        img = Image.open(file_name)

        # A numpy array containing the tiff data
        image = np.array(img)
        image = image.astype(np.float32)
        image = normalize(image)

        # crop image around optic
        image = image[150:350, 250:450]
        image = np.expand_dims(image, axis=-1)
        images[index] = image

    return images

class OpticalDamageDataset(torch.utils.data.Dataset):
    """
    A generic zipped dataset loader for EMDenoiser
    """
    def __init__(self, file_path: Path):
        self.path = file_path
        self.images = load_images(self.path)
        self.dataset_len = self.images.shape[0]

        self.images = torch.from_numpy(self.images.transpose((0, 3,1,2)) )

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        return self.images[index]


def get_data_generator(base_dataset_dir: Path, batch_size: int, is_inference=False):
    """
    Returns a data loader for training or inference datasets 
    based on the is_inference flag.
    """
    shuffle_flag = True
    if is_inference:
       shuffle_flag = False

    params = {
        'batch_size': batch_size,
        'shuffle': shuffle_flag,
        'num_workers': 1,
        'drop_last': True
    }

    if is_inference: 
        undamaged_path = base_dataset_dir / 'inference' / 'undamaged'
        inference_dataset = OpticalDamageDataset(undamaged_path)
        inference_generator = torch.utils.data.DataLoader(inference_dataset, **params)
        return  inference_generator
    else: 
        undamaged_path = base_dataset_dir / 'training' / 'undamaged'
        training_dataset = OpticalDamageDataset(undamaged_path)
        training_generator = torch.utils.data.DataLoader(training_dataset, **params)
        return  training_generator


