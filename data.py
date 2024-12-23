# uses code from https://github.com/JTT94/diffusion_schrodinger_bridge file bridge/data/two_dim.py
import numpy as np
import torch
from sklearn.datasets import make_circles, make_moons
from abc import ABC, abstractmethod
from PIL import Image
from os.path import join
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# Abstract base class
class DataSampler(ABC):
    @abstractmethod
    def sample(self, n):
        """
        Abstract method to generate n samples.
        
        Args:
            n (int): Number of samples to generate.
        """
        pass

# GaussianSampler class
class GaussianSampler(DataSampler):
    def __init__(self, mean: torch.Tensor, std: float):
        """
        Initialize a Gaussian sampler.
        
        Args:
            mean (torch.Tensor): Mean vector of the Gaussian distribution.
            std: noise
        """
        self.mean = mean
        self.std = std
    
    def sample(self, n: int) -> torch.Tensor:
        return self.mean + self.std * torch.rand((n, *self.mean.shape))
    
    def __str__(self):
        return f"GaussianSampler(mean={self.mean}, std={self.std})"

# TagSampler class
class TagSampler(DataSampler):
    def __init__(self, tag: str, n_samples: int):
        """
        Initialize a TagSampler to choose between various toy 2d distributions.
        
        Args:
            tag (str): Type of distribution.
            N (int): number of samples stored
        """
        if tag == 'circles':
            pos, label = make_circles(n_samples=n_samples, noise=0, factor=0.5, random_state=None)
        elif tag == 'moons':
            pos, label = make_moons(n_samples=n_samples, noise=0, random_state=None)
        else:
            raise Exception(f'Unknown tag: {tag}')
        
        self.tag = tag
        self.data = torch.tensor(pos, dtype=torch.float32)
        self.n_samples = self.data.shape[0]
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)
    
    def sample(self, n: int) -> torch.Tensor:
        random_idx = torch.randint(0, self.n_samples, (n,))
        return self.data[random_idx]
    
    def __str__(self):
        return f"TagSampler(tag={self.tag}), n_samples={self.n_samples} mean={self.mean}, cov={self.std}"
    

# ImageSampler class
class ImageSampler(DataSampler):
    def __init__(self, filename: str, n_samples: int=None):
        self.filename = filename
        data = torch.tensor(make_img_2d_dataset(filename), dtype=torch.float32)
        if n_samples: # we subset the points to have N points (instead of the natural number of points from the image)
            random_idx = torch.randint(0, data.shape[0], (n_samples,))
            data = data[random_idx]
        self.data = data
        self.n_samples = self.data.shape[0]
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)
    
    def sample(self, n: int) -> torch.Tensor:
        random_idx = torch.randint(0, self.n_samples, (n,))
        return self.data[random_idx]
    
    def __str__(self):
        return f"ImageSampler(filename={self.filename}), n_samples={self.n_samples} mean={self.mean}, std={self.std}"


def make_img_2d_dataset(filename: str):
    """
    creates a 2d dataset which look like the PyTorch logo
    """
    # Load the image
    img = Image.open(join('assets', filename)).convert('L')  # Convert to grayscale

    # Threshold the image to isolate the logo
    threshold = 200
    binary_img = np.array(img) < threshold

    # Extract 2D coordinates of the logo
    y_coords, x_coords = np.where(binary_img)  # Nonzero pixel positions
    coordinates = np.stack((x_coords, -y_coords), axis=1)

    # Normalize coordinates to fit in [-1, 1] range
    coordinates = coordinates.astype(float)
    coordinates -= coordinates.min(axis=0)  # Shift to zero
    coordinates /= coordinates.max(axis=0)  # Scale to [0, 1]
    coordinates = 2 * coordinates - 1  # Scale to [-1, 1]

    return coordinates


def config_to_p(pconfig: dict) -> DataSampler:
    """create a DataSampler from a config file"""
    if pconfig["type"] == "tag":
        p = TagSampler(tag=pconfig['tag'], n_samples=pconfig['n_samples'])
    elif pconfig["type"] == "image":
        p = ImageSampler(filename=pconfig['image'], n_samples=pconfig['n_samples'])
    elif pconfig["type"] == "MNIST":
        p = MNISTSampler()
    else:
        raise ValueError(f'Invalid p type. pconfig={pconfig}') 
    return p


class MNISTSampler(DataSampler):
    def __init__(self):
        """
        Initialize a sampler for MNIST.
        """      
        # Normalizer
        transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
            ])
        # Load MNIST dataset
        mnist = torchvision.datasets.MNIST(
            root="data/",
            train=True,
            transform=transform,
            download=True
        )
        self.data = torch.cat([mnist[idx][0] for idx in range(len(mnist))], dim=0)[:,None,:,:] # add axis to have shape (60_000,1,28,28)
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std()
        self.n_samples = self.data.shape[0]

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n random images from MNIST.
        """
        # Random sampling with replacement
        random_idx = torch.randint(0, self.n_samples, (n,))
        samples = self.data[random_idx]
        return samples
    
    def __str__(self):
        return f"MNISTSampler(n_data={self.n_samples})"