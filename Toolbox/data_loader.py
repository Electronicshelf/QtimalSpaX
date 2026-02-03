
from itertools import chain
import os
from pathlib import Path
from munch import Munch
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
torch.manual_seed(23)


def listdir(dname):
    """ List file directory"""
    fnames_only = []
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG', 'bmp']]))
    fnames = sorted(fnames)
    for path in fnames:
        file_name = Path(path).name
        fnames_only.append(file_name)
    print(fnames_only)
    print(len(fnames))
    return fnames

class DefaultDataset(data.Dataset):
    """
     A custom PyTorch Dataset class to load images
      from a specified directory.

        This dataset class reads image filenames from
        a given directory, optionally applies transformations,
        and returns a sample (image and its filename) suitable for PyTorch's DataLoader.

        Attributes:
            samples (list): A list of filenames for all image
             files in the specified root directory.
            transform (callable, optional): A function or series of
             functions to apply transformations to the images.
            targets (None): Placeholder for target labels, currently unused.
    """
    def __init__(self, root, transform=None):
        self.samples = self.make_dataset(root)
        self.transform = transform
        self.targets = None

    def make_dataset(self, root):
        domains = listdir(root)
        # String to remove #
        str_ = ".DS_Store"
        # Remove the string if it exists #
        domains = [item for item in domains if item != str_]
        fnames = []
        for idx, file in enumerate(sorted(domains)):
            fnames.append(file)

        return fnames

    def __getitem__(self, index):
        f_name = self.samples[index]
        img = Image.open(f_name).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, str(f_name)

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    """
        A custom PyTorch Dataset class to load images
        and their corresponding class labels from a structured directory.

        This dataset reads image files from subdirectories
        in the root directory where each subdirectory corresponds
        to a class label. The dataset stores both the image
        filenames and their associated labels and can optionally
        apply transformations to the images.

     Attributes:
        samples (list): A list of filenames for all image files in the directory.
        targets (list): A list of integer class labels
        corresponding to the image files.
        transform (callable, optional): A function or
        series of functions to apply transformations to the images.
    """
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.samples = sorted(self.samples)
        self.transform = transform

    def _make_dataset(self, root):

        domains = os.listdir(root)
        fnames, labels = [], []

        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)

            # print(cls_fnames)
            fnames += cls_fnames
            labels += [idx] * len(cls_fnames)

        # print(fnames)

        return fnames, labels

    def __getitem__(self, index):
        fname = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.targets)

def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def get_eval_loader(root, img_size=128, batch_size=8,
                    imagenet_normalize=False, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.Resize([height, width]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=8,
                    num_workers=4,  shuffle=True):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                      std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)
class InputFetcher:
    """
       A utility class for fetching batches of data from PyTorch DataLoader instances.
       The `InputFetcher` class allows seamless iteration over two data loaders:
       a primary loader (`loader`) and an optional reference loader (`loader_ref`).
       It handles the iteration process, fetching input data and resetting the iterator
       when the loader exhausts.
       Attributes:
           loader (torch.utils.data.DataLoader): The primary DataLoader for fetching input data.
           loader_ref (torch.utils.data.DataLoader, optional):
           The reference DataLoader for fetching reference data. Defaults to None.
           latent_dim (int): The dimension of the latent vector space. Defaults to 16.
           device (torch.device): The device where tensors will be moved (either 'cuda' or 'cpu').
           mode (str): Mode of operation. It can be used to
           specify the input-fetching mode or other custom operations.
    """

    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x,  y = next(self.iter_ref)
        return x, y

    def _fetch_ref_m(self):
        try:
            x, fnmr = next(self.iter_d_m)
        except (AttributeError, StopIteration):
            self.iter_d_m = iter(self.loader)
            x, fnmr = next(self.iter_d_m)
        return x, fnmr

    def _fetch_d_m(self):
        try:
            x, fnmd = next(self.iter_ref_m)
        except (AttributeError, StopIteration):
            self.iter_ref_m = iter(self.loader_ref)
            x, fnmd = next(self.iter_ref_m)
        return x, fnmd

    def loader_Match(self, f_r, f_d):

        base_f_r = os.path.basename(f_r[0])
        base_f_d = os.path.basename(f_d[0])
        try:
            if len(self.loader.dataset) != len(self.loader_ref.dataset):
                print(f" Distortion sets >> '{len(self.loader.dataset)}"
                      f"'is compared against single Reference Image  >> '{len(self.loader_ref.dataset)}' ")

            if base_f_r != base_f_d:
                print(f" Reference '{base_f_r}' is compared against '{base_f_d}' ")

        except(AttributeError, StopIteration):
            print("Datasets are NOT equal")

    def __next__(self):

        if self.mode == 'train':
            x, y = self._fetch_inputs()
            x_ref, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref, x_ref=x_ref, z_trg=z_trg)

        elif self.mode == 'val':
            x, y = self._fetch_inputs()
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y, x_ref=x_ref, y_ref=y_ref)

        elif self.mode == 'test_measure':
            x_r, f_nmr = self._fetch_ref_m()
            x_d, f_nmd = self._fetch_d_m()
            self.loader_Match(f_nmr, f_nmd)
            inputs = Munch(x_ref=x_r, x_dist=x_d, ref_fname=f_nmr, dist_fname=f_nmd)

        else:

            raise NotImplementedError

        return Munch({k: v for k, v in inputs.items()})


class InputFetcher_Val:
    """
           A utility class for fetching batches of data from PyTorch DataLoader instances.
           The `InputFetcher` class allows seamless iteration over two data loaders:
           a primary loader (`loader`) and an optional reference loader (`loader_ref`).
           It handles the iteration process, fetching input data and resetting the iterator
           when the loader exhausts.
       Attributes:
           loader (torch.utils.data.DataLoader): The primary DataLoader for fetching input data.
           loader_ref (torch.utils.data.DataLoader, optional): The reference
           DataLoader for fetching reference data. Defaults to None.
           latent_dim (int): The dimension of the latent vector space. Defaults to 16.
           device (torch.device): The device where tensors will be moved (either 'cuda' or 'cpu').
           mode (str): Mode of operation. It can be used to specify
           the input-fetching mode or other custom operations.
    """
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x= next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x = next(self.iter)

        return x

    def _fetch_refs(self):
        try:
            x  = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x = next(self.iter_ref)

        return x

    def __next__(self):
        x  = self._fetch_inputs()

        if self.mode == 'train':
            x_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, x_ref=x_ref, z_trg=z_trg)

        elif self.mode == 'val':
            x_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, x_ref=x_ref)

        elif self.mode == 'test_measure':
            inputs = Munch(x=x,)

        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})