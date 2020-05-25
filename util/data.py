import pdb
import sys
import os
import math
import torch
import numpy as np
from torchvision.datasets import VisionDataset
from torchvision import transforms
from PIL import Image
from random import random

from config import dataset_config as config

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_video_clips(video_paths, stride, video_frames):
    result = []
    last_index = len(video_paths) - (stride + 1) * (video_frames - 1)
    for start_index in range(last_index):
        indices = range(start_index, start_index + (stride + 1) * (video_frames - 1) + 1, stride + 1)
        result.append([video_paths[index] for index in indices])
    return result


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None, video_frames=1):
    videos = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:

        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            paths = []
            for i, fname in enumerate(sorted(fnames)):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    paths.append(path)
                if not len(paths) == 0 and len(paths) % video_frames == 0:
                    videos.append((paths, class_to_idx[target]))
                    paths = []
    return videos


class Mean(object):
    def __call__(self, tensor):
        return torch.mean(tensor, dim=0, keepdim=True)


class HorizontalFlip(object):
    def __init__(self, p=0.6):
        self.p = p

    def __call__(self, nparray):
        if random() > self.p:
            return np.flip(nparray, axis=2).copy()
        return nparray


default_transform = transforms.Compose([
    transforms.Resize(config['resolution']),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


class PreprocessFolder(VisionDataset):
    def __init__(self,
                 root,
                 flow_roots=config["flow_roots"],
                 bbox_roots=config["bbox_roots"],
                 loader=default_loader,
                 extensions=config["extension"],
                 target_transform=None,
                 is_valid_file=None,
                 transform=default_transform,
                 overall_transform=None):
        super(PreprocessFolder, self).__init__(root)
        self.target_transform = target_transform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file, 1)

        self.extensions = extensions
        self.flow_roots = flow_roots
        self.bbox_roots = bbox_roots
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.overall_transform = overall_transform() if overall_transform else None

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target)
                sample (tensor): Frame, Height, Width, Channel
                target (int): belongs to which video
        """
        paths, target = self.samples[index]

        img_path = paths[0]
        img_name = os.path.splitext(paths[0])[0]

        npy_name = '/'.join(img_name.split('/')[-3:]) + '.npy'

        img = self.loader(img_path)
        img = self.transform(img)

        flow_path = os.path.join(self.flow_roots, npy_name)
        flow = torch.tensor(np.load(flow_path))
        sample = torch.cat([img, flow], 0)

        bbox_path = os.path.join(self.bbox_roots, npy_name)
        bbox = torch.tensor(np.load(bbox_path))

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.overall_transform:
            sample = self.transform(sample)

        return sample, bbox, target

    def __len__(self):
        return len(self.samples)