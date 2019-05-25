import torch
import random
import numpy as np
from torchvision.transforms .functional import normalize

from PIL import Image, ImageOps


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        img_in = normalize(img_in, self.mean, self.std)
        img_bg = normalize(img_bg, self.mean, self.std)
        img_rf = normalize(img_rf, self.mean, self.std)

        if 'R2' in sample:
            img_rf2 = sample['R2']
            img_rf2 = normalize(img_rf2, self.mean, self.std)

            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf,
                    'R2': img_rf2}

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        img_bg = np.array(img_bg).astype(np.float32) / 255.
        img_in = np.array(img_in).astype(np.float32) / 255.
        img_rf = np.array(img_rf).astype(np.float32) / 255.

        img_shape = img_in.shape

        if len(img_shape) == 3:
            img_in = img_in.transpose((2, 0, 1))
            img_bg = img_bg.transpose((2, 0, 1))
            img_rf = img_rf.transpose((2, 0, 1))

            img_in = torch.from_numpy(img_in).float()
            img_bg = torch.from_numpy(img_bg).float()
            img_rf = torch.from_numpy(img_rf).float()
        else:
            img_in = torch.from_numpy(img_in).float().unsqueeze(0)
            img_bg = torch.from_numpy(img_bg).float().unsqueeze(0)
            img_rf = torch.from_numpy(img_rf).float().unsqueeze(0)

        if 'R2' in sample:
            img_rf2 = sample['R2']
            img_rf2 = np.array(img_rf2).astype(np.float32)
            if len(img_shape) == 3:
                img_rf2 = img_rf2.transpose((2, 0, 1))
                img_rf2 = torch.from_numpy(img_rf2).float() / 255.
            else:
                img_rf2 = torch.from_numpy(img_rf2).float().unsqueeze(0) / 255.

            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf,
                    'R2': img_rf2}

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        rand_num = random.random()
        if rand_num < 0.5:
            img_in = img_in.transpose(Image.FLIP_LEFT_RIGHT)
            img_bg = img_bg.transpose(Image.FLIP_LEFT_RIGHT)
            img_rf = img_rf.transpose(Image.FLIP_LEFT_RIGHT)

        if 'R2' in sample:
            img_rf2 = sample['R2']
            if rand_num < 0.5:
                img_rf2 = img_rf2.transpose(Image.FLIP_LEFT_RIGHT)

            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf,
                    'R2': img_rf2}

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        rotate_degree = random.uniform(-1 * self.degree, self.degree)

        img_in = img_in.rotate(rotate_degree, Image.BILINEAR)
        img_bg = img_bg.rotate(rotate_degree, Image.BILINEAR)
        img_rf = img_rf.rotate(rotate_degree, Image.BILINEAR)

        if 'R2' in sample:
            img_rf2 = sample['R2']
            img_rf2 = img_rf2.rotate(rotate_degree, Image.BILINEAR)

            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf,
                    'R2': img_rf2}

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img_in.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        padh = self.crop_size - oh if oh < self.crop_size else 0
        padw = self.crop_size - ow if ow < self.crop_size else 0
        # pad crop
        if short_size < self.crop_size:
            img_in = ImageOps.expand(img_in, border=(0, 0, padw, padh), fill=0)
            img_bg = ImageOps.expand(img_bg, border=(0, 0, padw, padh), fill=0)
            img_rf = ImageOps.expand(img_rf, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img_in.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if 'R2' in sample:
            img_rf2 = sample['R2']
            img_rf2 = img_rf2.resize((ow, oh), Image.BILINEAR)
            if short_size < self.crop_size:
                img_rf2 = ImageOps.expand(img_rf2, border=(0, 0, padw, padh), fill=0)
            img_rf2 = img_rf2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf,
                    'R2': img_rf2}

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        w, h = img_in.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img_in.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if 'R2' in sample:
            img_rf2 = sample['R2']
            img_rf2 = img_rf2.resize((ow, oh), Image.BILINEAR)
            img_rf2 = img_rf2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf,
                    'R2': img_rf2}

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        img_in = img_in.resize(self.size, Image.BILINEAR)
        img_bg = img_bg.resize(self.size, Image.BILINEAR)
        img_rf = img_rf.resize(self.size, Image.BILINEAR)

        if 'R2' in sample:
            img_rf2 = sample['R2']
            img_rf2 = img_rf2.resize(self.size, Image.BILINEAR)

            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf,
                    'R2': img_rf2}

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixedScalePadding(object):
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        w, h = img_in.size
        if h < w:
            ow = self.size
            oh = int(1.0 * h * ow / w)
        else:
            oh = self.size
            ow = int(1.0 * w * oh / h)
        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        pad_h = self.size - oh if oh < self.size else 0
        pad_w = self.size - ow if ow < self.size else 0

        img_in = ImageOps.expand(img_in, border=(0, 0, pad_w, pad_h), fill=0)
        img_bg = ImageOps.expand(img_bg, border=(0, 0, pad_w, pad_h), fill=0)
        img_rf = ImageOps.expand(img_rf, border=(0, 0, pad_w, pad_h), fill=0)

        if 'R2' in sample:
            img_rf2 = sample['R2']
            img_rf2 = img_rf2.resize((ow, oh), Image.BILINEAR)
            img_rf2 = ImageOps.expand(img_rf2, border=(0, 0, pad_w, pad_h), fill=0)

            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf,
                    'R2': img_rf2}

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}
