import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import os.path
from utils import combine_transforms as ctr
from PIL import Image
from torchvision import transforms


###############################################################################
# Dataset sets
###############################################################################

# dataset for reflection removal
class DatasetRR(data.Dataset):
    def __init__(self, data_opt, is_train=None, is_aligned=False):
        self.opt = data_opt
        self.root = data_opt.data_root
        self.dir_I = os.path.join(self.root, 'blended')
        self.dir_B = os.path.join(self.root, 'transmission')
        self.dir_R = os.path.join(self.root, 'reflection')
        self.is_aligned = is_aligned

        is_train = data_opt.is_train if is_train is None else is_train
        if is_train:
            self.length = 4000
        else:
            self.length = 1000

        self.I_paths = make_dataset(self.dir_I)
        self.B_paths = make_dataset(self.dir_B)
        self.R_paths = make_dataset(self.dir_R)

        self.I_paths = sorted(self.I_paths)
        self.B_paths = sorted(self.B_paths)
        self.R_paths = sorted(self.R_paths)
        self.I_size = len(self.I_paths)
        self.B_size = len(self.B_paths)
        self.R_size = len(self.R_paths)
        if not self.opt.serial_batches:
            self.transform = get_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                           new_size=data_opt.new_size, is_train=data_opt.is_train,
                                           no_flip=data_opt.no_flip,
                                           image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                           use_norm=data_opt.use_norm)
        else:
            self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                   new_size=data_opt.new_size, is_train=data_opt.is_train,
                                                   no_flip=data_opt.no_flip,
                                                   image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                   use_norm=data_opt.use_norm)

    def __getitem__(self, index):
        index_i = index % self.I_size
        path_i = self.I_paths[index_i]
        if self.opt.serial_batches:
            index_b = index % self.B_size
            index_r = index % self.R_size
        else:
            index_b = random.randint(0, self.B_size - 1)
            index_r = random.randint(0, self.R_size - 1)
        path_b = self.B_paths[index_b]
        path_r = self.R_paths[index_r]

        img_i = Image.open(path_i).convert('RGB')
        img_b = Image.open(path_b).convert('RGB')
        img_r = Image.open(path_r).convert('RGB')

        if not self.opt.serial_batches:
            img_i = self.transform(img_i)
            img_b = self.transform(img_b)
            img_r = self.transform(img_r)
            ret = {'I': img_i, 'B': img_b, 'R': img_r}
        else:
            ret = self.transform({'I': img_i, 'B': img_b, 'R': img_r})

        ret['name'] = path_i

        return ret

    def __len__(self):
        return self.length

    @staticmethod
    def name():
        return 'UnalignedDatasetRR'

    pass


# dataset for toy problem
class DatasetToy(data.Dataset):
    def __init__(self, data_opt, is_train=None):
        self.opt = data_opt
        self.use_color = data_opt.use_color
        if not self.opt.serial_batches:
            self.transform = get_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                           new_size=data_opt.new_size, is_train=data_opt.is_train,
                                           no_flip=data_opt.no_flip,
                                           image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                           use_norm=data_opt.use_norm)
        else:
            self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                   new_size=data_opt.new_size, is_train=data_opt.is_train,
                                                   no_flip=data_opt.no_flip,
                                                   image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                   use_norm=data_opt.use_norm)
        self.canvas_size = data_opt.canvas_size
        self.shapes = ['square', 'circle']

        is_train = data_opt.is_train if is_train is None else is_train
        if is_train:
            self.length = 4000
        else:
            self.length = 1000

    def _gen_layers(self):
        layers = []
        for idx, shape in enumerate(self.shapes):
            canvas = np.zeros(shape=(self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
            color = (np.random.randint(50, 220), np.random.randint(50, 220), np.random.randint(50, 220))
            is_fill = -1 if np.random.rand() > 0.5 else 1
            h, w, _ = canvas.shape
            size = int((0.1 + np.random.rand() * 0.5) * w + 0.5)
            thick = size // 20
            point_pos = (int(w * np.random.rand() * 0.6), int(h * np.random.rand() * 0.6))
            thick = 3 if thick < 3 else thick
            thick = -1 if is_fill else thick
            if shape == 'square':
                pt1 = point_pos
                pt2_x = point_pos[0] + size
                pt2_x = w if pt2_x > w else pt2_x
                pt2_y = point_pos[1] + size
                pt2_y = h if pt2_y > h else pt2_y
                pt2 = (pt2_x, pt2_y)
                cv2.rectangle(canvas, pt1, pt2, color, thick)
            elif shape == 'circle':
                cv2.circle(canvas, point_pos, size, color, thick)

            layers.append(canvas)

        alpha = 0.2 + np.random.rand() * 0.6
        layers[0] = np.asarray(layers[0] * alpha, np.uint8)
        layers[1] = np.asarray(layers[1] * (1 - alpha), np.uint8)

        if not self.use_color:
            layers[0] = cv2.cvtColor(layers[0], cv2.COLOR_BGR2GRAY)
            layers[1] = cv2.cvtColor(layers[1], cv2.COLOR_BGR2GRAY)
            if self.opt.input_dim_a > 1:
                layers[0] = cv2.cvtColor(layers[0], cv2.COLOR_GRAY2BGR)
                layers[1] = cv2.cvtColor(layers[1], cv2.COLOR_GRAY2BGR)

        return layers

    def __getitem__(self, index):

        layers = self._gen_layers()

        blended = layers[0] + layers[1]

        if not self.opt.serial_batches:
            layers = self._gen_layers()

        img_i = Image.fromarray(blended)
        img_b = Image.fromarray(layers[0])
        img_r = Image.fromarray(layers[1])

        if not self.opt.serial_batches:
            img_i = self.transform(img_i)
            img_b = self.transform(img_b)
            img_r = self.transform(img_r)

            sample = {'I': img_i, 'B': img_b, 'R': img_r}
        else:
            sample = {'I': img_i, 'B': img_b, 'R': img_r}
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.length

    def name(self):
        return 'DatasetToy'


# dataset for toy problem
class DatasetToy3(data.Dataset):
    def __init__(self, data_opt, is_train=None):
        self.opt = data_opt
        self.use_color = data_opt.use_color
        if not self.opt.serial_batches:
            self.transform = get_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                           new_size=data_opt.new_size, is_train=data_opt.is_train,
                                           no_flip=data_opt.no_flip,
                                           image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                           use_norm=data_opt.use_norm)
        else:
            self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                   new_size=data_opt.new_size, is_train=data_opt.is_train,
                                                   no_flip=data_opt.no_flip,
                                                   image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                   use_norm=data_opt.use_norm)
        self.canvas_size = data_opt.canvas_size
        self.shapes = ['square', 'circle', 'triangle']

        is_train = data_opt.is_train if is_train is None else is_train
        if is_train:
            self.length = 4000
        else:
            self.length = 1000

    def _gen_layers(self):
        layers = []
        for idx, shape in enumerate(self.shapes):
            canvas = np.zeros(shape=(self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
            color = (np.random.randint(50, 220), np.random.randint(50, 220), np.random.randint(50, 220))
            is_fill = -1 if np.random.rand() > 0.5 else 1
            h, w, _ = canvas.shape
            size = int((0.1 + np.random.rand() * 0.5) * w + 0.5)
            thick = size // 20
            point_pos = (int(w * np.random.rand() * 0.6), int(h * np.random.rand() * 0.6))
            thick = 3 if thick < 3 else thick
            thick = -1 if is_fill else thick
            if shape == 'square':
                pt1 = point_pos
                pt2_x = point_pos[0] + size
                pt2_x = w if pt2_x > w else pt2_x
                pt2_y = point_pos[1] + size
                pt2_y = h if pt2_y > h else pt2_y
                pt2 = (pt2_x, pt2_y)
                cv2.rectangle(canvas, pt1, pt2, color, thick)
            elif shape == 'circle':
                cv2.circle(canvas, point_pos, size, color, thick)
            elif shape == 'triangle':
                angle = np.random.random() * 2 * np.pi
                pp = point_pos
                # calculate the three points
                pos1 = (pp[0] + size * np.sin(angle + 0 / 3 * np.pi), pp[1] + size * np.cos(angle + 0 / 3 * np.pi))
                pos2 = (pp[0] + size * np.sin(angle + 2 / 3 * np.pi), pp[1] + size * np.cos(angle + 2 / 3 * np.pi))
                pos3 = (pp[0] + size * np.sin(angle + 4 / 3 * np.pi), pp[1] + size * np.cos(angle + 4 / 3 * np.pi))
                triangle = np.array([pos1, pos2, pos3], np.int32)

                cv2.fillConvexPoly(canvas, triangle, color=color)

            layers.append(canvas)

        layers = [np.asarray(l / 3, np.uint8) for l in layers]

        if not self.use_color:
            layers[0] = cv2.cvtColor(layers[0], cv2.COLOR_BGR2GRAY)
            layers[1] = cv2.cvtColor(layers[1], cv2.COLOR_BGR2GRAY)
            layers[2] = cv2.cvtColor(layers[2], cv2.COLOR_BGR2GRAY)

        return layers

    def __getitem__(self, index):

        layers = self._gen_layers()

        blended = layers[0] + layers[1] + layers[2]

        if not self.opt.serial_batches:
            layers = self._gen_layers()

        img_i = Image.fromarray(blended)
        img_b = Image.fromarray(layers[0])
        img_r = Image.fromarray(layers[1])
        img_r2 = Image.fromarray(layers[2])

        if not self.opt.serial_batches:
            img_i = self.transform(img_i)
            img_b = self.transform(img_b)
            img_r = self.transform(img_r)
            img_r2 = self.transform(img_r2)

            sample = {'I': img_i, 'B': img_b, 'R': img_r, 'R2': img_r2}
        else:
            sample = {'I': img_i, 'B': img_b, 'R': img_r, 'R2': img_r2}
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.length

    def name(self):
        return 'DatasetToy3'


class DatasetIdMIT(data.Dataset):
    def __init__(self, data_opt, is_train=None):
        self.opt = data_opt
        self.root = data_opt.data_root
        self.is_train = data_opt.is_train if is_train is None else is_train

        self.I_paths = []
        self.B_paths = []
        self.R_paths = []

        root_i = os.path.join(self.root, 'MIT', 'MIT-input')
        root_b = os.path.join(self.root, 'MIT', 'MIT-reflectance')
        root_r = os.path.join(self.root, 'MIT', 'MIT-shading')

        fname = 'train.txt' if self.is_train else 'test.txt'

        with open(os.path.join(self.root, 'MIT', fname), 'r') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip()
                self.I_paths.append(os.path.join(root_i, line))
                self.B_paths.append(os.path.join(root_b, line))
                self.R_paths.append(os.path.join(root_r, line))

            if not self.opt.serial_batches:
                self.transform = get_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                               new_size=data_opt.new_size, is_train=data_opt.is_train,
                                               no_flip=data_opt.no_flip,
                                               image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                               use_norm=data_opt.use_norm)
            else:
                self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                       new_size=data_opt.new_size, is_train=data_opt.is_train,
                                                       no_flip=data_opt.no_flip,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)

    def __len__(self):
        return len(self.I_paths)

    def _transform_image(self, image, seed=None):
        if self.transform is not None:
            seed = np.random.randint(100, 500000) if seed is None else seed
            torch.manual_seed(seed)
            random.seed = seed
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        if not self.opt.serial_batches:
            random.shuffle(self.I_paths)
            random.shuffle(self.B_paths)
            random.shuffle(self.R_paths)

        ret_dict = {
            'I': Image.open(self.I_paths[index]).convert('RGB'),
            'B': Image.open(self.B_paths[index]).convert('RGB'),
            'R': Image.open(self.R_paths[index]).convert('RGB'),
        }

        if not self.opt.serial_batches:
            ret_dict['I'] = self._transform_image(ret_dict['I'])
            ret_dict['B'] = self._transform_image(ret_dict['B'])
            ret_dict['R'] = self._transform_image(ret_dict['R'])
        else:
            ret_dict = self.transform(ret_dict)

        ret_dict['name'] = self.I_paths[index]

        return ret_dict


class DatasetIdMPI(data.Dataset):
    def __init__(self, data_opt, is_train=None):
        self.opt = data_opt
        self.root = data_opt.data_root
        self.is_train = data_opt.is_train if is_train is None else is_train

        self.I_paths = []
        self.B_paths = []
        self.R_paths = []

        root_i = os.path.join(self.root, 'MPI', 'MPI-auxilliary-input')
        root_b = os.path.join(self.root, 'MPI', 'MPI-auxilliary-albedo')
        root_r = os.path.join(self.root, 'MPI', 'MPI-auxilliary-shading')

        fname = 'train.txt' if self.is_train else 'test.txt'

        with open(os.path.join(self.root, 'MPI', fname), 'r') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip()
                self.I_paths.append(os.path.join(root_i, line))
                self.B_paths.append(os.path.join(root_b, line))
                self.R_paths.append(os.path.join(root_r, line))

            if not self.opt.serial_batches:
                self.transform = get_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                               new_size=data_opt.new_size, is_train=data_opt.is_train,
                                               no_flip=data_opt.no_flip,
                                               image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                               use_norm=data_opt.use_norm)
            else:
                self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                       new_size=data_opt.new_size, is_train=data_opt.is_train,
                                                       no_flip=data_opt.no_flip,
                                                       image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                       use_norm=data_opt.use_norm)

    def __len__(self):
        return len(self.I_paths)

    def _transform_image(self, image, seed=None):
        if self.transform is not None:
            seed = np.random.randint(100, 500000) if seed is None else seed
            torch.manual_seed(seed)
            random.seed = seed
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        seed = np.random.randint(100, 500000)
        if not self.opt.serial_batches:
            random.shuffle(self.I_paths)
            random.shuffle(self.B_paths)
            random.shuffle(self.R_paths)
            seed = None

        ret_dict = {
            'I': Image.open(self.I_paths[index]).convert('RGB'),
            'B': Image.open(self.B_paths[index]).convert('RGB'),
            'R': Image.open(self.R_paths[index]).convert('RGB'),
        }

        if not self.opt.serial_batches:
            ret_dict['I'] = self._transform_image(ret_dict['I'])
            ret_dict['B'] = self._transform_image(ret_dict['B'])
            ret_dict['R'] = self._transform_image(ret_dict['R'])
        else:
            ret_dict = self.transform(ret_dict)

        ret_dict['name'] = self.I_paths[index]

        return ret_dict


###############################################################################
# Fast functions
###############################################################################

def get_transform(name, load_size=300, new_size=256, is_train=True, no_flip=False,
                  image_mean=(0.4914, 0.4822, 0.4465), image_std=(0.2023, 0.1994, 0.2010), use_norm=True):
    transform_list = []
    if name == 'resize_and_crop':
        o_size = [load_size, load_size]
        transform_list.append(transforms.Scale(o_size, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(new_size))
    elif name == 'crop':
        transform_list.append(transforms.RandomCrop(new_size))
    elif name == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, load_size)))
    elif name == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, load_size)))
        transform_list.append(transforms.RandomCrop(new_size))

    if is_train and not no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor()]
    if use_norm:
        transform_list += [transforms.Normalize(image_mean, image_std)]
    return transforms.Compose(transform_list)


def get_combine_transform(name, load_size=300, new_size=256, is_train=True, no_flip=False,
                          image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), use_norm=True):
    transform_list = []
    if name == 'resize_and_crop':
        # o_size = [load_size, load_size]
        # transform_list.append(transforms.Scale(o_size, Image.BICUBIC))
        transform_list.append(ctr.RandomScaleCrop(load_size, new_size))
    elif name == 'crop':
        transform_list.append(ctr.RandomScaleCrop(load_size, new_size))
    elif name == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, load_size)))
    elif name == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, load_size)))
        transform_list.append(transforms.RandomCrop(new_size))

    if is_train and not no_flip:
        transform_list.append(ctr.RandomHorizontalFlip())

    transform_list += [ctr.ToTensor()]

    if use_norm:
        transform_list += [ctr.Normalize(image_mean, image_std)]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
###############################################################################

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(s_dir):
    images = []
    assert os.path.isdir(s_dir), '%s is not a valid directory' % s_dir

    for root, _, fnames in sorted(os.walk(s_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class ImageFileList(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFileList(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

