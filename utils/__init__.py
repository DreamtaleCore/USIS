import os
import shutil
import time

import cv2
import math
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

"""
@tensor2img                 :   convert Variable or tensor into image with un-normalized operation
@show_image                 :   show an image with opencv
@check_dir                  :   if a dir is not exist, create it
@remove_dir                 :   if a dir is exist, remove it
@get_dataloader
@get_scheduler
@print_networ
@weights_init
"""


def tensor2img(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), use_norm=True):
    if type(tensor) is Variable or tensor.is_cuda:
        tensor = tensor.cpu().data
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    to_pil = transforms.ToPILImage()
    n_channel = tensor.shape[0]
    if n_channel == 3 and use_norm:
        # color image with normalization [-x, y] ==> [0, 1]
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1
    pil_img = to_pil(tensor)
    img = np.asarray(pil_img, np.uint8)
    return img


def show_image(image, name='image', delay=0):
    cv2.imshow(name, image)

    c = cv2.waitKey(delay)
    if chr(c & 255) == 'q':
        return False     # need to exist
    else:
        return True      # keep going


def check_dir(s_dir):
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)


def remove_dir(s_dir):
    if os.path.exists(s_dir):
        shutil.rmtree(s_dir)


def get_data_loaders(opt):
    batch_size = opt.batch_size
    num_workers = opt.num_workers

    if opt.name == 'rr_removal':
        from data import DatasetRR
        train_set = DatasetRR(data_opt=opt, is_train=True)
        test_set = DatasetRR(data_opt=opt, is_train=False)
    elif opt.name == 'toy':
        from data import DatasetToy
        train_set = DatasetToy(data_opt=opt, is_train=True)
        test_set = DatasetToy(data_opt=opt, is_train=False)
    elif opt.name == 'toy3':
        from data import DatasetToy3
        train_set = DatasetToy3(data_opt=opt, is_train=True)
        test_set = DatasetToy3(data_opt=opt, is_train=False)
    elif opt.name == 'intrinsic':
        from data import DatasetIdMIT, DatasetIdMPI
        if 'mit' in opt.sub_name.lower():
            train_set = DatasetIdMIT(data_opt=opt, is_train=True)
            test_set = DatasetIdMIT(data_opt=opt, is_train=False)
        else:
            train_set = DatasetIdMPI(data_opt=opt, is_train=True)
            test_set = DatasetIdMPI(data_opt=opt, is_train=False)
    else:
        raise NotImplementedError

    print('==> Using dataset ', type(train_set).__name__)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_dataset(opt):
    if opt.name == 'rr_removal':
        from data import DatasetRR
        train_set = DatasetRR(data_opt=opt, is_train=True)
        test_set = DatasetRR(data_opt=opt, is_train=False)
    elif opt.name == 'toy':
        from data import DatasetToy
        train_set = DatasetToy(data_opt=opt, is_train=True)
        test_set = DatasetToy(data_opt=opt, is_train=False)
    elif opt.name == 'toy3':
        from data import DatasetToy3
        train_set = DatasetToy3(data_opt=opt, is_train=True)
        test_set = DatasetToy3(data_opt=opt, is_train=False)
    elif opt.name == 'intrinsic':
        from data import DatasetIdMIT, DatasetIdMPI
        if 'mit' in opt.sub_name.lower():
            train_set = DatasetIdMIT(data_opt=opt, is_train=True)
            test_set = DatasetIdMIT(data_opt=opt, is_train=False)
        else:
            train_set = DatasetIdMPI(data_opt=opt, is_train=True)
            test_set = DatasetIdMPI(data_opt=opt, is_train=False)

    return train_set, test_set


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.n_iter) / float(opt.n_iter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.n_iter_decay, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network_info(net, print_struct=True):
    print('Network %s structure: ' % type(net).__name__)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if print_struct:
        print(net)
    print('===> In network %s, total trainable number of parameters: %d' % (type(net).__name__, num_params))


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
