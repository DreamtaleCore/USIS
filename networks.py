import math
import torch
import functools
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as ta_grad
import torch.utils.model_zoo as model_zoo
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


##################################################################################
# Interfaces
##################################################################################

def get_discriminator(dis_opt, train_mode):
    """Get a discriminator, dis_opt is options of dis, train mode determine whether use multi-scale or not"""
    return MsImageDis(dis_opt)


def get_generator(gen_opt, train_mode):
    """Get a generator, gen_opt is options of gen, train mode determine which generator model to use"""
    if 'vgg' in train_mode:
        return CrossGenMS(gen_opt)
    else:
        if '3' in train_mode:
            return PsiUnetGenMS(gen_opt)
        else:
            return XUnetGenMS(gen_opt)


def get_manual_criterion(name, params=None):
    if 'distance' in name.lower():
        return DistanceLoss()
    elif 'gan' in name.lower():
        return GANLoss(tensor=params['tensor'], use_lsgan=params['use_lsgan'])
    elif 'identity' in name.lower():
        return IdentityLoss()
    elif 'ssim' in name.lower():
        return SSIMLoss()
    else:
        raise NotImplementedError(name + 'should in [distance/gan/identity/ssim...]')


##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, dis_opt):
        super(MsImageDis, self).__init__()
        self.dim = dis_opt.dim
        self.norm = dis_opt.norm
        self.activ = dis_opt.activ
        self.grad_w = dis_opt.grad_w
        self.pad_type = dis_opt.pad_type
        self.gan_type = dis_opt.gan_type
        self.n_layers = dis_opt.n_layers
        self.use_grad = dis_opt.use_grad
        self.input_dim = dis_opt.input_dim
        self.num_scales = dis_opt.num_scales
        self.use_wasserstein = dis_opt.use_wasserstein
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.models = nn.ModuleList()
        self.sigmoid_func = nn.Sigmoid()

        for _ in range(self.num_scales):
            cnns = self._make_net()
            if self.use_wasserstein:
                cnns += [nn.Sigmoid()]

            self.models.append(nn.Sequential(*cnns))

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layers - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        return cnn_x

    def forward(self, x):
        output = None
        for model in self.models:
            out = model(x)
            if output is not None:
                _, _, h, w = out.shape
                output = F.interpolate(output, size=(h, w), mode='bilinear')
                output = output + out
            else:
                output = out

            x = self.downsample(x)

        output = output / len(self.models)
        output = self.sigmoid_func(output)

        return output

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

            # Gradient penalty
            grad_loss = 0
            if self.use_grad:
                eps = Variable(torch.rand(1), requires_grad=True)
                eps = eps.expand(input_real.size())
                eps = eps.cuda()
                x_tilde = eps * input_real + (1 - eps) * input_fake
                x_tilde = x_tilde.cuda()
                pred_tilde = self.calc_gen_loss(x_tilde)
                gradients = ta_grad(outputs=pred_tilde, inputs=x_tilde,
                                    grad_outputs=torch.ones(pred_tilde.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                grad_loss = self.grad_w * gradients

                input_real = self.downsample(input_real)
                input_fake = self.downsample(input_fake)

            loss += ((grad_loss.norm(2, dim=1) - 1) ** 2).mean()

        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).to(self.device), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Generator
##################################################################################

class CrossGenMS(nn.Module):
    def __init__(self, gen_opt):
        super(CrossGenMS, self).__init__()
        self.dim = gen_opt.dim
        self.norm = gen_opt.norm
        self.mode = gen_opt.mode
        self.activ = gen_opt.activ
        self.pad_type = gen_opt.pad_type
        self.n_layers = gen_opt.n_layers
        self.input_dim = gen_opt.input_dim
        self.pretrained = gen_opt.vgg_pretrained
        self.feature_dim = gen_opt.feature_dim
        self.output_dim1 = gen_opt.output_dim_b
        self.output_dim2 = gen_opt.output_dim_c
        self.encoder_name = gen_opt.encoder_name

        # Feature extractor as Encoder
        if self.encoder_name == 'vgg11':
            self.encoder_a = Vgg11EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
            self.encoder_b = Vgg11EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
        else:
            self.encoder_a = Vgg19EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
            self.encoder_b = Vgg19EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)

        self.decoder_a = DecoderMS(self.input_dim, dim=self.dim, output_dim=self.output_dim1,
                                   n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                   norm=self.norm)
        self.decoder_b = DecoderMS(self.input_dim, dim=self.dim, output_dim=self.output_dim1,
                                   n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                   norm=self.norm)
        self.fusion = Conv2dBlock(input_dim=1024, output_dim=512, kernel_size=1, stride=1)

    def decode(self, x, name, feats=None):
        if name == 'a':
            return self.decoder_a(x, feats)
        elif name == 'b':
            return self.decoder_b(x, feats)
        else:
            raise NotImplementedError

    def encode(self, x, name):
        if name == 'a':
            return self.encoder_a(x)
        else:
            return self.encoder_b(x)

    def forward(self, x):
        feats_a = self.encoder_a(x)
        feats_b = self.encoder_b(x)
        if self.mode == 'fusion':
            features_a = torch.cat([feats_a['out'], feats_b['out']], dim=1)
            features_a = self.fusion(features_a)
            features_b = torch.cat([feats_b['out'], feats_a['out']], dim=1)
            features_b = self.fusion(features_b)
        elif self.mode == 'minus':
            features_a = torch.cat([feats_a['out'] - feats_b['out']], dim=1)
            features_b = torch.cat([feats_b['out'] - feats_a['out']], dim=1)
        elif self.mode == 'direct':
            features_a = feats_a
            features_b = feats_b
        else:
            raise NotImplementedError

        out_a = self.decode(x, 'a', features_a)
        out_b = self.decode(x, 'b', features_b)
        return out_a, out_b, feats_a, feats_b


class XUnetGenMS(nn.Module):
    """Extract with procedure features"""

    def __init__(self, gen_opt):
        super(XUnetGenMS, self).__init__()
        self.dim = gen_opt.dim
        self.norm = gen_opt.norm
        self.mode = gen_opt.mode
        self.activ = gen_opt.activ
        self.n_layers = gen_opt.n_layers
        self.input_dim = gen_opt.input_dim
        self.output_dim_b = gen_opt.output_dim_b
        self.output_dim_c = gen_opt.output_dim_c
        self.use_dropout = gen_opt.use_dropout

        self.feat_id_map = {int(self.n_layers * 1 / 4): 'low',
                            int(self.n_layers * 2 / 4): 'mid',
                            int(self.n_layers * 3 / 4): 'deep',
                            int(self.n_layers * 4 / 4): 'out'
                            }

        self.encoder_b = UnetEncoder(input_nc=self.input_dim, dim=self.dim, n_layers=self.n_layers,
                                     norm_layer=self.norm, use_dropout=self.use_dropout, activ=self.activ)
        self.encoder_c = UnetEncoder(input_nc=self.input_dim, dim=self.dim, n_layers=self.n_layers,
                                     norm_layer=self.norm, use_dropout=self.use_dropout, activ=self.activ)
        self.decoder_b = UnetDecoder(output_nc=self.output_dim_b, dim=self.dim, n_layers=self.n_layers,
                                     norm_layer=self.norm, use_dropout=self.use_dropout, activ=self.activ)
        self.decoder_c = UnetDecoder(output_nc=self.output_dim_c, dim=self.dim, n_layers=self.n_layers,
                                     norm_layer=self.norm, use_dropout=self.use_dropout, activ=self.activ)

    def forward(self, x):
        en_feats_b = self.encoder_b(x)
        en_feats_c = self.encoder_c(x)
        de_feats_b = self.decoder_b(en_feats_b)
        de_feats_c = self.decoder_c(en_feats_c)
        out_feats_b = {}
        out_feats_c = {}
        for idx, key in enumerate(self.encoder_b.model_dict.keys()):
            if idx in self.feat_id_map:
                out_feats_b[self.feat_id_map[idx]] = en_feats_b[key]
                out_feats_c[self.feat_id_map[idx]] = en_feats_c[key]
        return de_feats_b['0_deconv'], de_feats_c['0_deconv'], out_feats_b, out_feats_c

    def encode(self, x, name='a'):
        if name == 'a':
            raw_feats = self.encoder_b(x)
        elif name == 'b':
            raw_feats = self.encoder_c(x)
        else:
            raise NotImplementedError
        out_feats = {}
        for idx, key in enumerate(self.encoder_b.model_dict.keys()):
            if idx in self.feat_id_map:
                out_feats[self.feat_id_map[idx]] = raw_feats[key]
        return out_feats


class PsiUnetGenMS(nn.Module):
    """Extract with procedure features"""

    def __init__(self, gen_opt):
        super(PsiUnetGenMS, self).__init__()
        self.dim = gen_opt.dim
        self.norm = gen_opt.norm
        self.mode = gen_opt.mode
        self.activ = gen_opt.activ
        self.n_layers = gen_opt.n_layers
        self.input_dim = gen_opt.input_dim
        self.output_dim_b = gen_opt.output_dim_b
        self.output_dim_c = gen_opt.output_dim_c
        self.output_dim_e = gen_opt.output_dim_e
        self.use_dropout = gen_opt.use_dropout

        self.feat_id_map = {int(self.n_layers * 1 / 4): 'low',
                            int(self.n_layers * 2 / 4): 'mid',
                            int(self.n_layers * 3 / 4): 'deep',
                            int(self.n_layers * 4 / 4): 'out'
                            }

        self.encoder = UnetEncoder(input_nc=self.input_dim, dim=self.dim, n_layers=self.n_layers,
                                   norm_layer=self.norm, use_dropout=self.use_dropout, activ=self.activ)
        self.decoder_b = UnetDecoder(output_nc=self.output_dim_b, dim=self.dim, n_layers=self.n_layers,
                                     norm_layer=self.norm, use_dropout=self.use_dropout, activ=self.activ)
        self.decoder_c = UnetDecoder(output_nc=self.output_dim_c, dim=self.dim, n_layers=self.n_layers,
                                     norm_layer=self.norm, use_dropout=self.use_dropout, activ=self.activ)
        self.decoder_e = UnetDecoder(output_nc=self.output_dim_e, dim=self.dim, n_layers=self.n_layers,
                                     norm_layer=self.norm, use_dropout=self.use_dropout, activ=self.activ)

    def forward(self, x):
        en_feats = self.encoder(x)
        de_feats_b = self.decoder_b(en_feats)
        de_feats_c = self.decoder_c(en_feats)
        de_feats_e = self.decoder_e(en_feats)
        out_feats_b = {}
        out_feats_c = {}
        out_feats_e = {}
        for idx, key in enumerate(self.encoder.model_dict.keys()):
            if idx in self.feat_id_map:
                out_feats_b[self.feat_id_map[idx]] = en_feats[key]
                out_feats_c[self.feat_id_map[idx]] = en_feats[key]
                out_feats_e[self.feat_id_map[idx]] = en_feats[key]
        return de_feats_b['0_deconv'], de_feats_c['0_deconv'], de_feats_e['0_deconv'], \
               out_feats_b, out_feats_c, out_feats_e


##################################################################################
# Encoder and Decoders
##################################################################################

class Vgg11EncoderMS(nn.Module):
    """Vgg encoder wiht multi-scales"""

    def __init__(self, input_dim, pretrained):
        super(Vgg11EncoderMS, self).__init__()
        features = list(vgg11(pretrained=pretrained, in_channels=input_dim).features)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1',
                       'conv2_1',
                       'conv3_1', 'conv3_2',
                       'conv4_1', 'conv4_2',
                       'conv5_1', 'conv5_2']
        idx = 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 3, 6, 8, 11, 13, 16, 18}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'low': result_dict['conv1_1'],
            'mid': result_dict['conv2_'],
            'deep': result_dict['conv3_2'],
            'out': result_dict['conv5_2']
        }
        return out_feature


class Vgg19EncoderMS(nn.Module):
    def __init__(self, input_dim, pretrained):
        super(Vgg19EncoderMS, self).__init__()
        features = list(vgg19(pretrained=pretrained, in_channels=input_dim).features)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        idx = 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'low': result_dict['conv1_2'],
            'mid': result_dict['conv2_2'],
            'deep': result_dict['conv3_2'],
            'out': result_dict['conv5_4']
        }
        return out_feature


class DecoderMS(nn.Module):
    def __init__(self, input_dim, dim, output_dim, n_layers, pad_type, activ, norm):
        """output_shape = [H, W, C]"""
        super(DecoderMS, self).__init__()

        self.fuse_out = Conv2dBlock(512, 256, kernel_size=3, stride=1,
                                    pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_deep = Conv2dBlock(512, 128, kernel_size=3, stride=1,
                                     pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_mid = Conv2dBlock(256, 64, kernel_size=3, stride=1,
                                    pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_low = Conv2dBlock(128, 32, kernel_size=3, stride=1,
                                    pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_input = Conv2dBlock(32 + input_dim, dim, kernel_size=3, stride=1, padding=1,
                                      pad_type=pad_type, activation=activ, norm=norm)
        self.contextual_blocks = []
        for i in range(n_layers):
            self.contextual_blocks += [Conv2dBlock(dim, dim, kernel_size=3, dilation=2 ** i, padding=2 ** i,
                                                   pad_type=pad_type, activation=activ, norm=norm)]

        # use reflection padding in the last conv layer
        self.contextual_blocks += [
            Conv2dBlock(dim, dim, kernel_size=3, padding=1, norm='in', activation=activ, pad_type=pad_type)]
        self.contextual_blocks += [
            Conv2dBlock(dim, output_dim, kernel_size=1, norm='none', activation='none', pad_type=pad_type)]
        self.contextual_blocks = nn.Sequential(*self.contextual_blocks)

    @staticmethod
    def _fuse_feature(x, feature):
        _, _, h, w = feature.shape
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([x, feature], dim=1)
        return x

    def forward(self, input_x, feat_dict):
        x = feat_dict['out']
        x = self.fuse_out(x)
        x = self._fuse_feature(x, feat_dict['deep'])
        x = self.fuse_deep(x)
        x = self._fuse_feature(x, feat_dict['mid'])
        x = self.fuse_mid(x)
        x = self._fuse_feature(x, feat_dict['low'])
        x = self.fuse_low(x)
        x = self._fuse_feature(x, input_x)
        x = self.fuse_input(x)

        x = self.contextual_blocks(x)
        return x


# the left part of unet
class UnetEncoder(nn.Module):
    def __init__(self, input_nc=3, dim=16, max_dim=512, n_layers=6, norm_layer='bn', use_dropout=False, activ='lrelu'):
        super(UnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.dim = dim
        self.max_dim = max_dim
        self.n_layers = n_layers
        self.norm_layer = _get_norm_layer(norm_layer)
        self.use_dropout = use_dropout
        self.activ = _get_active_function(activ)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        conv_in = nn.Conv2d(input_nc, dim, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.model_dict = nn.ModuleDict()
        self.model_dict['0_conv'] = conv_in
        _dim = dim
        for i in range(self.n_layers):
            in_dim = _dim if _dim < max_dim else max_dim
            out_dim = _dim * 2 if _dim * 2 < max_dim else max_dim
            _conv = nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model_list = []
            if self.activ is not None:
                model_list.append(self.activ)
            model_list.append(_conv)
            if self.norm_layer is not None:
                model_list.append(self.norm_layer(out_dim))
            if self.use_dropout:
                model_list.append(nn.Dropout())
            conv_block = nn.Sequential(*model_list)
            self.model_dict['%d_conv' % (i + 1)] = conv_block
            _dim *= 2

    def forward(self, x):
        feature_dict = {}

        is_cuda = isinstance(x, torch.cuda.FloatTensor)
        if is_cuda:
            for key in self.model_dict.keys():
                self.model_dict[key].cuda()

        for key in sorted(self.model_dict.keys()):
            x = self.model_dict[key](x)
            feature_dict[key] = x

        return feature_dict


# the right part of unet
class UnetDecoder(nn.Module):
    def __init__(self, output_nc=3, dim=16, max_dim=512, n_layers=6, norm_layer='bn', use_dropout=False, activ='relu'):
        super(UnetDecoder, self).__init__()
        self.output_nc = output_nc
        self.dim = dim
        self.max_dim = max_dim
        self.n_layers = n_layers
        self.norm_layer = _get_norm_layer(norm_layer)
        self.use_dropout = use_dropout
        self.activ = _get_active_function(activ)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # here, dim*2 means we need merge skip connections here
        conv_out = nn.ConvTranspose2d(dim * 2, self.output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.model_dict = nn.ModuleDict()
        self.model_dict['0_deconv'] = conv_out
        _dim = dim
        for i in range(self.n_layers):
            out_dim = _dim if _dim < max_dim else max_dim
            in_dim = _dim * 2 if _dim * 2 < max_dim else max_dim
            in_dim = in_dim * 2 if i < self.n_layers - 1 else in_dim  # for merging the skip connections
            _conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model_list = []
            if self.activ is not None:
                model_list.append(self.activ)
            model_list.append(_conv)
            if self.norm_layer is not None:
                model_list.append(self.norm_layer(out_dim))
            if self.use_dropout:
                model_list.append(nn.Dropout())
            conv_block = nn.Sequential(*model_list)
            self.model_dict['%d_deconv' % (i + 1)] = conv_block
            _dim *= 2

    def forward(self, in_features):
        x = in_features['%d_conv' % self.n_layers]

        is_cuda = isinstance(x, torch.cuda.FloatTensor)
        if is_cuda:
            for key in self.model_dict.keys():
                self.model_dict[key].cuda()

        x = self.model_dict['%d_deconv' % self.n_layers](x)
        feature_dict = {'%d_deconv' % self.n_layers: x}
        # x = torch.cat([x, feat], 1)

        for idx in reversed(range(len(self.model_dict) - 1)):
            feat = in_features['%d_conv' % idx]
            x = torch.cat([x, feat], 1)
            x = self.model_dict['%d_deconv' % idx](x)
            feature_dict['%d_deconv' % idx] = x

        return feature_dict


##################################################################################
# Modified VGG
##################################################################################

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(in_channels=3, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        in_channels (int):
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg19(in_channels=3, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        in_channels (int):
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


##################################################################################
# Basic Blocks
##################################################################################

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                              bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        # if self.norm:
        #     x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



##################################################################################
# Normalization layers
##################################################################################

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def _get_norm_layer(norm_type='in'):
    if norm_type == 'bn':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'in':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def _get_active_function(act_type='relu'):
    if act_type == 'relu':
        act_func = nn.ReLU(True)
    elif act_type == 'lrelu':
        act_func = nn.LeakyReLU(0.2, True)
    elif act_type == 'prelu':
        act_func = nn.PReLU()
    elif act_type == 'selu':
        act_func = nn.SELU(inplace=True)
    elif act_type == 'sigmoid':
        act_func = nn.Sigmoid()
    elif act_type == 'tanh':
        act_func = nn.Tanh()
    elif act_type == 'none':
        act_func = None
    else:
        raise NotImplementedError('activation function [%s] is not found' % act_type)
    return act_func


##################################################################################
# Distribution distance measurements and losses blocks
##################################################################################

class KLDivergence(nn.Module):
    def __init__(self, size_average=None, reduce=True, reduction='mean'):
        super(KLDivergence, self).__init__()
        self.eps = 1e-12
        self.log_softmax = nn.LogSoftmax()
        self.kld = nn.KLDivLoss(size_average=size_average, reduce=reduce, reduction=reduction)
        pass

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        return self.kld(x, y)


class JSDivergence(KLDivergence):
    def __init__(self, size_average=True, reduce=True, reduction='mean'):
        super(JSDivergence, self).__init__(size_average, reduce, reduction)

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        m = 0.5 * (x + y)

        return 0.5 * (self.kld(x, m) + self.kld(y, m))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        """
        SSIM Loss, return 1 - SSIM
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    @staticmethod
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img_a, img_b):
        (_, channel, _, _) = img_a.size()

        if channel == self.channel and self.window.data.type() == img_a.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            if img_a.is_cuda:
                window = window.cuda(img_a.get_device())
            window = window.type_as(img_a)

            self.window = window
            self.channel = channel

        ssim_v = self._ssim(img_a, img_b, window, self.window_size, channel, self.size_average)

        return 1 - ssim_v


class DistanceLoss(nn.Module):
    def __init__(self, alpha=1.4, scale=10.):
        """
        DistanceLoss
        :param alpha: see eq. 6
        :param scale: scale the L1 distance between two image features
        """
        super(DistanceLoss, self).__init__()
        self.eps = 1e-12
        self.alpha = alpha
        self.scale = scale
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.sigmoid = nn.Sigmoid()
        self.l1_diff = nn.L1Loss()
        pass

    def _compute_dist_loss(self, dist):
        # normalize
        d_l1 = dist * self.scale
        g_ab = -(d_l1 - self.alpha * np.exp(self.alpha)) / (self.alpha ** 2)
        d_psi = 1 / (1 + torch.exp(g_ab))

        return 1 - d_psi

    def forward(self, features_x, features_y):
        loss = 0
        n_sum = 0
        if isinstance(features_x, dict):
            for key in features_x.keys():
                loss += self._compute_dist_loss(self.l1_diff(features_x[key], features_y[key]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)
        elif isinstance(features_x, list):
            for idx in range(len(features_x)):
                loss += self._compute_dist_loss(self.l1_diff(features_x[idx], features_y[idx]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)

        else:
            loss = self._compute_dist_loss(self.l1_diff(features_x, features_y))

        return loss


class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.l1_diff = nn.L1Loss()
        self.eps = 1e-12
        pass

    def forward(self, features_x, features_y):
        loss = 0
        n_sum = 0
        if isinstance(features_x, dict):
            for key in features_x.keys():
                loss += self.l1_diff(features_x[key], features_y[key])
                n_sum += 1

            loss = loss / (n_sum + self.eps)
        elif isinstance(features_x, list):
            for idx in range(len(features_x)):
                loss += self.l1_diff(features_x[idx], features_y[idx])
                n_sum += 1

            loss = loss / (n_sum + self.eps)
        else:
            loss = self.l1_diff(features_x, features_y)

        return loss


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)




