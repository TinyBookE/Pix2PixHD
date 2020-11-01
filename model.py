import os
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import numpy as np


##################################
#            NetWork             #
##################################
class ResnetBlock(nn.Module):
    def __init__(self, dim, planes):
        super(ResnetBlock, self).__init__()
        model = [nn.Conv2d(dim, planes, kernel_size=1),
                 nn.InstanceNorm2d(planes), nn.ReLU(True),
                 nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                 nn.InstanceNorm2d(planes), nn.ReLU(True),
                 nn.Conv2d(planes, dim, kernel_size=1),
                 nn.InstanceNorm2d(dim)]
        self.model = nn.Sequential(*model)
        self.active = nn.ReLU(True)

    def forward(self, x):
        residual = x
        out = self.model(x)

        out += residual
        out = self.active(out)

        return out


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(GlobalGenerator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7),
                 nn.InstanceNorm2d(ngf), nn.ReLU(True)]

        self.net = []
        self.net.append(nn.Sequential(*model))

        self.n_downsample = 4
        n_downsample = 4

        # downsample
        for i in range(n_downsample):
            mult = 2**i
            model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf*mult*2), nn.ReLU(True)]
            self.net.append(nn.Sequential(*model))

        # resnet
        model = []
        mult = 2**n_downsample
        for i in range(9):
            model += [ResnetBlock(ngf * mult, int(ngf * mult / 4))]
        self.net.append(nn.Sequential(*model))

        # upsample
        for i in range(n_downsample):
            mult = 2**(n_downsample-i)
            model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult /2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf*mult/2)), nn.ReLU(True)]
            self.net.append(nn.Sequential(*model))

        model = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7), nn.Tanh()]
        self.net.append(nn.Sequential(*model))

        for i in range(len(self.net)):
            setattr(self, 'model'+str(i), self.net[i])

    def forward(self, x):
        size = x.size()
        random_dim = torch.randn((1, 1, size[2], size[3])).cuda()
        x = torch.cat((x, random_dim), 1)
        result = [x]
        for i in range(self.n_downsample+2):
            model = self.net[i]
            result.append(model(result[-1]))
        for i in range(self.n_downsample):
            model = self.net[i+self.n_downsample+2]
            result.append(model(result[-1]+result[self.n_downsample-i+1]))
        final = self.net[-1]
        return final(result[-1])


    def get_optimizer(self, lr, fix_global=True):
        params = []
        for i in range(2 * self.n_downsample + 3):
            models = self.net[i]
            params += models.parameters()
        optimizer = torch.optim.Adam(params, lr, betas=(0.5,0.999))
        return optimizer


class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32):
        super(LocalEnhancer, self).__init__()

        # global
        globalGenerator = GlobalGenerator(input_nc+1, output_nc, ngf*2).model
        globalGenerator = [globalGenerator[i] for i in range(len(globalGenerator)-3)]


        # downsample
        model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                            nn.InstanceNorm2d(ngf), nn.ReLU(True),
                            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                            nn.InstanceNorm2d(ngf * 2), nn.ReLU(True)]

        model_resnet = []
        # resnet
        for i in range(3):
            model_resnet += [ResnetBlock(ngf*2, int(ngf/2))]

        # upsample
        model_upsample = [nn.ConvTranspose2d(ngf*2, ngf,
                                             kernel_size=3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(ngf), nn.ReLU(True)]

        # final
        model_generate = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7), nn.Tanh()]

        self.model = nn.Sequential(*globalGenerator)
        self.model_downsample = nn.Sequential(*model_downsample)
        self.model_resnet = nn.Sequential(*model_resnet)
        self.model_upsample = nn.Sequential(*model_upsample)
        self.model_generate = nn.Sequential(*model_generate)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        downsample_x = self.downsample(x)

        out_1 = self.model(downsample_x)
        out_2 = self.model_downsample(x) + out_1
        out = self.model_resnet(out_2)
        out = self.model_upsample(out+out_2)
        out = self.model_generate(out)

        return out

    def get_optimizer(self, lr, fix_global = True):
        params = []
        if not fix_global:
            params += self.model.parameters()
        params += self.model_downsample.parameters()
        params += self.model_resnet.parameters()
        params += self.model_upsample.parameters()
        params += self.model_generate.parameters()

        optimizer = torch.optim.Adam(params, lr, betas=(0.5,0.999))
        return optimizer


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf = 64):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 0 # int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]


        nf = ndf

        for n in range(1, 3):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                         nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]]


        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                     nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        sequence += [[nn.AdaptiveAvgPool2d((7,7)), nn.Flatten()]]

        sequence += [[nn.Linear(49, 1), nn.Sigmoid()]]
        for i in range(7):
            setattr(self, 'model' + str(i), nn.Sequential(*sequence[i]))

    def forward(self, x):
        result = [x]
        for i in range(5):
            model = getattr(self, 'model'+str(i))
            result.append(model(result[-1]))
        model = getattr(self, 'model5')
        final = model(result[-1])
        model = getattr(self, 'model6')
        final = model(final)
        result.append(final)

        return result[1:]


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(MultiscaleDiscriminator, self).__init__()

        self.D_1 = Discriminator(input_nc, ndf)
        self.D_2 = Discriminator(input_nc, ndf)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        result = []
        downsample_x = self.downsample(x)

        result.append(self.D_1(downsample_x))
        result.append(self.D_2(x))

        return result

    def get_optimizer(self, lr):
        params = self.parameters()

        optimizer = torch.optim.Adam(params, lr, betas=(0.5, 0.999))
        return optimizer


class Pix2PixHD(nn.Module):
    def __init__(self, input_nc, isTrain = True, net_G = 'global'):
        super(Pix2PixHD, self).__init__()
        self.isTrain = isTrain

        self.input_nc = input_nc
        # generator
        if net_G == 'global':
            self.net_G = GlobalGenerator(input_nc+1, 3)
        elif net_G == 'local':
            self.net_G = LocalEnhancer(input_nc, 3)
        else:
            raise ('error type of generator')

        # discriminator
        if self.isTrain:
            self.net_D = MultiscaleDiscriminator(input_nc+3)

        # loss and optimizer
        if self.isTrain:
            self.GANLoss = GANLoss()
            self.VGGLoss = VGGLoss()
            self.FeatLoss = nn.L1Loss()

    def encode_label(self, label_img, real_img = None):
        size = label_img.size()
        oneHotSize = (size[0], self.input_nc, size[2], size[3])
        oneHot_lable = torch.cuda.FloatTensor(torch.Size(oneHotSize)).zero_()
        oneHot_lable = oneHot_lable.scatter_(1,label_img.data.long().cuda(),1.0)


        if real_img is not None:
            real_img = Variable(real_img.data.cuda())

        return oneHot_lable, real_img

    def discriminate(self, label, img):
        concat = torch.cat((label, img.detach()), dim=1)
        return self.net_D(concat)

    def forward(self, label, real, infer=False):
        label_img, real_img = self.encode_label(label, real)

        fake_img = self.net_G(label_img)

        # net discriminator
        # fake
        pred_fake_d = self.discriminate(label_img, fake_img)
        loss_D_fake = self.GANLoss(pred_fake_d, False)

        # real
        pred_real = self.discriminate(label_img, real_img)
        loss_D_real = self.GANLoss(pred_real, True)

        # net generator
        pred_fake = self.discriminate(label_img, fake_img)
        loss_G_fake = self.GANLoss(pred_fake, True)

        # feature
        loss_G_feat = 0
        for i in range(2):
            for j in range(len(pred_fake[i])-1):
                loss_G_feat += 0.5 * self.FeatLoss(pred_fake[i][j], pred_real[i][j].detach()) * 10

        # vgg
        loss_G_VGG = self.VGGLoss(fake_img, real_img) * 10
        return [loss_G_fake, loss_G_feat, loss_G_VGG, loss_D_real, loss_D_fake], None if not infer else fake_img

    def infer(self, label):
        label_img, _ = self.encode_label(label)
        fake_img = self.net_G(label_img)
        return fake_img

    def save_network(self, network, network_type, epoch, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch, network_type)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

    def load_network(self, network_type, save_path):
        if not os.path.isfile(save_path):
            raise ('network load failed')

        if network_type == 'G':
            network = self.net_G
        elif network_type == 'D':
            network = self.net_D
        else:
            raise ('wrong network type')

        try:
            network.load_state_dict(torch.load(save_path))
        except:
            pretrained_dict = torch.load(save_path)
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                network.load_state_dict(pretrained_dict)
                print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_type)
            except:
                print('Pretrained network %s has fewer layers; The following are not initialized:' % network_type)
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                not_initialized = set()

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])

                print(sorted(not_initialized))
                network.load_state_dict(model_dict)

    def save(self, epoch, save_dir = './checkpoints'):
        self.save_network(self.net_G, 'G', epoch, save_dir)
        self.save_network(self.net_D, 'D', epoch, save_dir)

    def get_optimizer(self, lr, fix_global=True):
        optimizer_G = self.net_G.get_optimizer(lr, fix_global)
        optimizer_D = self.net_D.get_optimizer(lr)

        return optimizer_G, optimizer_D


##################################
#             Loss               #
##################################
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.net = vgg_pretrained_features[:30]

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.net(x)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = self.criterion(x_vgg, y_vgg.detach())
        return loss


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.real_label = 1
        self.fake_label = 0
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.FloatTensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor).cuda()
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.FloatTensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor).cuda()
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, x, target_is_real):
        if isinstance(x, list):
            loss = 0
            pred_list = []
            for discri_res in x:
                pred_list.append(discri_res[-1])
            pred = torch.mean(torch.stack(pred_list),0)
            target_tensor = self.get_target_tensor(pred, target_is_real)
            loss = self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(x[-1], target_is_real)
            return self.loss(x[-1], target_tensor)
