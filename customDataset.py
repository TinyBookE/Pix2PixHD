import torch.utils.data as data
from PIL import Image
import os
import torchvision
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_real_img_name(label_img_name):
    tmp = label_img_name.split('_')
    real_img_name = tmp[0]+'_'+tmp[1]+'_'+tmp[2]+'_leftImg8bit.png'
    return real_img_name


class CustomDataset(data.Dataset):
    def __init__(self, label_dir, real_dir = None, isGlobal = True):
        self.label = []
        self.real = []
        self.isGlobal = isGlobal
        if real_dir is not None:
            label_fnames = os.listdir(label_dir)

            for label in label_fnames:
                real = get_real_img_name(label)
                real_path = os.path.join(real_dir, real)
                if os.path.exists(real_path):
                    label_path = os.path.join(label_dir, label)
                    self.label.append(label_path)
                    self.real.append(real_path)

        else:
            label_fnames = os.listdir(label_dir)
            for label in label_fnames:
                label_path = os.path.join(label_dir, label)
                self.label.append(label_path)
        self.norm = torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        if self.isGlobal:
            self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 1024), 0),
            torchvision.transforms.ToTensor()
        ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        result = {}
        label_img = Image.open(self.label[item]).convert('RGB')
        label_img = self.transform(label_img)
        label_img = label_img*255
        result['label'] = label_img

        if len(self.real) > 0:
            real_img = Image.open(self.real[item]).convert('RGB')
            real_img = self.transform(real_img)
            real_img = self.norm(real_img)
            result['real'] = real_img
        return result
