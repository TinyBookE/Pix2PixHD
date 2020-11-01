from torch.utils.data.dataloader import DataLoader
import utils
import os
import numpy as np
from PIL import Image
import torch

from model import Pix2PixHD
from customDataset import CustomDataset

def test_global():

    customdata = CustomDataset('./datasets/test_label', isGlobal=True)
    dataset = DataLoader(customdata, batch_size=1, shuffle=True)

    with torch.no_grad():
        model = Pix2PixHD(35)

        start_epoch, epoch_iter = utils.load_model(model, './checkpoints/global')

        model.cuda()
        print(list(model.children()))

        for i, data in enumerate(dataset):
            if i > 100:
                break
            generated = model.infer(data['label'])

            generated = generated.cpu().numpy()[0]
            generated = ((np.transpose(generated, (1, 2, 0)) + 1) * 0.5 * 255).astype(np.uint8)

            label = data['label'].numpy()[0].astype(np.uint8)
            label = utils.Colorize(35)(label)
            label = np.transpose(label, (1, 2, 0)).astype(np.uint8)

            label = Image.fromarray(label)
            label_path = os.path.join('results/global', 'input_%d.png' % (i))
            label.save(label_path)

            generated = Image.fromarray(generated)
            generated_path = os.path.join('results/global', 'generated_%d.png' % (i))
            generated.save(generated_path)


def test_local():

    customdata = CustomDataset('./datasets/test_label', isGlobal=True)
    dataset = DataLoader(customdata, batch_size=1, shuffle=True)

    with torch.no_grad():
        model = Pix2PixHD(35, net_G='local')

        start_epoch, epoch_iter = utils.load_model(model, './checkpoints/local')

        model.cuda()
        print(list(model.children()))

        for i, data in enumerate(dataset):
            if i > 100:
                break
            generated = model.infer(data['label'])

            generated = generated.cpu().numpy()[0]
            generated = ((np.transpose(generated, (1, 2, 0)) + 1) * 0.5 * 255).astype(np.uint8)

            label = data['label'].numpy()[0].astype(np.uint8)
            label = utils.Colorize(35)(label)
            label = np.transpose(label, (1, 2, 0)).astype(np.uint8)

            label = Image.fromarray(label)
            label_path = os.path.join('results/local', 'input_%d.png' % (i))
            label.save(label_path)

            generated = Image.fromarray(generated)
            generated_path = os.path.join('results/local', 'generated_%d.png' % (i))
            generated.save(generated_path)


if __name__ == '__main__':
    test_local()