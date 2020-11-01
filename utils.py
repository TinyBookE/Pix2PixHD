import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def log(file, msg):
    with open(file, 'a+') as f:
        f.write(msg)


def load_model(model, save_dir='./checkpoints', prefix=None):
    if prefix is None:
        prefix = 'latest'

    iter_path = os.path.join(save_dir, 'iter.txt')
    generator_path = os.path.join(save_dir, '%s_net_G.pth' % prefix)
    discriminate_path = os.path.join(save_dir, '%s_net_D.pth' % prefix)

    model.load_network('G', generator_path)
    if os.path.isfile(discriminate_path):
        model.load_network('D', discriminate_path)

    start_epoch = 1
    epoch_iter = 0
    if os.path.isfile(iter_path):
        txt = np.loadtxt(iter_path)
        start_epoch = int(txt[0])
        epoch_iter = int(txt[1])
    return start_epoch, epoch_iter


def save_model(model, epoch, iter, save_dir='./checkpoints', save_epoch = False):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if save_epoch:
        model.save(epoch, save_dir)
    model.save('latest', save_dir)

    iter_path = os.path.join(save_dir, 'iter.txt')
    np.savetxt(iter_path, (epoch, iter), fmt='%d')


def save_image_from_tensor(image, epoch, iter, save_dir='./checkpoints/imgs'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    label = image['label'].numpy()[0].astype(np.uint8)
    label = Colorize(35)(label)
    label = np.transpose(label, (1, 2, 0)).astype(np.uint8)
    real = image['real'].numpy()[0]
    real = ((np.transpose(real, (1, 2, 0)) + 1) * 0.5 * 255).astype(np.uint8)
    generated = image['generated'].numpy()[0]
    generated = ((np.transpose(generated, (1, 2, 0)) + 1) * 0.5 * 255).astype(np.uint8)

    label = Image.fromarray(label)
    label_path = os.path.join(save_dir, 'input_%d.png' % (epoch))
    label.save(label_path)

    real = Image.fromarray(real)
    real_path = os.path.join(save_dir, 'real_%d.png' % (epoch))
    real.save(real_path)

    generated = Image.fromarray(generated)
    generated_path = os.path.join(save_dir, 'generated_%d.png' % (epoch))
    generated.save(generated_path)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[1], size[2]))

        for label in range(0, len(self.cmap)):
            mask = np.argwhere(gray_image[0]==label)
            if len(mask) == 0:
                continue
            color_image[0][mask[:, 0], mask[:, 1]] = self.cmap[label][0]
            color_image[1][mask[:, 0], mask[:, 1]] = self.cmap[label][1]
            color_image[2][mask[:, 0], mask[:, 1]] = self.cmap[label][2]

        return color_image

def plotLossLog(file):
    epoch_list = []
    GANLoss_list = []
    FeatureLoss_list = []
    with open(file) as f:
        epoch_line = f.readline()
        loss_line = f.readline()
        f.readline()
        while epoch_line != '':
            epoch = int(epoch_line.split()[3])
            epoch_list.append(epoch)
            loss = loss_line.split('\t')
            GANLoss_list.append(float(loss[0].split()[1]))
            FeatureLoss_list.append(float(loss[1].split()[1]))

            epoch_line = f.readline()
            loss_line = f.readline()
            f.readline()

    plt.plot(epoch_list, GANLoss_list, label='GANLoss')
    plt.plot(epoch_list, FeatureLoss_list, label='PatchLoss')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    plotLossLog('checkpoints/local/train_log.txt')