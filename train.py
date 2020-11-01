import datetime
import numpy as np
from torch.utils.data.dataloader import DataLoader
import utils
import torch

from model import Pix2PixHD
from customDataset import CustomDataset

display_freq = 100
print_freq = 100
save_latest_freq = 1000
save_epoch_freq = 10


def train_global(load=False):
    all_epoch = 150
    batch_size = 1
    init_lr = 0.0002
    lr_decay_epoch = 50

    customdata = CustomDataset('./datasets/train_label', './datasets/train_img', isGlobal=True)
    dataset = DataLoader(customdata, batch_size=batch_size)

    dataset_size = len(dataset)

    model = Pix2PixHD(35)

    if load:
        start_epoch, epoch_iter = utils.load_model(model, './checkpoints/global')
        print('Model continue to train.\nepoch {} / {}\titer {}'.format(start_epoch, all_epoch, epoch_iter))
    else:
        start_epoch = 1
        epoch_iter = 0

    model.cuda()
    print(list(model.children()))

    if start_epoch > lr_decay_epoch:
        lr = init_lr - (start_epoch - lr_decay_epoch + 1) * init_lr / 100.0
    else:
        lr = init_lr

    # steps that have been trained
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

    display_delta = total_steps % display_freq
    print_delta = total_steps % print_freq
    save_delta = total_steps % save_latest_freq

    for epoch in range(start_epoch, all_epoch + 1):

        # learn rate decay
        if epoch > lr_decay_epoch:
            lr = lr - init_lr / 100.0

        loss_dict = {'GANLoss': [], 'FeatureLoss': [], 'VGGLoss': [], 'RealLoss': [],
                     'FakeLoss': []}

        for i, data in enumerate(dataset, start=epoch_iter):
            total_steps += batch_size
            epoch_iter += batch_size

            # whether save the generated
            infer = total_steps % display_freq == display_delta

            losses, generated = model(data['label'], data['real'], infer)
            optimizer_g, optimizer_d = model.get_optimizer(lr)

            # loss
            loss_d = 0.5 * losses[3] + 0.5 * losses[4]
            loss_g = losses[0] + losses[1] + losses[2]

            loss_dict['GANLoss'].append(losses[0].detach().cpu().numpy())
            loss_dict['FeatureLoss'].append(losses[1].detach().cpu().numpy())
            loss_dict['VGGLoss'].append(losses[2].detach().cpu().numpy())
            loss_dict['RealLoss'].append(losses[3].detach().cpu().numpy())
            loss_dict['FakeLoss'].append(losses[4].detach().cpu().numpy())

            # optimize
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()


            # display info
            if infer:
                data['generated'] = generated.cpu().detach()
                utils.save_image_from_tensor(data, epoch, epoch_iter, './checkpoints/global/imgs')

            if total_steps % print_freq == print_delta:
                t = datetime.datetime.now()
                msg = 'Now, epoch {} / {}\titer {}\tTime: {}\nGANLoss: {}\tFeatureLoss: {}\tVGGLoss: {}\t' \
                      'RealLoss: {}\tFakeLoss: {}\n----------------\n'.format(
                    epoch, all_epoch, epoch_iter, t, loss_dict['GANLoss'][-1],
                    loss_dict['FeatureLoss'][-1], loss_dict['VGGLoss'][-1],
                    loss_dict['RealLoss'][-1], loss_dict['FakeLoss'][-1])
                print(msg)

            if total_steps % save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                utils.save_model(model, epoch, epoch_iter, save_dir='./checkpoints/global')

        # end of one epoch
        epoch_iter = 0

        t = datetime.datetime.now()
        msg = 'End of epoch {} / {} \t Time: {}\nGANLoss: {}\tFeatureLoss: {}\tVGGLoss: {}\tRealLoss: {}\tFakeLoss: ' \
              '{}\n----------------\n'.format(epoch, all_epoch, t, np.average(loss_dict['GANLoss']),
                                              np.average(loss_dict['FeatureLoss']), np.average(loss_dict['VGGLoss']),
                                              np.average(loss_dict['RealLoss']), np.average(loss_dict['FakeLoss']))
        utils.log('./checkpoints/global/train_log.txt', msg)

        # save model
        if epoch % save_epoch_freq == 0:
            utils.save_model(model, epoch, epoch_iter, './checkpoints/global', save_epoch=True)

    # end of training
    utils.save_model(model, all_epoch, 0, './checkpoints/global')

def train_local(load=False):
    all_epoch = 150
    batch_size = 1
    init_lr = 0.0002
    lr_decay_epoch = 50
    fix_epoch = 70

    customdata = CustomDataset('./datasets/train_label', './datasets/train_img', isGlobal=True)
    dataset = DataLoader(customdata, batch_size=batch_size, shuffle=True)

    dataset_size = len(dataset)

    model = Pix2PixHD(35, net_G='local')

    if load:
        start_epoch, epoch_iter = utils.load_model(model, './checkpoints/local')
        print('Model continue to train.\nepoch {} / {}\titer {}'.format(start_epoch, all_epoch, epoch_iter))
    else:
        utils.load_model(model, './checkpoints/global')
        start_epoch, epoch_iter = 1, 0


    model.cuda()
    print(list(model.children()))

    if start_epoch > lr_decay_epoch:
        lr = init_lr - (start_epoch - lr_decay_epoch + 1) * init_lr / 100.0
    else:
        lr = init_lr

    # steps that have been trained
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

    display_delta = total_steps % display_freq
    print_delta = total_steps % print_freq
    save_delta = total_steps % save_latest_freq

    for epoch in range(start_epoch, all_epoch + 1):

        # learn rate decay
        if epoch > lr_decay_epoch:
            lr = lr - init_lr / 100.0

        loss_dict = {'GANLoss': [], 'FeatureLoss': [], 'VGGLoss': [], 'RealLoss': [],
                     'FakeLoss': []}

        for i, data in enumerate(dataset, start=epoch_iter):
            total_steps += batch_size
            epoch_iter += batch_size

            # whether save the generated
            infer = total_steps % display_freq == display_delta

            losses, generated = model(data['label'], data['real'], infer)
            optimizer_g, optimizer_d = model.get_optimizer(lr, epoch < fix_epoch)

            # loss
            loss_d = 0.5 * losses[3] + 0.5 * losses[4]
            loss_g = losses[0] + losses[1] + losses[2]

            loss_dict['GANLoss'].append(losses[0].detach().cpu().numpy())
            loss_dict['FeatureLoss'].append(losses[1].detach().cpu().numpy())
            loss_dict['VGGLoss'].append(losses[2].detach().cpu().numpy())
            loss_dict['RealLoss'].append(losses[3].detach().cpu().numpy())
            loss_dict['FakeLoss'].append(losses[4].detach().cpu().numpy())

            # optimize
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()


            # display info
            if infer:
                data['generated'] = generated.cpu().detach()
                utils.save_image_from_tensor(data, epoch, epoch_iter, './checkpoints/local/imgs')

            if total_steps % print_freq == print_delta:
                t = datetime.datetime.now()
                msg = 'Now, epoch {} / {}\titer {}\tTime: {}\nGANLoss: {}\tFeatureLoss: {}\tVGGLoss: {}\t' \
                      'RealLoss: {}\tFakeLoss: {}\n----------------\n'.format(
                    epoch, all_epoch, epoch_iter, t, loss_dict['GANLoss'][-1],
                    loss_dict['FeatureLoss'][-1], loss_dict['VGGLoss'][-1],
                    loss_dict['RealLoss'][-1], loss_dict['FakeLoss'][-1])
                print(msg)

            if total_steps % save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                utils.save_model(model, epoch, epoch_iter, save_dir='./checkpoints/local')

        # end of one epoch
        epoch_iter = 0

        t = datetime.datetime.now()
        msg = 'End of epoch {} / {} \t Time: {}\nGANLoss: {}\tFeatureLoss: {}\tVGGLoss: {}\tRealLoss: {}\tFakeLoss: ' \
              '{}\n----------------\n'.format(epoch, all_epoch, t, np.average(loss_dict['GANLoss']),
                                              np.average(loss_dict['FeatureLoss']), np.average(loss_dict['VGGLoss']),
                                              np.average(loss_dict['RealLoss']), np.average(loss_dict['FakeLoss']))
        utils.log('./checkpoints/local/train_log.txt', msg)

        # save model
        if epoch % save_epoch_freq == 0:
            utils.save_model(model, epoch, epoch_iter, './checkpoints/local', save_epoch=True)

    # end of training
    utils.save_model(model, all_epoch, 0, './checkpoints/local')

if __name__ == '__main__':
    train_global()
