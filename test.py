import pdb
import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from module.data_brats import BraTS
from module.data_cifar10 import Cifar10
from module.options import opts_init
from module.utils import vzimages
# from module.net import SegNet
from module.net import *
from module.loss import *
import visdom
import time


opts = opts_init()
path_param = '/home/y182202001/Projects/Attention/result/' + opts.env + '/30.pkl'
viz = visdom.Visdom(server=opts.server, port=opts.port, env='test')
if not viz.win_exists('progress'):
    viz.text('--> visdom connect', win='progress', opts=dict(title='progress'))


def test(num_bs, validate=True, inteval=(-2, -1)):
    '''
        1num_bs == 8num_pic == 8*5/1num_val == 8*5/4num_train, 0 means total 13357
        validate different data_size select different data(from the last data_size)
    '''
    data_size = (num_bs*40) if validate else (num_bs*10)
    trainset = BraTS(root=opts.path_train, length=data_size, model=opts.data_mode, val=validate)
    # trainset = Cifar10(root=opts.path_train, val=validate)
    trainloader = DataLoader(trainset, batch_size=opts.batch_size, shuffle=False, num_workers=4)

    # net = SegNet()
    net = UNet()
    # net = CoUNet()
    net.cuda()
    net.load_state_dict(torch.load(path_param))

    dice_total = []
    for i, (CT, MR, GT) in enumerate(trainloader):
        CT, MR, GT = CT.cuda(), MR.cuda(), GT.cuda()
        # feature, _ = net(CT, MR)
        feature = net(CT, MR)
        result = torch.where(feature>=0.5, torch.tensor(1.0, device='cuda'), torch.tensor(0.0, device='cuda'))
        imgs = torch.cat((CT, result, GT), dim=0)
        if (i+1) in range(inteval[0], inteval[1]+1):
            temp_title = opts.env + '_' + str(i+1)
            vzimages(viz, imgs, nrow=opts.batch_size, title=temp_title)
        inter = torch.sum(result * GT)
        union = torch.sum(result) + torch.sum(GT)
        dice = (2 * inter) / (union + 1e-5)
        dice_total.append(dice.item())
    print(np.mean(dice_total))
    text_out = opts.env + ', ' + 'val ' + str(validate) + ', ' + ' mean dice: {:.4f}'.format(np.mean(dice_total))
    if opts.remark:
        text_out = text_out + '  (' + opts.remark + ')'
    viz.text(text_out, win='progress', append=True)
    return np.mean(dice_total)

if __name__ == '__main__':
    # test(0, validate=False, inteval=(8, 8))
    # test(0, validate=True, inteval=(8, 8))

    # total_dice = []
    # for i in range(1, 20):
    #     total_dice.append(test(i, validate=False))
    #     x = torch.arange(len(total_dice)) + 1
    #     viz.line(X=x, Y=total_dice, win='dice', opts=dict(title='dice'))
    test(0, validate=False, inteval=(7, 12))
