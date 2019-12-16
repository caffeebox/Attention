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
from module.utils import pltimg
from module.net import *
from module.loss import *
import visdom
import time


opts = opts_init()
viz = visdom.Visdom(server=opts.server, port=opts.port, env=opts.env)
viz.text('--> visdom connect', win='progress', opts=dict(title='progress'))
os.environ['CUDA_VISIBLE_DEVICES'] = opts.cudaid

viz.text('--> loading dataset', win='progress', append=True)
start_time = time.time()
# trainset = Cifar10(root=opts.path_train, val=False)
trainset = BraTS(root=opts.path_train, length=opts.data_size, model=opts.data_mode, val=False)
trainloader = DataLoader(trainset, batch_size=opts.batch_size, shuffle=False, num_workers=4)
loss_iter = len(trainset) // (20 * opts.batch_size)   # one epoch 20 loss point

# net = SegNet()
# net = UNet()
net = CoUNet()
net.cuda()

criterion = nn.BCELoss()
# criterion = binary_focal_loss(alpha=opts.focal_alpha, gamma=opts.focal_gamma)
optimizer = optim.Adam(net.parameters(), lr=opts.lr, betas=(0.5, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts.scheduler_step, gamma=opts.scheduler_gamma)

viz.text('using: {:.2f}s'.format(time.time()-start_time), win='progress', append=True)
viz.text('--> starting train', win='progress', append=True)
if not os.path.exists('result/' + opts.env):
    os.makedirs('result/' + opts.env)
loss_total = []
for epoch in range(opts.EPOCH):
    start_time = time.time()
    loss_epoch = []
    for i, (CT, MR, GT) in enumerate(trainloader):
        CT, MR, GT = CT.cuda(), MR.cuda(), GT.cuda()
        feature, SFL = net(CT, MR)
        # feature = net(MR, CT)
        loss = criterion(feature, GT)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch.append(loss.item())

        if (i + 1) % loss_iter == 0:
            loss_total.append(np.mean(loss_epoch))
            loss_epoch = []
            x = torch.arange(len(loss_total))
            viz.line(X=x, Y=loss_total, win='loss', opts=dict(title='loss'))
    viz.text('finished epoch: {}, using {:.2f}s'.format(epoch, time.time()-start_time), win='progress', append=True)
    scheduler.step()

    if (epoch+1) % opts.image_step == 0:
        with torch.no_grad():
            for i, (CT, MR, GT) in enumerate(trainloader):
                if i==7:
                    CT, MR, GT = CT.cuda(), MR.cuda(), GT.cuda()
                    feature, SFL = net(CT, MR)
                    imgs = torch.cat((CT, feature, GT), dim=0)
                    title = 'image_' + str(epoch+1)
                    vzimages(viz, imgs, nrow=opts.batch_size, title=title)
                    path = 'result/' + opts.env + '/' + str(epoch + 1) + '.png'
                    CT, GT, SFL = CT.cpu(), GT.cpu(), SFL.cpu()
                    pltimg(CT, GT, SFL, path)
                    break
    if (epoch+1) % opts.param_step == 0:
        path = 'result/' + opts.env + '/' + str(epoch+1) + '.pkl'
        torch.save(net.state_dict(), path)
        viz.text('net state dict saved', win='progress', append=True)

viz.text('--> ending train', win='progress', append=True)
