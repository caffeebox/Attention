import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def vzimages(viz, imgs, nrow=1, win=None, title='test'):
    for i in range(imgs.shape[0]):
        if isinstance(imgs, torch.Tensor):
            imgs[i] = (imgs[i] - torch.min(imgs[i])) / (torch.max(imgs[i]) - torch.min(imgs[i]) + 1e-6)
        else:
            imgs[i] = (imgs[i] - np.min(imgs[i])) / (np.max(imgs[i]) - np.min(imgs[i]) + 1e-6)
    if win:
        viz.images(imgs, nrow=nrow, win=win, opts=dict(title=title))
    else:
        viz.images(imgs, nrow=nrow, opts=dict(title=title))

def pltimg(img, GT, SFL, path):
    nimg, nSFL, nGT = img.numpy()[0, 0], SFL.numpy()[0], GT.numpy()[0, 0]
    h, w = nimg.shape
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.imshow(nimg, cmap=plt.cm.gray)
    plt.scatter(nSFL[:, 0], nSFL[:, 1], s=1, c='r')
    plt.subplot(2, 1, 2)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.imshow(nGT, cmap=plt.cm.gray)
    plt.savefig(path)
