import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform


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

def stats_sample(n, h, w, points, symbol=0):
    if symbol == 0:
        # mean = torch.tensor([w / 2, h / 2])
        # covariance = torch.tensor([[4 * w, 0], [0, 4 * h]], dtype=torch.float32)
        # m = MultivariateNormal(mean, covariance)
        # cordi = [m.sample() for i in range(points)]
        # SF = torch.stack(cordi, dim=0).expand(n, points, 2) # n*p*2
        x = torch.normal(mean=w / 2, std=w / 4, size=(points,)).clamp(0, w - 1)
        y = torch.normal(mean=h / 2, std=h / 4, size=(points,)).clamp(0, h - 1)
        SF = torch.stack((x, y), dim=0).permute(1, 0).expand(n, points, 2)  # n*p*2
    if symbol == 1:
        m = Uniform(torch.tensor([0.0, 0.0]), torch.tensor([w, h], dtype=torch.float32))
        cordi = [m.sample() for i in range(points)]
        SF = torch.stack(cordi, dim=0).expand(n, points, 2) # n*p*2
    SFL = SF.long()
    return SFL
