# import nibabel as nib
from matplotlib import pyplot as plt
# import visdom
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform


# viz = visdom.Visdom(port=8087)
# path = '/home/y182202001/Projects/Attention/data/brats/HGG/Brats18_2013_2_1/flair2/25.npy'
# img = np.load(path)
# h, w = img.shape
# mean = torch.tensor([w/2, h/2])
# covariance = torch.tensor([[4*w, 0], [0, 4*h]], dtype=torch.float32)
# m = MultivariateNormal(mean, covariance)
# cordi = [m.sample() for i in range(256)]
# cordi = torch.stack(cordi, dim=0).numpy()
# plt.figure()
# plt.xlim(0, w)
# plt.ylim(0, h)
# plt.imshow(img, cmap=plt.cm.gray)
# plt.scatter(cordi[:, 0], cordi[:, 1], s=1, c='r')
# plt.show()

m = Uniform(torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0]))
h = [m.sample() for i in range(10)]
print(h)
