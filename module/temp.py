import nibabel as nib
import visdom
import numpy as np
import torch
from module import utils


viz = visdom.Visdom(port=8087)
path = '/home/y182202001/Projects/Attention/data/brats/HGG/Brats18_2013_2_1/Brats18_2013_2_1_seg.nii.gz'
img = nib.load(path).get_fdata()

index = [len(np.nonzero(img[:, :, i])[0])>200 for i in range(img.shape[2])]
# index = np.sum(img, axis=(0, 1)) != 0
npimg = img[:, :, index]

img = torch.from_numpy(npimg).permute(2, 0, 1).unsqueeze(1)
utils.vzimages(viz, img, nrow=8, title='nonzero')


