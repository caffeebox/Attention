import nibabel as nib
import visdom
import torch
import os
import numpy as np
from module.utils import vzimages
from module.options import opts_init
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json
from matplotlib import pyplot as plt


opts = opts_init()
class BraTS(Dataset):
    ''' len=13357(slice, hgg) no norm, aug, crop to 160
    original flair and t1: (240, 240, C), float64[0, 7779]
    original seg: (240, 240, C), float64[0, 1, 2, 4]
    '''
    def __init__(self, root=None, length=0, model='slice', val=False):
        self.root = root
        self.model = model
        self.val = val
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(160),
            transforms.ToTensor()
        ])
        path_json = self.root + self.model + '.json'
        with open(path_json, 'r') as f:
            self.indexs = json.loads(f.read())
        self.length = length if length else len(self.indexs['seg'])

    def __len__(self):
        if self.val:
            return self.length // 5
        else:
            return self.length - self.length//5

    def __getitem__(self, item):
        modalities = dict(flair='flair', t1='t1', seg='seg')
        if self.val:
            item = item + self.length - self.length//5
        for k, v in modalities.items():
            if self.model == 'total':
                path, id = self.indexs[k][item]
                path = self.root + path
                im = nib.load(path)
                img = np.expand_dims(im.get_fdata()[:, :, id], axis=2).astype(np.float32)
            else:
                if k=='t1':
                    path = self.root + self.indexs[k][item]
                else:
                    path = self.root + self.indexs[k][item]
                img = np.load(path)
                img = np.expand_dims(img, axis=2).astype(np.float32)
            img = self.transform(img).permute(0, 2, 1)
            if k == 'seg':
                modalities[k] = img.clamp(0, 1)
            else:
                modalities[k] = (img - torch.min(img)) / (torch.max(img) - torch.min(img) + 1e-5)
        return modalities['flair'], modalities['t1'], modalities['seg']

def indexs_generate(root='/ckh/Projects/Attention/data/brats/HGG/'):
    '''Create json index to load data, return {'flair':[(path, inner_id), ...], 't1', 'seg'}'''
    path_save = root + 'train.json'
    files = os.listdir(root)
    files.sort()
    indexs = {'flair':[], 't1':[], 'seg':[]}
    for k, v in indexs.items():
        i = 0
        for n in files:
            filename = n + '_' + k + '.nii.gz'
            path = n + '/' + filename
            im = nib.load(root+path)
            imgs = im.get_fdata()
            for j in range(imgs.shape[2]):
                v.append((path, j))
            i = i+imgs.shape[2]
    with open(path_save, 'a+') as f:
        f.write(json.dumps(indexs))


def slice(root=None):
    root = opts.path_train
    files = os.listdir(root)
    mod = ['flair', 't1', 'seg']
    files.sort()
    for dir in files[10:]:
        print(dir)
        rt = root + dir + '/'
        path_seg = rt + dir + '_' + 'seg' + '.nii.gz'
        path_flair = rt + dir + '_' + 'flair' + '.nii.gz'
        path_t1 = rt + dir + '_' + 't1' + '.nii.gz'
        for m in mod:
            if not os.path.exists(rt+m):
                os.makedirs(rt+m)

        im_seg = nib.load(path_seg)
        im_flair = nib.load(path_flair)
        im_t1 = nib.load(path_t1)
        imgs_seg = im_seg.get_fdata()
        imgs_flair = im_flair.get_fdata()
        imgs_t1 = im_t1.get_fdata()
        j = 0
        for i in range(imgs_seg.shape[2]):
            if np.max(imgs_seg[:, :, i]):
                j = j + 1
                path_array = rt + 'seg' + '/' + str(j) + '.npy'
                np.save(path_array, imgs_seg[:, :, i])

                path_array = rt + 'flair' + '/' + str(j) + '.npy'
                np.save(path_array, imgs_flair[:, :, i])

                path_array = rt + 't1' + '/' + str(j) + '.npy'
                np.save(path_array, imgs_t1[:, :, i])

def slice2():
    root = opts.path_train
    files = os.listdir(root)
    mod = ['flair2', 't12', 'seg2']
    files.sort()
    for dir in files:
        print(dir)
        rt = root + dir + '/'
        path_seg = rt + dir + '_' + 'seg' + '.nii.gz'
        path_flair = rt + dir + '_' + 'flair' + '.nii.gz'
        path_t1 = rt + dir + '_' + 't1' + '.nii.gz'
        for m in mod:
            if not os.path.exists(rt+m):
                os.makedirs(rt+m)

        im_seg = nib.load(path_seg)
        im_flair = nib.load(path_flair)
        im_t1 = nib.load(path_t1)
        imgs_seg = im_seg.get_fdata()
        imgs_flair = im_flair.get_fdata()
        imgs_t1 = im_t1.get_fdata()
        j = 0
        for i in range(imgs_seg.shape[2]):
            if len(np.nonzero(imgs_seg[:, :, i])[0]) > 200:
                j = j + 1
                path_array = rt + 'seg2' + '/' + str(j) + '.npy'
                np.save(path_array, imgs_seg[:, :, i])

                path_array = rt + 'flair2' + '/' + str(j) + '.npy'
                np.save(path_array, imgs_flair[:, :, i])

                path_array = rt + 't12' + '/' + str(j) + '.npy'
                np.save(path_array, imgs_t1[:, :, i])


def slice_indexs(root='/home/y182202001/Projects/Attention/data/brats/HGG/'):
    '''Create json index to load data, return {'flair':[path, ...], 't1', 'seg'}'''
    path_save = root + 'train2.json'
    indexs = dict(flair=[], t1=[], seg=[])
    for k in indexs.keys():
        i = 0
        P = os.listdir(root)
        P.sort()
        for dir in P:
            dir2 = dir + '/' + k + '2/'
            tifs = os.listdir(root+dir2)
            tifs.sort(key=lambda x: int(x[:-4]))
            for tif in tifs:
                indexs[k].append(dir2 + tif)
                i = i+1

    with open(path_save, 'a+') as f:
        f.write(json.dumps(indexs))

if __name__ == '__main__':
    # slice2()
    slice_indexs()
    # path = '/ckh/Projects/Attention/data/brats/HGG/train.json'
    # with open(path, 'r') as f:
    #     indexs = json.loads(f.read())
    #     for k, v in indexs.items():
    #         print(v[:77])
    #         break
    # indexs_generate()
    # viz = visdom.Visdom(server='223.2.17.239', port=8087, env='attention-BraTS')
    # dataset = BraTS('/home/y182202001/Projects/Attention/data/brats/HGG/', length=0, model='slice', val=True)
    # print(len(dataset))
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
    # for i, (flair, t1, seg) in enumerate(dataloader):
    #     if i==3:
    #         vzimages(viz, flair, nrow=8)
    #         vzimages(viz, t1, nrow=8)
    #         break

