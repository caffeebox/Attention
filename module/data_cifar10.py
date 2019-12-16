import os
from module.utils import vzimages
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json


class Cifar10():
    ''' len=668
    original CT and MR: (153, 193), uint8[0, 255]
    original GT: (153, 193), uint8[0, 255, ]
    return (1, 128, 128), float32[0, 1]
    '''
    def __init__(self, root, val=False):
        self.modalites = dict(CT='CT', MR='MR', GT='GT')
        self.root = root
        self.val = val
        self.transform = transforms.Compose([transforms.CenterCrop((128, 128)),
                                             transforms.ToTensor()])
        path_json = root + 'train.json'
        with open(path_json, 'r') as f:
            self.indexs = json.loads(f.read())
        self.length = len(self.indexs['GT'])

    def __len__(self):
        if self.val:
            return self.length // 5
        else:
            return self.length - self.length // 5

    def __getitem__(self, item):
        if self.val:
            item = item + self.length - self.length//5
        for k, v in self.modalites.items():
            path = self.root + self.indexs[k][item]
            im = Image.open(path)
            self.modalites[k] = self.transform(im)
        return self.modalites['CT'], self.modalites['MR'], self.modalites['GT']



def indexs_generate(root='/ckh/Projects/Attention/data/prostate/'):
    '''Create json index to load data, return {'CT':[path, ...], 'MR', 'GT'}'''
    path_save = root + 'train2.json'
    indexs = dict(CT=[], MR=[], GT=[])
    for k in indexs.keys():
        i = 0
        P = os.listdir(root)
        P.sort(key=lambda x: int(x[1:]))
        for dir in P:
            dir2 = dir + '/' + k + '/'
            tifs = os.listdir(root+dir2)
            tifs.sort(key=lambda x: int(x[:-4]))
            for tif in tifs:
                indexs[k].append(dir2 + tif)
                i = i+1

    with open(path_save, 'a+') as f:
        f.write(json.dumps(indexs))

if __name__ == '__main__':
    # trainset = Cifar10(root='/ckh/Projects/Attention/data/prostate/')
    # print(len(trainset))
    # indexs_generate()
    path = '/ckh/Projects/Attention/data/prostate/train.json'
    with open(path, 'r') as f:
        indexs = json.loads(f.read())
        print(indexs['CT'])
