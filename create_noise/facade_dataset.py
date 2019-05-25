import os

#import numpy
from PIL import Image
#import six

import numpy as np

#from io import BytesIO
#import pickle
#import json
#import numpy as np

#import skimage.io as io

from chainer.dataset import dataset_mixin

import glob
#import itertools

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='./data', dataDir2 = './data2',
                 dataDir3 = './data3', datas_type='train'):
        print("load dataset start")
        print("    from: %s"%dataDir)
        datas_names = glob.glob(dataDir + '/*.jpg')
        datas_sets = np.array(datas_names)
        self.dataDir = dataDir
        self.dataset = []
        np.random.shuffle(datas_sets)
        for i,datas in enumerate(datas_sets):
            base_name = os.path.basename(datas)
            datas_non_detect = dataDir2+ '/' + base_name
            datas_assist = dataDir3 + '/' + base_name
            img = Image.open(datas_non_detect)
            label = Image.open(datas)
            assist = Image.open(datas_assist)
            img = np.asarray(img).transpose(2,0,1)
            label = np.asarray(label).transpose(2,0,1)
            assist = np.asarray(assist)
            assist = assist[np.newaxis,:,:]
            img = img + assist
            img = np.asarray(img).astype("f")/128.0 - 1.0
            label = np.asarray(label).astype("f")/128.0 - 1.0
            assist = np.asarray(assist).astype('f')/128.0 - 1.0
            img = np.concatenate([img, assist], axis=0)
            self.dataset.append((label,img))
            if datas_type == 'test' and i >= 100:
                break

        print("load dataset done")
    
    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i):
        return self.dataset[i][1], self.dataset[i][0]
    
