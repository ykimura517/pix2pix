#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import os


import glob
from PIL import Image
import numpy as np



def main():

    
    # Load datasets
    print('---Loading datasets---')
    data_sets = glob.glob('./result/*.npy')
    print('Loaded datasets')


    # Create image
    print('---Creating image---')

    out_dir = './image/'
    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)

    data_np = 0
    for num,data in enumerate(data_sets):
        numpy = np.load(data)
        data_np = np.array(numpy).astype('u1')
        data_np = data_np.transpose(1,2,0)
        
        image = Image.fromarray(data_np)
        image.save(out_dir + os.path.basename(data)[:-4] + '.jpg')
        print('Created {} / {}'.format(num+1, len(data_sets)))
    print('Created images!')
 
    print('Finished all process!')
    print('Image shape : ', data_np.shape)
    print('Number of images : ', len(data_sets))
            

if __name__ == '__main__':
    main()
