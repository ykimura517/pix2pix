#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os

import chainer
from chainer import cuda
from chainer import serializers, Variable

from net import Encoder
from net import Decoder

import glob
from PIL import Image
import random


def main():
    parser = argparse.ArgumentParser(description='chainer creating pictures of seat')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--data', '-d', default='./detect',
                        help='Directory of image files.')
    parser.add_argument('--mask', '-ma', default='./mask',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='./result',
                        help='Directory to output the result')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--model', '-m', default='snapshot_iter_83000.npz',
			help='Loading model')
    parser.add_argument('--batchsize', '-b', default=16,
			type=int, help='The same value as that of trainer')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    # Set up a neural network to train
    enc = Encoder(in_ch=4)
    dec = Decoder(out_ch=3)
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()

    xp = cuda.cupy

    # Setup an model
    model = args.model
    print('---Loading model---')
    serializers.load_npz(model, enc, path='updater/model:enc/')
    serializers.load_npz(model, dec, path='updater/model:dec/')

    # Trun off a trainer
    chainer.config.train = False
    chainer.config.debug = False
    chainer.config.enable_backprop = False
    print('Setuped model!')

    # Load datasets
    print('---Loading datasets---')
    data_path = args.data
    data_sets = glob.glob(data_path + '/*.jpg')
    data_mask = glob.glob(args.mask + '/*.jpg')
    dataset = []
    names = []
    for data in data_sets:
        d_name = os.path.basename(data)
        d_name = d_name[:-4] 
        img = Image.open(data)
        img = xp.asarray(img).transpose(2,0,1)
        for _ in range(10):
            mask = random.choice(data_mask)
            mask = Image.open(mask)
            mask = xp.asarray(mask)
            mask = mask[xp.newaxis,:,:]
            img_ = img + mask
            img_ = xp.asarray(img_).astype('f') / 128.0 - 1.0
            mask = xp.asarray(mask).astype('f') / 128.0 -1.0
            img_ = xp.concatenate([img_,mask], axis=0)
            dataset.append(img_)
            f_name = d_name + '_' + str(_)
            names.append(f_name)
    print('Setuped datasets!')
    

    # Create picture
    print('---Creating---')
    in_ch = 4
    in_h = 256
    in_w = 256
    out_put = 0
     
    batch_size = args.batchsize
    out_dir = args.out

    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)

    _ = 0
    for name, data in zip(names,dataset):
        X_in = xp.zeros((batch_size, in_ch, in_h, in_w)).astype("f")
        for i in range(batch_size):
            X_in[i,:] = xp.asarray(data)
        X_in = Variable(X_in)
        
        z = enc(X_in)
        X_out = dec(z)
        out_put = xp.asarray(X_out.data)
        out_put = out_put[0]
        out_put += 1.0
        out_put *= 128.0
        xp.save(out_dir + '/' + name, out_put)
        _ += 1
        print('created {} / {}'.format(_, len(dataset)))

    print('Finished all process!')
    print('Numpy shape : ', out_put.shape)
    print('Number of Numpy file : ', len(dataset))
            

if __name__ == '__main__':
    main()
