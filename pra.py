#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os

import chainer
from chainer import cuda
from chainer.training import extensions
from chainer import serializers, Variable

from net import Encoder
from net import Decoder

import glob
from PIL import Image
import chainer.cuda as cuda



def main():
    parser = argparse.ArgumentParser(description='chainer creating pictures of seat')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    # parser.add_argument('--data2', '-i', default='./data2',
    #                     help='Directory of image files.')
    parser.add_argument('--out', '-o', default='./result',
                        help='Directory to output the result')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--model', '-m', default='snapshot_iter_20000.npz',
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
    # data_path = args.data
    # data_sets = glob.glob(data_path + '/*.jpg')
    # dataset = []
    # for data in data_sets:
    #     img = Image.open(data)
    #     w,h = img.size
    #     w_128 = int(w / 128) + 1
    #     h_128 = int(h / 128) + 1
    #     w_pad = int(w_128 * 128 - w)
    #     h_pad = int(h_128 * 128 - h)
    #     img = xp.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
    #     height = img.shape[1]
    #     width = img.shape[2]
    #     pad_w = xp.zeros(3 * w_pad * height).reshape((3, height, w_pad)).astype("f")
    #     pad_h = xp.zeros(3 * (w_pad + width) * h_pad).reshape((3, h_pad, (w_pad + width))).astype("f")
    #     img = xp.concatenate([pad_w, img], axis = 2)
    #     img = xp.concatenate([pad_h, img], axis = 1)
    #     dataset.append(img)
    data2 = './data2/0_C_shot00_01.jpg'
    data3 = './data3/0_C_shot00_01.jpg'
    img = Image.open(data2)
    assist = Image.open(data3)
    img = xp.asarray(img).transpose(2,0,1)
    assist = xp.asarray(assist)
    assist = assist[xp.newaxis, :, :]
    img = img + assist
    img = xp.asarray(img).astype('f') / 128.0 - 1.0
    img = xp.concatenate([img, assist], axis=0)
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

    # for num, data in enumerate(dataset):
    #     num += 1
    #     X_in = xp.zeros((batch_size, in_ch, in_h, in_w)).astype("f")
    #     for i in range(batch_size):
    #         X_in[i,:] = xp.asarray(data)
    #     X_in = Variable(X_in)
    #
    #     z = enc(X_in)
    #     X_out = dec(z)
    #     out_put = xp.asarray(X_out.data)
    #     out_put = out_put[0]
    #     out_put += 1.0
    #     out_put *= 128.0
    #     xp.save(out_dir + '/' + str(num), out_put)
    #     print('created {} / {}'.format(num, len(dataset)))
    X_in = xp.zeros((batch_size, in_ch, in_h, in_w)).astype('f')
    for i in range(batch_size):
        X_in[i,:] = xp.asarray(img)
    X_in = Variable(X_in)

    z = enc(X_in)
    X_out = dec(z)
    out_put = xp.asarray(X_out.data)
    out_put = out_put[0]
    out_put += 1.0
    out_put *= 128.0
    out_put = out_put.transpose(1,2,0).astype('u1')
    result_view = Image.fromarray(out_put)
    result_view.save('a.jpg')

    print('Finished all process!')
    print('Numpy shape : ', out_put.shape)
    # print('Number of Numpy file : ', len(dataset))


if __name__ == '__main__':
    main()
