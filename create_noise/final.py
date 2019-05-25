import glob
import os
from PIL import Image
import numpy as np

creates = glob.glob('./image/*.jpg')
positions = glob.glob('./position/*.npy')

for i,create in enumerate(creates):
    detect_image = Image.open(create)
    f_name = os.path.basename(create)
    numpy_name = f_name[:-6] + '.npy'
    position = np.load('./position/' + numpy_name)
    down = position[0]
    up = position[1]
    left = position[2]
    right = position[3]
    detect_numpy = np.asarray(detect_image)
    sample = f_name[2:-6]
    try:
        sample = Image.open('./sample/' + sample + '.jpg')
    except:
        sample = sample[1:]
        sample = Image.open('./sample/' + sample + '.jpg')
        
    sample_numpy = np.asarray(sample)
    sample_numpy.flags.writeable = True
    sample_numpy[down:up,left:right,:] = detect_numpy
    final = Image.fromarray(sample_numpy)
    final.save('./final/' + f_name)   
    print(str(i+1) + ' / ' + str(len(creates)))     
    