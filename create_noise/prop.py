from PIL import Image
import numpy as np
import os

lines = 0
with open("./position.txt", "r") as f:
    lines = f.readlines()

data = []
for line in lines:
    line = line.replace("\n", "")
    a = line.split(",")
    data.append(a)

for d in data:
    d_name = d.pop(0)
    print(d_name)
    print(d)

    if not(os.path.exists('detect')):
        os.mkdir('detect')    
    
    if not(os.path.exists('position')):
        os.mkdir('position')    
    
    for i,position in enumerate(d):
        im = np.array(Image.open('sample/' + d_name))
        position = position.split("_")
        print(position)
        print(im.shape)
        height = im.shape[0]
        width = im.shape[1]
        p_wid = int(position[0])
        p_hei = int(position[1])
        if p_wid + 128 > width:
            right = width
            left = width -256
        elif p_wid - 128 < 0:
            right = 256
            left = 0
        else:
            right = p_wid + 128
            left = p_wid - 128

        if p_hei + 128 > height:
            up = height
            down = height - 256
        elif p_hei - 128 < 0:
            up = 256
            down = 0
        else:
            up = p_hei + 128
            down = p_hei - 128
        print(down,up,left,right)
        prop_im = im[down:up,left:right,:].astype('u1')
        im[down:up,left:right,:] = 255
        im = np.array(im).astype('u1')
        print(prop_im.shape)
        prop_im = Image.fromarray(prop_im)
        prop_sample = Image.fromarray(im)
        prop_im.save('detect/' + str(i) + '_' + d_name)
        positions = np.array([down,up,left,right])
        name, _ = os.path.splitext(d_name)
        np.save('position/' + str(i) + '_' + name + '.npy', positions)
        
