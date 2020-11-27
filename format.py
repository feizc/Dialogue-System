import os 
import numpy as np
import scipy as sp 
from PIL import Image, ImageSequence
from skimage import io, transform
import matplotlib.pyplot as plt  





'''
# 读取jpg
img = io.imread('./image/大意.jpg')
img = transform.resize(img,(256,256,3))
print(img.shape)

np.save('./npy_image/大意.npy', img)

# 读取gif
gif = Image.open("./image/耗子为止.gif")
iter = ImageSequence.Iterator(gif)
for g in iter:
    img = np.array(g)
    img = transform.resize(img, (256, 256, 3))
    print(img.shape)
    #plt.imshow(gif)
    #plt.show()
'''