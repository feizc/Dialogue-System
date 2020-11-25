import os 
import numpy as np
import scipy as sp 
from skimage import io, transform
import matplotlib.pyplot as plt  

img = io.imread('./image/大意.jpg')
img = transform.resize(img,(256,256,3))
print(img.shape)

np.save('./npy_image/大意.npy', img)

'''
plt.imshow(img)
plt.show()
'''