# coding = UTF-8
import nibabel as nib
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

img = nib.load('C:/Users/猪儿虫/Desktop/predictions/radiopaedia_4_85506_1.nii')
img_arr = img.get_fdata()
img_arr = np.squeeze(img_arr)
for i in range(200):
    # io.imshow(img_arr[150])
    # io.show()
    plt.imshow(img_arr[i+100])
    plt.show()
