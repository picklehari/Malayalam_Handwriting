import cv2
import numpy as np
from matplotlib import pyplot as plt
import urllib
from scipy import ndimage

def clipping_image(new):
  colsums = np.sum(new, axis=0)
  linessum = np.sum(new, axis=1)
  colsums2 = np.nonzero(0-colsums)                                                                 
  linessum2 = np.nonzero(0-linessum)    
  
  xx=linessum2[0][0]                                                               
  yy=linessum2[0][linessum2[0].shape[0]-1]    
  ww=colsums2[0][0]
  hh=colsums2[0][colsums2[0].shape[0]-1]
  print(xx,ww,yy,hh) 
  imgcrop = new[xx:yy, ww:hh]
  plt.imshow(imgcrop)
  plt.show()
  return imgcrop

img = cv2.imread('test.jpg',0)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,thresh = cv2.threshold(blur,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(thresh)
plt.show()

num_labels, labels_im = cv2.connectedComponents(thresh)

plt.imshow(labels_im)
plt.show()
print(num_labels)

for i in range(1,num_labels):
  new, nr_objects = ndimage.label(labels_im == i) 
  #dst="/content/drive/My Drive/project_image_output/"+str(i)+".png"
  print(i)
  new=clipping_image(new)
  print(new)