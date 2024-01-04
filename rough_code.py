import torch
import numpy as np
import scipy.io
import pdb
import cv2

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import cv2
import copy
#inter_mat = scipy.io.loadmat(r"C:\Users\Games\PycharmProjects\PRNet\OUTPUT\0/frame7_depth.mat")
#print(inter_mat['depth'][0,0])

for imi in range(70):
  dep = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\New folder/dmapDR" +str(imi) + ".npy")[0][0]
  #dep = dep[0][0]
  #print(dep.shape)
  #print(dep)

  img = cv2.imread(r'F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN/Untitled2.png', 0)
  for i in range(256):
      for j in range(256):
          val = dep[j,i]
          val = 255*val
          val = int(val)
          img[j,i] = val

  cv2.imwrite(r'F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\New folder\VIS/' + str(imi) + ".png",img)

#dep = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\New folder/dmapDR0.npy")[0][0]
#print(dep)
#print(np.min(dep))
#print(np.max(dep))

#for i in range(187):
#    im = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\NEW_TAR_DEPTH/frame" + str(i) + ".png",0)
#    arr = np.zeros(shape=(1,1,256,256),dtype=np.float32)
#
#    OLDMAX_Z = np.max(im)
#    OLDMIN_Z = np.min(im)
#    NEWMAX_Z = 0.06182186
#    NEWMIN_Z = 0.02674815
#    OldRangeZ = (OLDMAX_Z - OLDMIN_Z)
#    NewRangeZ = (NEWMAX_Z - NEWMIN_Z)
#
#    for mi in range(256):
#     for ji in range(256):
#        oldval = im[ji,mi]
#        NewValue = (((oldval - OLDMIN_Z) * NewRangeZ) / OldRangeZ) + NEWMIN_Z
#        arr[0,0,ji,mi] = NewValue
#    np.save(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\NEW_TAR_DEPTH_npy/frame" +str(i) + ".npy" ,arr)


#for i in range(117):
#    nn = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KS/frame" + str(i)  + ".npy")
#    print("HH: " + str(i))
#    print(nn)