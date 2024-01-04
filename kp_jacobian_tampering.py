import torch
import numpy as np
import torch.nn.functional as F
import cv2
from modules.util import Hourglass, AntiAliasInterpolation2d
#def kp2gaussian(kp, spatial_size, kp_variance):
#    """
#    Transform a keypoint into gaussian like representation
#    """
#    mean = kp
#    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
#    number_of_leading_dimensions = len(mean.shape) - 1
#    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
#    coordinate_grid = coordinate_grid.view(*shape)
#    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
#    coordinate_grid = coordinate_grid.repeat(*repeats)
#    # Preprocess kp shape
#    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
#    mean = mean.view(*shape)
#    mean_sub = (coordinate_grid - mean)
#    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
#    return out
#
#def make_coordinate_grid(spatial_size, type):
#    """
#    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
#    """
#    h, w = spatial_size
#    x = torch.arange(w).type(type)
#    y = torch.arange(h).type(type)
#    x = (2 * (x / (w - 1)) - 1)
#    y = (2 * (y / (h - 1)) - 1)
#    yy = y.view(-1, 1).repeat(1, w)
#    xx = x.view(1, -1).repeat(h, 1)
#    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
#    return meshed
#
#def gaussian2kp(heatmap):
#    """
#    Extract the mean and from a heatmap
#    """
#    shape = heatmap.shape
#    heatmap = heatmap.unsqueeze(-1)
#    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
#    value = (heatmap * grid).sum(dim=(2, 3))
#    kp = {'value': value}
#
#    return kp
#
#np_file = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\results/che.npy")
#
#for i in range(58):
#    for j in range(58):
#       print(str(i) + " " + str(j))
#       print(np_file[i,j])
#img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\results/chet.png")
#img[22,22] = [0,0,0]
#cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\results/chet.png",img)

##np_file[0] = (np_file[0]*127.5)+127.5
#np_file = torch.tensor(np_file).cuda()
##print(np_file)
#np_fileS = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\results/HERRs.npy")
##np_fileS[0] = (np_fileS[0]*127.5)+127.5
#np_fileS = torch.tensor(np_fileS).cuda()
##print(np_file)
##print(type(np_file))
##down = AntiAliasInterpolation2d(3, 0.25)
##source_image = torch.tensor(cv2.imread("F:/PRESERVE_SPACE_PROJECTS/CVPR2022-DaGAN/youtube-taichi/frame0.png")).cuda()
##source_image = down(source_image)
#spatial_size = torch.tensor([58,58])
##print(spatial_size)
#res = kp2gaussian(np_file,spatial_size,kp_variance=0.01)
##heatmap = prediction.view(final_shape[0], final_shape[1], -1)
#heatmap = F.softmax(res / 0.1, dim=2)
##heatmap = heatmap.view(*final_shape)
#print(res)
##resS = kp2gaussian(np_fileS,spatial_size,kp_variance=0.01)
##heatmap = res - resS
##zeros = torch.zeros(heatmap.shape[0], 1, 1, 1).type(heatmap.type())
##heatmap = torch.cat([zeros, heatmap], dim=1)
##heatmap = heatmap.unsqueeze(2)
###print(type(res))
##print(heatmap.shape)
##print(heatmap)
#
##print(gaussian2kp(heatmap))


#np_file = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/frame0.npy")[0]
#print(type(np_file))
#arr = np.zeros(shape=(1,15,2),dtype=np.float64)
#img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\results/Untitled2.png")
#for i in range(15):
#    x = np_file[i,0]
#    y = np_file[i,1]
#    x = (x*127.5)+127.5
#    y = (y * 127.5) + 127.5
#    x = x*0.25
#    y = y*0.25
#    x =  (x-127.5)/127.5
#    y = (y - 127.5) / 127.5
#    #img[int(y),int(x)] = [255,0,0]
#    print(x)
#    print(y)
#    arr[0,i,0] = x
#    arr[0,i,1] = y
##cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\results/im2.png",img)
#np.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\results/frame0.npy",arr)


#MAIN CODE
#ker_sizex = 5
#ker_sizey = 8
#red_factorx = 0.1128/5
#red_factory = 0.1128/8
#for i in range(70):
# arr = np.zeros(shape = (1, 15, 1, 58, 58),dtype=np.float32)
# file = open(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plotfor_heatmap_numeric/" + str(i) + ".txt",'r')
# for k in range(15):
#     txt = file.readline()
#     txt = txt.strip()
#     txt = txt.split()
#     x = int(float(txt[0]))
#     y = int(float(txt[1]))
#     arr[0,k,0,x,y] = 0.05
#     for zwei in range((x-ker_sizex),(x+ker_sizex)):
#         for kwei in range(y-ker_sizey,y+ker_sizey):
#             diff_zwei = abs(x-zwei)
#             diff_kwei = abs(y - kwei)
#             if zwei<58 and zwei>=0 and kwei<58 and kwei>=0:
#              arr[0,k,0,zwei,kwei] = 0.05 - abs(diff_zwei*red_factorx) - abs(diff_kwei*red_factory)
#              #add code
#              #pass
# np.save(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\HEATMAPS/" + str(i) + ".npy",arr)


#SHIFT BASED FULL TRANSFORM
#for i in range(45):
#    file = open(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plotfor_heatmap_numeric/" + str(i) + ".txt",'r')
#    arr = np.load(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\ORIGINAL_HEATMAPS/" + str(i) + ".npy")
#    print(arr.shape)
#    for k in range(15):
#        txt = file.readline()
#        txt = txt.strip()
#        txt = txt.split()
#        xx =float(txt[0])
#        yy = float(txt[1])
#        maxx = np.max(arr[0,k,0])
#        #print(maxx)
#        Y,X = np.where(arr[0,k,0]==maxx)
#        #print(Y)
#        #print(X)
#        shiftx = int(xx-X)
#        shifty = int(yy-Y)
#        print(shiftx)
#        print(shifty)
#        shiftx = -2
#        shifty = 6
#        arr[0, k, 0] = np.roll(arr[0, k, 0],shifty,axis=0)
#        if shifty>=0:
#            arr[0, k, 0][:shifty,:] = 0
#        else:
#            arr[0, k, 0][shifty:, :] = 0
#
#        arr[0, k, 0] = np.roll(arr[0, k, 0], shiftx, axis=1)
#        if shiftx>=0:
#            arr[0, k, 0][:,:shiftx] = 0
#        else:
#            arr[0, k, 0][:,shiftx:] = 0
#
#        Y, X = np.where(arr[0, k, 0] == maxx)
#        #print(Y)
#        #print(X)
#        print("GG")
#    np.save(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\HEATMAPS/" + str(i) + ".npy",arr)


#map = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/feature.npy")
#print(map.shape)
#for i in range(36):
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/Untitled.png")
#    arr = map[0][i]
#    for k in range(64):
#        for g in range(64):
#            val = int(arr[k,g]*255)
#            img[k,g] = [val,0,0]
#    cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/1/" + str(i) + ".png",img )


#for zimmer in range(45):
# map = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\coordinate_grid/" + str(zimmer)+".npy")
# img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/0s.png")
# img2 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/Untitleds.png")
# #print(map.shape)
# map = map[0][14]
# for i in range(64):
#     for j in range(64):
#         x = (map[i,j,1]*32) + 32
#         y = (map[i, j, 0]*32) + 32
#         #print("GG")
#         #print(str(j) + " " + str(i))
#         #print(str(int(x)) + " " +str(int(y)))
#         diffx = j-int(x)
#         diffy = i- int(y)
#         #print(diffx)
#         #print(diffy)
#         #if abs(diffx)>4 or abs(diffy)>4:
#         #print("GG")
#         #print(str(j) + " " + str(i))
#         if x<0:
#             x = 0
#         if y<0:
#             y = 0
#         if x>=64:
#             x = 63
#         if y>=64:
#             y = 63
#
#         #print(str(int(x)) + " " +str(int(y)))
#         #print("HELL")
#         img2[i,j] = img[int(x),int(y)]
#         if i==22 and j==32 and (zimmer==0 or zimmer==18):
#             print(str(int(x)) + " " + str(int(y)))
#
# cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\TESSES/" + str(zimmer) + ".png",img2)
#
#
#file0 = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/frame0.npy")[0]
#kp = file0[14]
#kpx = (kp[0] * 32)+32
#kpy = (kp[1] * 32)+32
#print(str(kpx) + " " + str(kpy))
#
#file1 = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/frame18.npy")[0]
#kp = file1[14]
#kpx = (kp[0] * 32)+32
#kpy = (kp[1] * 32)+32
#print(str(kpx) + " " + str(kpy))


for ki in range(16):
 arr = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\deformed_Src/15.npy")
 arrn = np.zeros(shape=(64,64,3),dtype=int)
 print(arr.shape)
 #print(np.min(arr[0][0]))
 arr = arr[0][ki]
 for i in range(64):
     for j in range(64):
         for k in range(3):
             arrn[i,j,k] = int(arr[k,i,j] *255)
 print(arrn)
 cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\def/syn" + str(ki)+".png",arrn)


#import copy
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_0.png")
#Y,X =np.where(np.all(base_img == [255, 255, 255], axis=2))
#Y_base0 = (np.min(Y) + np.max(Y)) / 2
#X_base0 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_1.png")
#Y,X =np.where(np.all(base_img == [0, 255, 0], axis=2))
#Y_base1 = (np.min(Y) + np.max(Y)) / 2
#X_base1 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_2.png")
#Y,X =np.where(np.all(base_img == [0, 0, 255], axis=2))
#Y_base2 = (np.min(Y) + np.max(Y)) / 2
#X_base2 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_3.png")
#Y,X =np.where(np.all(base_img == [255, 0, 0], axis=2))
#Y_base3 = (np.min(Y) + np.max(Y)) / 2
#X_base3 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_4.png")
#Y,X =np.where(np.all(base_img ==[255, 255, 127], axis=2))
#Y_base4 = (np.min(Y) + np.max(Y)) / 2
#X_base4 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_5.png")
#Y,X =np.where(np.all(base_img ==[255, 127, 255], axis=2))
#Y_base5 = (np.min(Y) + np.max(Y)) / 2
#X_base5 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_6.png")
#Y,X =np.where(np.all(base_img ==[127, 255, 255], axis=2))
#Y_base6 = (np.min(Y) + np.max(Y)) / 2
#X_base6 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_7.png")
#Y,X =np.where(np.all(base_img ==[255, 255, 63], axis=2))
#Y_base7 = (np.min(Y) + np.max(Y)) / 2
#X_base7 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_8.png")
#Y,X =np.where(np.all(base_img ==[255, 63, 255], axis=2))
#Y_base8 = (np.min(Y) + np.max(Y)) / 2
#X_base8 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_9.png")
#Y,X =np.where(np.all(base_img ==[63, 255, 255], axis=2))
#Y_base9 = (np.min(Y) + np.max(Y)) / 2
#X_base9 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_10.png")
#Y,X =np.where(np.all(base_img ==[0, 0, 63], axis=2))
#Y_base10 = (np.min(Y) + np.max(Y)) / 2
#X_base10 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_11.png")
#Y,X =np.where(np.all(base_img ==[0, 63, 0], axis=2))
#Y_base11 = (np.min(Y) + np.max(Y)) / 2
#X_base11 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_12.png")
#Y,X =np.where(np.all(base_img ==[63, 0, 0], axis=2))
#Y_base12 = (np.min(Y) + np.max(Y)) / 2
#X_base12 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_13.png")
#Y,X =np.where(np.all(base_img ==[0, 0, 127], axis=2))
#Y_base13 = (np.min(Y) + np.max(Y)) / 2
#X_base13 = (np.min(X) + np.max(X)) / 2
#
#base_img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_14.png")
#Y,X =np.where(np.all(base_img ==[0, 127, 0], axis=2))
#Y_base14 = (np.min(Y) + np.max(Y)) / 2
#X_base14 = (np.min(X) + np.max(X)) / 2
#
#
#ARRO = np.zeros(shape=(64,64,2),dtype=int)
#for ii in range(64):
#    for jj in range(64):
#        ARRO[ii,jj,0] = jj
#        ARRO[ii, jj, 1] = ii
#
#for i in range(70):
#    arr = np.zeros(shape=(15,64,64,2),dtype=int)
#    for zi in range(15):
#        arr[zi] = copy.deepcopy(ARRO)
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" +str(i) + "_0.png")
#    Y,X =np.where(np.all(img == [255, 255, 255], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base0-x)*(-1)*(-1)
#    diffy = (Y_base0 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[0][:,:,0] =arr[0][:,:,0] - diffx
#    arr[0][:, :, 1] = arr[0][:, :, 1] - diffy
#    #arr[0] = np.where(arr[0]<0,0,arr[0])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_1.png")
#    Y, X = np.where(np.all(img == [0, 255, 0], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base1 - x)*(-1)*(-1)
#    diffy = (Y_base1 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[1][:, :, 0] = arr[1][:, :, 0] - diffx
#    arr[1][:, :, 1] = arr[1][:, :, 1] - diffy
#    #arr[1] = np.where(arr[1] < 0, 0, arr[1])
#
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_2.png")
#    Y, X = np.where(np.all(img ==[0, 0, 255], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base2 - x)*(-1)*(-1)
#    diffy = (Y_base2 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[2][:, :, 0] = arr[2][:, :, 0] - diffx
#    arr[2][:, :, 1] = arr[2][:, :, 1] - diffy
#    #arr[2] = np.where(arr[2] < 0, 0, arr[2])
#
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_3.png")
#    Y, X = np.where(np.all(img == [255, 0, 0], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base3 - x)*(-1)*(-1)
#    diffy = (Y_base3 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[3][:, :, 0] = arr[3][:, :, 0] - diffx
#    arr[3][:, :, 1] = arr[3][:, :, 1] - diffy
#    #arr[3] = np.where(arr[3] < 0, 0, arr[3])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_4.png")
#    Y, X = np.where(np.all(img == [255, 255, 127], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base4 - x)*(-1)*(-1)
#    diffy = (Y_base4 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[4][:, :, 0] = arr[4][:, :, 0] - diffx
#    arr[4][:, :, 1] = arr[4][:, :, 1] - diffy
#    #arr[4] = np.where(arr[4] < 0, 0, arr[4])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_5.png")
#    Y, X = np.where(np.all(img == [255, 127, 255], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base5 - x)*(-1)*(-1)
#    diffy = (Y_base5 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[5][:, :, 0] = arr[5][:, :, 0] - diffx
#    arr[5][:, :, 1] = arr[5][:, :, 1] - diffy
#    #arr[5] = np.where(arr[5] < 0, 0, arr[5])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_6.png")
#    Y, X = np.where(np.all(img == [127, 255, 255], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base6 - x)*(-1)*(-1)
#    diffy = (Y_base6 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[6][:, :, 0] = arr[6][:, :, 0] - diffx
#    arr[6][:, :, 1] = arr[6][:, :, 1] - diffy
#    #arr[6] = np.where(arr[6] < 0, 0, arr[6])
#
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_7.png")
#    Y, X = np.where(np.all(img == [255, 255, 63], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base7 - x)*(-1)*(-1)
#    diffy = (Y_base7 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[7][:, :, 0] = arr[7][:, :, 0] - diffx
#    arr[7][:, :, 1] = arr[7][:, :, 1] - diffy
#    #arr[7] = np.where(arr[7] < 0, 0, arr[7])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_8.png")
#    Y, X = np.where(np.all(img ==[255, 63, 255], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base8 - x)*(-1)*(-1)
#    diffy = (Y_base8 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[8][:, :, 0] = arr[8][:, :, 0] - diffx
#    arr[8][:, :, 1] = arr[8][:, :, 1] - diffy
#    #arr[8] = np.where(arr[8] < 0, 0, arr[8])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_9.png")
#    Y, X = np.where(np.all(img == [63, 255, 255], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base9 - x)*(-1)*(-1)
#    diffy = (Y_base9 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[9][:, :, 0] = arr[9][:, :, 0] - diffx
#    arr[9][:, :, 1] = arr[9][:, :, 1] - diffy
#    #arr[9] = np.where(arr[9] < 0, 0, arr[9])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_10.png")
#    Y, X = np.where(np.all(img == [0, 0, 63], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base10 - x)*(-1)*(-1)
#    diffy = (Y_base10 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[10][:, :, 0] = arr[10][:, :, 0] - diffx
#    arr[10][:, :, 1] = arr[10][:, :, 1] - diffy
#    #arr[10] = np.where(arr[10] < 0, 0, arr[10])
#
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_11.png")
#    Y, X = np.where(np.all(img ==[0, 63, 0], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base11 - x)*(-1)*(-1)
#    diffy = (Y_base11 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[11][:, :, 0] = arr[11][:, :, 0] - diffx
#    arr[11][:, :, 1] = arr[11][:, :, 1] - diffy
#    #arr[11] = np.where(arr[11] < 0, 0, arr[11])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_12.png")
#    Y, X = np.where(np.all(img == [63, 0, 0], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base12 - x)*(-1)*(-1)
#    diffy = (Y_base12 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[12][:, :, 0] = arr[12][:, :, 0] - diffx
#    arr[12][:, :, 1] = arr[12][:, :, 1] - diffy
#    #arr[12] = np.where(arr[12] < 0, 0, arr[12])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_13.png")
#    Y, X = np.where(np.all(img == [0, 0, 127], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base13 - x)*(-1)*(-1)
#    diffy = (Y_base13 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[13][:, :, 0] = arr[13][:, :, 0] - diffx
#    arr[13][:, :, 1] = arr[13][:, :, 1] - diffy
#    #arr[13] = np.where(arr[13] < 0, 0, arr[13])
#
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_14.png")
#    Y, X = np.where(np.all(img == [0, 127, 0], axis=2))
#    y = (np.min(Y) + np.max(Y)) / 2
#    x = (np.min(X) + np.max(X)) / 2
#    diffx = (X_base14 - x)*(-1)*(-1)
#    diffy = (Y_base14 - y)*(-1)*(-1)
#    diffx = int(diffx)
#    diffy = int(diffy)
#    arr[14][:, :, 0] = arr[14][:, :, 0] - diffx
#    arr[14][:, :, 1] = arr[14][:, :, 1] - diffy
#    #arr[14] = np.where(arr[14] < 0, 0, arr[14])
#    np.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\own_sparse/" + str(i) + ".npy",arr)
#import os
#collection = r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/own_sparse"
#for k,filename in enumerate(os.listdir(collection)):
# file2 = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\own_sparse/" + filename)
# #print(file2.shape)
# file2 = (file2-32)/32
# #print(np.min(file2))
# #print(np.max(file2))
# np.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\own_sparse_ranged/" + filename,file2)



#file1 = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\coordinate_grid/0.npy")
#print(file1.shape)
#file2 = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\own_sparse/15.npy")
#print(file2.shape)
#map = file2
#img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/0s.png")
#img2 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/Untitleds.png")
##print(map.shape)
#map = map[12]
#for i in range(64):
#    for j in range(64):
#        x = map[i,j,1]
#        y = map[i, j, 0]
#        #x = (map[i,j,1]*32) + 32
#        #y = (map[i, j, 0]*32) + 32
#        #print("GG")
#        #print(str(j) + " " + str(i))
#        #print(str(int(x)) + " " +str(int(y)))
#        #diffx = j-int(x)
#        #diffy = i- int(y)
#        #print(diffx)
#        #print(diffy)
#        #if abs(diffx)>4 or abs(diffy)>4:
#        #print("GG")
#        #print(str(j) + " " + str(i))
#        if x<0:
#            x = 0
#        if y<0:
#            y = 0
#        if x>=64:
#            x = 63
#        if y>=64:
#            y = 63
#        #print(str(int(x)) + " " +str(int(y)))
#        #print("HELL")
#        img2[i,j] = img[int(x),int(y)]
#        #if i==22 and j==32 and (zimmer==0 or zimmer==18):
#        #    print(str(int(x)) + " " + str(int(y)))
#cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\TESS1.png",img2)



#from random import randrange
#from PIL import Image
#col_list = []
#for i in range(64):
#    rand_color = (randrange(255), randrange(255), randrange(255))
#    col_list.append(rand_color)
#for mi in range(45):
#  file = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\own_sparse/" + str(mi)+".npy")[13]
#  #print(file)
#  img1 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/Untitleds.png")
#  img2 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/Untitleds.png")
#  #cols = [[255,0,0], [0,255,0],  [0,0,255],  [0,0,0]]#, [127,0,0], [0,127,0], [0,0,127]  , [55,0,0]]
#  count = 0
#  for i in range(0,64,9):
#      for j in range(0,64,9):
#          elemx = file[i,j][0]
#          elemy = file[i, j][1]
#          img1[i,j] = col_list[count]
#
#          x = file[i, j][0]
#          y = file[i, j][1]
#          #x = int((file[i,j][0]*32)+32)
#          #y = int((file[i, j][1] * 32) + 32)
#          #print(y)
#          #print(x)
#          #print("Gg")
#          if y>=64:
#              print(y)
#              y = 63
#          if x>=64:
#              print(x)
#              x = 63
#          img2[y,x] = col_list[count]
#          count+=1
#  #cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/mappa1.png",img1)
#  cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizations/" + str(mi) + ".png",img2)
#  img = Image.open(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizations/" + str(mi) + ".png")
#  img = img.resize((512,512),Image.NEAREST)
#  img.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizations/" + str(mi) + ".png")
##file1 = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/frame0.npy")
##file2 = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/frame35.npy")
##file1 = (file1*32)+32
##file2 = (file2*32)+32
##print(file1)
##print("gg")
##print(file2)


#file = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/frame0.npy")[0][3]
#print(file)
#x = file[0]
#y = file[1]
#x = (x*127.5)+127.5
#y = (y*127.5)+127.5
#print(x)
#print(y)
#xo = 145
#yo = 122
#print(xo)
#print(yo)
##xo = 1
##yo = 1
###print((x*127.5)+127.5)
###print((y*127.5)+127.5)
#for i in range(15):
#    ff = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/" + str(i) + ".npy")
#    print("GG")
#    #print(ff)
#    x = (ff[1,0]*yo) +  (ff[1,1]*xo)
#    y = (ff[0,0]*xo) +  (ff[0,1]*yo)
#    print(x)
#    print(y)
###print((x*127.5)+127.5)
###print((y*127.5)+127.5)

#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/5.npy")
#identity = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/5.npy")
#print(jacobian.shape)
#print(jacobian)
#print(driving[0][0][0][0])
#x = (0.09431507*127.5)+127.5
#y = (0.7138301*127.5)+127.5
#print(x)
#print(y)
##print(identity.shape)
##print(identity)
#identity = identity[0][0]
#identity = (identity*32) + 32
#print(identity.shape)
##print(identity)
##for i in range(64):
##    for j in range(64):
#        #print(str(i) + " " + str(j))
#        #print(identity[j,i])
#
#print(driving.shape)
#driving = (driving[0][0]*32)+32
#print(driving)
#coordinate_grid = identity - driving
#print(coordinate_grid.shape)
##print(coordinate_grid[0][0].shape)
##print(coordinate_grid[0][0])
#for i in range(64):
#    for j in range(64):
#        print(str(i) + " " + str(j))
#        print(coordinate_grid[j,i])
#
#var = coordinate_grid[54,35]
#print(var)
#multx = (jacobian[0,0]*var[0]) +   (jacobian[0,1]*var[1])
#multy = (jacobian[1,0]*var[0]) +   (jacobian[1,1]*var[1])
#source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#print(multx)
#print(multy)
##print(source.shape)
##print(source[0][0])
#print(source[0][0])

#file = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/frame35.npy")
##print(file)
#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/35.npy")
##print(jacobian.shape)
##print(jacobian[0][1])
#jacobian = torch.tensor(jacobian).cuda()
#identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#identity_grid = torch.tensor(identity_grid).cuda()
#driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/35.npy")
#driving = torch.tensor(driving).cuda()
#source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
##print(source.shape)
##print(source[0][1])
#x = source[0][1][0][0][0]
#y = source[0][1][0][0][1]
#x =0.01821449
#y =0.30644906
#x = 0.01991449
#y = 0.32644906
#xv = (x*127.5)+127.5
#yv = (y*127.5)+127.5
#print(str(xv) + " " + str(yv))
#x = (x*32)+32
#y = (y*32)+32
##x = x+5
##y = y+5
#print(str(int(x)) + " " + str(int(y)))
#source= torch.tensor(source).cuda()
#coordinate_grid = identity_grid - driving
#print(coordinate_grid.shape)
##print(coordinate_grid[0][1][32,41])
##for i in range(64):
##    for j in range(64):
##        print(str(i) + " " + str(j))
##        print(coordinate_grid[0][1][j,i])
#print(coordinate_grid[0][1][int(y),int(x)])
#jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
#coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#coordinate_grid = coordinate_grid.squeeze(-1)
#driving_to_source = coordinate_grid + source.view(1, 15, 1, 1, 2)
##print(driving_to_source.shape)
##print(driving_to_source[0][1][24,33])
#xn = (driving_to_source[0][1][int(y),int(x)][0]*127.5) + 127.5
#yn = (driving_to_source[0][1][int(y),int(x)][1]*127.5) + 127.5
##xn = driving_to_source[0][1][int(y),int(x)][0]
##yn = driving_to_source[0][1][int(y),int(x)][1]
#print(str(xn) + " " + str(yn))