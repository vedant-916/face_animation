import torch
import numpy as np
import cv2
import torch.nn.functional as F
#nep = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\tryout_pair\feature/0.npy")
#print(nep.shape)
#nep = torch.tensor(nep)
#nep.view(1,15,64,64)
def create_deformed_source_image(self, source_image, sparse_motions):
    """
    Eq 7. in the paper \hat{T}_{s<-d}(z)
    """
    bs, _, h, w = source_image.shape
    source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
    source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
    sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
    sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
    sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
    # print("deformed")
    # print(type(sparse_deformed))
    # print(sparse_deformed.shape)
    # np.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/deformed.npy",np.asarray(sparse_deformed.cpu()))
    return sparse_deformed

#nep = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\rough/0.npy")
#print(nep.shape)

#from random import randrange
#from PIL import Image
#col_list = []
#for i in range(64):
#    rand_color = (randrange(255), randrange(255), randrange(255))
#    col_list.append(rand_color)
#for mi in range(45):
#  file = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\rough/" + str(mi)+".npy")
#  #print(file)
#  img1 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/Untitleds.png")
#  img2 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/Untitleds.png")
#  #cols = [[255,0,0], [0,255,0],  [0,0,255],  [0,0,0]]#, [127,0,0], [0,127,0], [0,0,127]  , [55,0,0]]
#  count = 0
#  for i in range(0,64,9):
#      for j in range(0,64,9):
#          elemx = ( (file[0][2][j,i][0] *32) + 32 ) *1
#          elemy = ( (file[0][2][j,i][1] *32) + 32 ) *1
#          img1[i,j] = col_list[count]
#          #x = file[i, j][0]
#          #y = file[i, j][1]
#          ##x = int((file[i,j][0]*32)+32)
#          ##y = int((file[i, j][1] * 32) + 32)
#          ##print(y)
#          ##print(x)
#          ##print("Gg")
#          #if y>=64:
#          #    print(y)
#          #    y = 63
#          #if x>=64:
#          #    print(x)
#          #    x = 63
#          if int(elemy)<64 and int(elemx)<64 and int(elemy)>=0 and int(elemx)>=0:
#           img1[int(elemy),int(elemx)] = col_list[count]
#           start_point = (i, j)
#           end_point = (int(elemx), int(elemy))
#           color = (col_list[count][0], col_list[count][1],col_list[count][2])
#           thickness = 1
#           img1 = cv2.arrowedLine(img1, start_point, end_point,
#                                   color, thickness, tipLength=0.5)
#          count+=1
#  cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\rough\base/" + str(mi) + ".png",img1)
#  #cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\rough\mot/" + str(mi) + ".png",img2)
#  img = Image.open(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\rough\base/" + str(mi) + ".png")
#  img = img.resize((512,512),Image.NEAREST)
#  img.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\rough\base/" + str(mi) + ".png")
#  #img = Image.open(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\rough\mot/" + str(mi) + ".png")
#  #img = img.resize((512, 512), Image.NEAREST)
#  #img.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\rough\mot/" + str(mi) + ".png")


from random import randrange
import os
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import image
for zay in range(3,4):
  index = zay
  os.mkdir(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\rough\base/" + str(zay))
  for mi in range(300):
    kp0 = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\kp_driving/0.npy")
    kpp = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\kp_driving/" +str(mi)+ ".npy")
    file = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\rough/" + str(mi)+".npy")
    img1 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/Untitleds.png")
    img2 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/Untitleds.png")
    arr_base= []
    arr_motion =[]
    for i in range(0,64,7):
        for j in range(0,64,7):
            elemx = ( (file[0][index+1][j,i][0] *32) + 32 ) *1
            elemy = ( (file[0][index+1][j,i][1] *32) + 32 ) *1
            arr_base.append([i,j])
            arr_motion.append([elemx,elemy])
    xpoints =[]
    ypoints = []
    for i in range(len(arr_base)):
        xpoints.append(arr_base[i][0])
        ypoints.append(arr_base[i][1])
    xpoints = np.asarray(xpoints)
    ypoints = np.asarray(ypoints)
    data = image.imread(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\rough\images/frame" + str(mi) + ".png")
    plt.plot(xpoints, ypoints,'o',c="blue")
    xpoints = []
    ypoints = []
    for i in range(len(arr_motion)):
        xpoints.append(arr_motion[i][0])
        ypoints.append(arr_motion[i][1])
    xpoints = np.asarray(xpoints)
    ypoints = np.asarray(ypoints)
    plt.scatter(xpoints, ypoints, c="red")
    for i in range(len(arr_motion)):
        basex = arr_base[i][0]
        basey = arr_base[i][1]
        headx = arr_motion[i][0]
        heady = arr_motion[i][1]
        #print("GG")
        plt.annotate('', xy=(headx, heady), xytext=(basex, basey), arrowprops={})
    kp0x = (kp0[0][index][0] * 32) + 32
    kp0y = (kp0[0][index][1] * 32) + 32
    # arrkp0 = np.asarray([kp0x,kp0y])
    plt.plot(kp0x, kp0y,'o',c="green")
    kppx = (kpp[0][index][0] * 32) + 32
    kppy = (kpp[0][index][1] * 32) + 32
    # arrkpp = np.asarray([kppx, kppy])
    plt.plot(kppx, kppy,'o', c="purple")
    plt.annotate('', xy=(kp0x, kp0y), xytext=(kppx, kppy), arrowprops={})
    plt.imshow(data)
    plt.savefig(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\rough\base/" + str(zay)+ "/" + str(mi) + ".png")
    plt.close()


#for i in range(45):
# nep = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\kp_driving/0.npy")
# arrbx = []
# arrby = []
# for k in range(15):
#     x = (nep[0][k][0] * 32) + 32
#     y = (nep[0][k][1] * 32) + 32
#     arrbx.append(x)
#     arrby.append(y)
# arrbx = np.asarray(arrbx)
# arrby = np.asarray(arrby)
# plt.scatter(arrbx, arrby, c="blue")
# for k in range(15):
#     plt.text(arrbx[k], arrby[k], str(k))
# nep = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\kp_driving/" +str(i)+ ".npy")
# arrx = []
# arry = []
# for k in range(15):
#     #print(nep[0][k])
#     x = (nep[0][k][0]*32) + 32
#     y = (nep[0][k][1] *32) + 32
#     arrx.append(x)
#     arry.append(y)
# arrx = np.asarray(arrx)
# arry = np.asarray(arry)
# plt.scatter(arrx, arry, c="red")
# for k in range(len(arrx)):
#     basex = arrbx[k]
#     basey = arrby[k]
#     headx = arrx[k]
#     heady = arry[k]
#     plt.annotate('', xy=(headx, heady), xytext=(basex, basey), arrowprops={})
# plt.savefig(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\visualizing_aid\New folder\PLOT\kp_driving/" + str(i) + ".png")
# plt.close()

