import numpy as np
from math import cos, sin, atan2, asin
import cv2
def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))
    if R[2,0] !=1 or R[2,0] != -1:
        x = asin(R[2,0])
        y = atan2(R[2,1]/cos(x), R[2,2]/cos(x))
        z = atan2(R[1,0]/cos(x), R[0,0]/cos(x))
    else:# Gimbal lock
        z = 0 #can be anything
        if R[2,0] == -1:
            x = np.pi/2
            y = z + atan2(R[0,1], R[0,2])
        else:
            x = -np.pi/2
            y = -z + atan2(-R[0,1], -R[0,2])
    return x, y, z
def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    '''
    t2d = P[:2, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t2d


def compute_similarity_transform(points_static, points_to_transform):
    #http://nghiaho.com/?page_id=671
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T
    t0 = -np.mean(p0, axis=1).reshape(3,1)
    t1 = -np.mean(p1, axis=1).reshape(3,1)
    t_final = t1 -t0
    p0c = p0+t0
    p1c = p1+t1
    covariance_matrix = p0c.dot(p1c.T)
    U,S,V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:,2] *= -1
    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0)**2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0)**2))
    s = (rms_d0/rms_d1)
    P = np.c_[s*np.eye(3).dot(R), t_final]
    return P

def estimate_pose2(vertices2,vertices1):
    P = compute_similarity_transform(vertices2, vertices1)
    _,R,_ = P2sRt(P) # decompose affine matrix to s, R, t
    pose = matrix2angle(R)

    return R
def estimate_pose3(vertices2,vertices1):
    P = compute_similarity_transform(vertices2, vertices1)
    s,R,tra = P2sRt(P) # decompose affine matrix to s, R, t
    pose = matrix2angle(R)

    return tra

#img = cv2.imread( r"F:\image (3) - Copy.png")
#ht, wd, cc= img.shape
#cc = 3
#ww = 256
#hh = 256
#color = (0,0,0)
#result = np.full((hh,ww,cc), color, dtype=np.uint8)
#xx = (ww - wd) // 2
#yy = (hh - ht) // 2
#result[yy:yy+ht, xx:xx+wd ] = img
#cv2.imwrite(r"F:\image (3) - CopyP.png",result)


for i in range(70):
    image = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\NEW_TAR/" + str(i) + ".png")
    nn = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/frame" + str(i)  + ".npy")[0]
    for mi in range(15):
        xx = (nn[mi,0]*127.5)+127.5
        yy = (nn[mi, 1]*127.5)+127.5
        image = cv2.circle(image, (int(xx),int(yy)), 5, (0,0,255), 2)
    cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\UT/frame" +str(i) + ".png" ,image)


#from scipy.signal import savgol_filter
#maxw = -10000
#maxh = -10000
#centx_arr =[]
#centy_arr = []
#for i in range(117):
#    file = open(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\CROPKS/frame" + str(i) + ".txt",'r')
#    xx_arr =[]
#    yy_arr =[]
#    for k in range(68):
#        txt = file.readline()
#        txt = txt.strip()
#        txt = txt.split()
#        xx = float(txt[0])
#        yy = float(txt[1])
#        xx_arr.append(xx)
#        yy_arr.append(yy)
#    xx_arr = np.array(xx_arr,dtype=np.float16)
#    yy_arr = np.array(yy_arr, dtype=np.float16)
#    wid = (np.max(xx_arr)) - (np.min(xx_arr))
#    hei = (np.max(yy_arr)) - (np.min(yy_arr))
#    maxw = max(wid,maxw)
#    maxh = max(hei, maxh)
#    centx = (np.min(xx_arr)+np.max(xx_arr))/2
#    centy = (np.min(yy_arr) + np.max(yy_arr)) / 2
#    centx_arr.append(centx)
#    centy_arr.append(centy)
#maxx = max(maxw,maxh)
#wcentx =savgol_filter(np.array(centx_arr), 115, 2)
#wcenty =savgol_filter(np.array(centy_arr), 115, 2)
#for i in range(len(wcentx)):
#    img = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\Face-Depth-Network-master\assets\New_folder\New_folder/frame" + str(i) + ".png")
#    if i>91:
#        wcenty[i] = wcenty[91]
#        wcentx[i] = wcentx[91]
#    img = img[(int(wcenty[i])-int(maxx/2)):(int(wcenty[i])+int(maxx/2)),(int(wcentx[i])-int(maxx/2)):(int(wcentx[i])+int(maxx/2))]
#    img = cv2.resize(img,(256,256),cv2.INTER_LINEAR)
#    cv2.imwrite(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\NEW_TAR/frame" +  str(i) + ".png",img)


#import os
#collection = r"F:\PRESERVE_SPACE_PROJECTS\Face-Depth-Network-master\assets\New_folder\TARG\ks"
#for k, filename in enumerate(os.listdir(collection)):
#  ARRX = []
#  ARRY = []
#  file = open(r"F:\PRESERVE_SPACE_PROJECTS\Face-Depth-Network-master\assets\New_folder\TARG\ks/" + filename)
#  txt = file.readlines()
#  ARRX.append(  float((txt[0].strip()).split()[0])  )
#  ARRX.append(  float((txt[36].strip()).split()[0])  )
#  ARRX.append(  float((txt[45].strip()).split()[0])  )
#  ARRX.append(  float((txt[16].strip()).split()[0])  )
#  ARRX.append(  float((txt[30].strip()).split()[0])  )
#  ARRX.append(  float((txt[1].strip()).split()[0])  )
#  ARRX.append(  float((txt[2].strip()).split()[0])  )
#  ARRX.append(  float((txt[3].strip()).split()[0])  )
#  ARRX.append(  float((txt[4].strip()).split()[0])  )
#  ARRX.append(  float((txt[5].strip()).split()[0])  )
#  ARRX.append(  float((txt[15].strip()).split()[0])  )
#  ARRX.append(  float((txt[14].strip()).split()[0])  )
#  ARRX.append(  float((txt[13].strip()).split()[0])  )
#  ARRX.append(   float((txt[12].strip()).split()[0])   )
#  ARRX.append(   float((txt[11].strip()).split()[0])  )
#  arr = np.zeros(shape=(15,2), dtype=np.float32)
#  OLDMAX = 255
#  OLDMIN = 0
#  NEWMAX = 1
#  NEWMIN = -1
#  OldRange = (OLDMAX - OLDMIN)
#  NewRange = (NEWMAX - NEWMIN)
#  for mi in range(15):
#          oldval = ARRX[mi]
#          NewValue = (((oldval - OLDMIN) * NewRange) / OldRange) + NEWMIN
#          arr[mi,0] = NewValue
#  ARRY.append(float((txt[0].strip()).split()[1]))
#  ARRY.append(float((txt[36].strip()).split()[1]))
#  ARRY.append(float((txt[45].strip()).split()[1]))
#  ARRY.append(float((txt[16].strip()).split()[1]))
#  ARRY.append(float((txt[30].strip()).split()[1]))
#  ARRY.append(float((txt[1].strip()).split()[1]))
#  ARRY.append(float((txt[2].strip()).split()[1]))
#  ARRY.append(float((txt[3].strip()).split()[1]))
#  ARRY.append(float((txt[4].strip()).split()[1]))
#  ARRY.append(float((txt[5].strip()).split()[1]))
#  ARRY.append(float((txt[15].strip()).split()[1]))
#  ARRY.append(float((txt[14].strip()).split()[1]))
#  ARRY.append(float((txt[13].strip()).split()[1]))
#  ARRY.append(float((txt[12].strip()).split()[1]))
#  ARRY.append(float((txt[11].strip()).split()[1]))
#  OLDMAX = 255
#  OLDMIN = 0
#  NEWMAX = 1
#  NEWMIN = -1
#  OldRange = (OLDMAX - OLDMIN)
#  NewRange = (NEWMAX - NEWMIN)
#  for mi in range(15):
#      oldval = ARRY[mi]
#      NewValue = (((oldval - OLDMIN) * NewRange) / OldRange) + NEWMIN
#      arr[mi,1] = NewValue
#  np.save(r"F:\PRESERVE_SPACE_PROJECTS\Face-Depth-Network-master\assets\New_folder\TARG\ksfilt/" + filename[:-4] + ".npy",arr)
#  print(filename)
#  print(arr)