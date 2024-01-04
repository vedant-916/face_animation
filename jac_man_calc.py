import numpy as np
import torch
import cv2
##jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\synthhesized_jacobs/15.npy")
##print(jacobian[0][9])
#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/15.npy")
#
#jacobian[0][8][0,0] = 1.4917
#jacobian[0][8][0,1] = -0.4144
#jacobian[0][8][1,0] = 0.0300
#jacobian[0][8][1,1] =0.9433
#print(jacobian[0][8])
##jacobian[0][1][0,1] =-0.38911905156475113
##jacobian[0][1][1,0] = 1.7848230264971512
##jacobian[0][1][1,1] = -0.06277820224518205
#jacobian = torch.tensor(jacobian).cuda()
#identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#identity_grid = torch.tensor(identity_grid).cuda()
#driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/15.npy")
#driving = torch.tensor(driving).cuda()
#source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
##x = source[0][1][0][0][0]
##y = source[0][1][0][0][1]
#######################################PT1
#x = 119
#y = 213
#x = (x-127.5)/127.5
#y = (y-127.5)/127.5
#X = -0.027450980392156862
#Y = 0.6313725490196078
#x = X
#y = Y
##x =-0.113725490196
##y =0.3568627450980
##x22 = 0.01991449
##y22 = 0.32644906
#xv = (x*127.5)+127.5
#yv = (y*127.5)+127.5
#print(str(xv) + " " + str(yv))
#x = (x*32)+32
#y = (y*32)+32
#source= torch.tensor(source).cuda()
#coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
#i1 = coordinate_grid[0][8][int(y),int(x)][0]
#i2 = coordinate_grid[0][8][int(y),int(x)][1]
#jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
#coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#coordinate_grid = coordinate_grid.squeeze(-1)
#driving_to_source = coordinate_grid + source.view(1, 15, 1, 1, 2)
################################PT1 PREDICT
#xn = driving_to_source[0][8][int(y),int(x)][0]
#yn = driving_to_source[0][8][int(y),int(x)][1]
#print("PT1 PREDICT: " +   str(xn*127.5+127.5) + " " + str(yn*127.5+127.5))
#
#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/15.npy")
#jacobian = torch.tensor(jacobian).cuda()
#identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#identity_grid = torch.tensor(identity_grid).cuda()
#driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/15.npy")
#driving = torch.tensor(driving).cuda()
#source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
##############################################PT2
#jacobian[0][8][0,0] = 1.4917
#jacobian[0][8][0,1] = -0.4144
#jacobian[0][8][1,0] = 0.0300
#jacobian[0][8][1,1] =0.9433
#x = 129
#y = 215
#x = (x-127.5)/127.5
#y = (y-127.5)/127.5
#X_DASH = 0.00392156862745098
#Y_DASH = 0.6705882352941176
#x = X_DASH
#y = Y_DASH
#xv = (x*127.5)+127.5
#yv = (y*127.5)+127.5
#print(str(xv) + " " + str(yv))
#x = (x*32)+32
#y = (y*32)+32
#source= torch.tensor(source).cuda()
#coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
##print(coordinate_grid[0][8][int(y),int(x)])
#i3 = coordinate_grid[0][8][int(y),int(x)][0]
#i4 = coordinate_grid[0][8][int(y),int(x)][1]
#jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
#coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#coordinate_grid = coordinate_grid.squeeze(-1)
#driving_to_source = coordinate_grid + source.view(1, 15, 1, 1, 2)
#############################################PT2 PREDICT
#xn2 = driving_to_source[0][8][int(y),int(x)][0]
#yn2 = driving_to_source[0][8][int(y),int(x)][1]
#print("PT2 PREDICT: " +str(xn2*127.5+127.5) + " " + str(yn2*127.5+127.5))

#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/15.npy")
#jacobian = torch.tensor(jacobian).cuda()
#identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#identity_grid = torch.tensor(identity_grid).cuda()
#driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/15.npy")
#driving = torch.tensor(driving).cuda()
#source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#jacobian[0][8][0,0] = 1.4917
#jacobian[0][8][0,1] = -0.4144
##############################################PT3
#x = 119
#y = 203
#x = (x-127.5)/127.5
#y = (y-127.5)/127.5
#xv = (x*127.5)+127.5
#yv = (y*127.5)+127.5
#print(str(x) + " " + str(y))
#x = (x*32)+32
#y = (y*32)+32
#source= torch.tensor(source).cuda()
#coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
##print(coordinate_grid[0][8][int(y),int(x)])
#i5 = coordinate_grid[0][8][int(y),int(x)][0]
#i6 = coordinate_grid[0][8][int(y),int(x)][1]
#jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
#coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#coordinate_grid = coordinate_grid.squeeze(-1)
#driving_to_source = coordinate_grid + source.view(1, 15, 1, 1, 2)
###############################################PT3 PREDICT
#xn3 = driving_to_source[0][8][int(y),int(x)][0]
#yn3 = driving_to_source[0][8][int(y),int(x)][1]
#print("PT3 PREDICT: " +str(xn3*127.5+127.5) + " " + str(yn3*127.5+127.5))

#S = source[0][8][0][0][0]
#S2 = source[0][8][0][0][1]
#print(S)
#a1 =4.33499834
#a2 =-2.59589886
#a3 = 1.73507929
#a4 = 3.93974879
#a5 =-2.36384918
#a6 = 2.08271951
#print(i1)
#print(i2)
#print(i3)
#print(i4)
#v1 = ((((X_DASH*a1) + (Y_DASH*a2) + a3)*i2) - (S*i2) - (i4*( (X*a1) + (Y*a2) + a3 - S))) / ( (i3*i2) - (i1*i4) )
#print(v1)
#v2 =((X*a1) + (Y*a2) + a3 - S - (v1*i1))/i2
#print(v2)
#v3 = ((((X_DASH*a4) + (Y_DASH*a5) + a6)*i2) - (S2*i2) - (i4*( (X*a4) + (Y*a5) + a6 - S2))) / ( (i3*i2) - (i1*i4) )
#print(v3)
#v4 = ((X*a4) + (Y*a5) + a6 - S2 - (v3*i1))/i2
#print(v4)
##x2 = np.asarray(xn2.cpu())
#x = np.asarray(xn.cpu())
##x2 =  np.asarray(xn2.cpu())
#x2 = (119-127.5)/127.5
##x3 =  np.asarray(xn3.cpu())
#x3 =  (127-127.5)/127.5
#y = np.asarray(yn.cpu())
##y2 =  np.asarray(yn2.cpu())
#y2 =  (183-127.5)/127.5
#y3 =  np.asarray(yn3.cpu())
#y3 =  (178-127.5)/127.5
#s1 = np.asarray(source[0][8][0][0][0].cpu())
#s2 = np.asarray(source[0][8][0][0][1].cpu())
#i1 = np.asarray(i1.cpu())
#i2 = np.asarray(i2.cpu())
#i3 = np.asarray(i3.cpu())
#i4 = np.asarray(i4.cpu())
#i5 = np.asarray(i5.cpu())
#i6 = np.asarray(i6.cpu())
#print(x2)
#print(s1)
#print(x)
#print(i1)
#print(i2)
#print(i3)
#print(i4)
#v1 =((i2*(x3+x2)) - (2*s1*i2) - (x*i4) - (x*i6) + (s1*i4) + (s1*i6))/ ((i3*i2) + (i5*i2) -(i1*i4) - (i1*i6))
#print(v1)
#v2 = ((x*((i3*i2) + (i5*i2) - (i1*i4) - (i1*i6))) - (s1*( (i3*i2) + (i5*i2) - (i1*i4) - (i1*i6))) - (i1*( (i2*(x3+x2)) - (2*s1*i2) - (x*i4) - (x*i6) + (s1*(i4+i6)))) ) / (i2*( (i3*i2) + (i5*i2) - (i1*i4) - (i1*i6)))
#print(v2)
#v3 = (i2*(y3+y2) - 2*s2*i2 - y*i4 - y*i6 + s2*i4 + s2*i6)/ (i3*i2 + i5*i2 - i1*i4 - i1*i6  )
#print(v3)
#v4 = ((y*((i3*i2) + (i5*i2) - (i1*i4) - (i1*i6))) - (s2*( (i3*i2) + (i5*i2) - (i1*i4) - (i1*i6))) - (i1*( (i2*(y3+y2)) - (2*s2*i2) - (y*i4) - (y*i6) + (s2*(i4+i6)))) ) / (i2*( (i3*i2) + (i5*i2) - (i1*i4) - (i1*i6)))
#print(v4)


#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/35.npy")
#jacobian = torch.tensor(jacobian).cuda()
#identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#identity_grid = torch.tensor(identity_grid).cuda()
#driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/35.npy")
#driving = torch.tensor(driving).cuda()
#source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#x22 = 0.01991449
#y22 = 0.32644906
#xv = (x22*127.5)+127.5
#yv = (y22*127.5)+127.5
##print(str(xv) + " " + str(yv))
#x = (x22*32)+32
#y = (y22*32)+32
#source= torch.tensor(source).cuda()
#coordinate_grid2 = identity_grid - driving
##print(coordinate_grid2[0][1][int(y),int(x)])
#i3 = coordinate_grid2[0][1][int(y),int(x)][0]
#i4 = coordinate_grid2[0][1][int(y),int(x)][1]
#jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
#coordinate_grid2 = torch.matmul(jacobian, coordinate_grid2.unsqueeze(-1))
#coordinate_grid2 = coordinate_grid2.squeeze(-1)
#driving_to_source = coordinate_grid2 + source.view(1, 15, 1, 1, 2)
##print(driving_to_source.shape)
##print(driving_to_source[0][1][24,33])
##xn2 = (driving_to_source[0][1][int(y),int(x)][0]*127.5) + 127.5
##yn2 = (driving_to_source[0][1][int(y),int(x)][1]*127.5) + 127.5
#xn2 = driving_to_source[0][1][int(y),int(x)][0]
#yn2 = driving_to_source[0][1][int(y),int(x)][1]
#print(str(xn2) + " " + str(yn2))
#x = np.asarray(xn.cpu())
#y = np.asarray(yn.cpu())
#x2 =np.asarray(xn2.cpu())
#y2 = np.asarray(yn2.cpu())
#s1 = np.asarray(source[0,1,0,0,0].cpu())
#s2 = np.asarray(source[0,1,0,0,1].cpu())
#i1 = np.asarray(i1.cpu())
#i2 = np.asarray(i2.cpu())
#i3 = np.asarray(i3.cpu())
#i4 = np.asarray(i4.cpu())

#v1 =  ((x2*i2) - (s1*i2) - (i4*x) + (s1*i4))/((i3*i2)-(i1*i4))
#v2 = ((x*i3) - (s1*i3) - (i1*x2) + (s1*i1)) / ((i3*i2) - (i1*i4))
#v3 = ((s2*i4) - (y*i4) + (i2*y2) -(i2*s2))/((i2*i3) - (i4*i1))
#v4 = ((y*i3) - (s2*i3) - (y2*i1) + (s2*i1))/((i2*i3) - (i4*i1))
#print(v1)
#print(v2)
#print(v3)
#print(v4)



#color_arr = [ [255,255,255],[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 127], [255, 127, 255], [127, 255, 255]
#     ,[255, 255, 63], [255, 63, 255], [63, 255, 255], [0, 0, 63], [0, 63, 0],[63, 0, 0], [0, 0, 127], [0, 127, 0], ]
#for i in range(45):
#    identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#    identity_grid = torch.tensor(identity_grid).cuda()
#    #driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot_numeric_arr/" + str(i) + ".npy")
#    driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/" + str(i) + ".npy")
#    driving = torch.tensor(driving).cuda()
#    source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#    source = torch.tensor(source).cuda()
#    jacob_arr = np.zeros(shape=(1,15, 2,2), dtype=np.float32)
#    for k in range(15):
#        print(k)
#        deter = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_" + str(k) + ".png")
#        colY, colX = np.where(np.all(deter == color_arr[k], axis=2))
#        colY = (np.max(colY) + np.min(colY)) / 2
#        colX = (np.max(colX) + np.min(colX)) / 2
#        PTS1 = [colX,colY]
#        X = colX
#        Y = colY
#        colX = (colX - 127.5) / 127.5
#        colY = (colY - 127.5) / 127.5
#        cor_x = colX
#        cor_y = colY
#        cor_x = (cor_x * 32) + 32
#        cor_y = (cor_y * 32) + 32
#        coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
#        #print(coordinate_grid.shape)
#        i1 = coordinate_grid[0][k][int(cor_y), int(cor_x)][0]
#        i2 = coordinate_grid[0][k][int(cor_y), int(cor_x)][1]
#        xen = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_" +str(k) + ".png")
#        xenY, xenX = np.where(np.all(xen == color_arr[k], axis=2))
#        xenY = (np.max(xenY) + np.min(xenY)) / 2
#        xenX = (np.max(xenX) + np.min(xenX)) / 2
#        print("GG: " + str(i) + " " + str(k))
#        #print(str(xenX) + " " + str(xenY))
#        PTT1 = [xenX, xenY]
#        xenY = (xenY-127.5)/127.5
#        xenX = (xenX - 127.5) / 127.5
#        xen2 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_" + str(k) + "_N.png")
#        xen2Y, xen2X = np.where(np.all(xen2 == color_arr[k], axis=2))
#        xen2Y = (np.max(xen2Y) + np.min(xen2Y)) / 2
#        xen2X = (np.max(xen2X) + np.min(xen2X)) / 2
#        #print(str(xen2X) + " " + str(xen2Y))
#        PTT2 = [xen2X, xen2Y]
#        xen2Y = (xen2Y - 127.5) / 127.5
#        xen2X = (xen2X - 127.5) / 127.5
#        deter = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_" + str(k)+"_N.png")
#        col = color_arr[k]
#        colY,colX = np.where(np.all(deter == color_arr[k], axis=2))
#        colY = (np.max(colY) + np.min(colY)) / 2
#        colX = (np.max(colX) + np.min(colX)) / 2
#        PTS2 = [colX, colY]
#        X_DASH = colX
#        Y_DASH = colY
#        colX = (colX-127.5)/127.5
#        colY = (colY - 127.5) / 127.5
#        colX = (colX * 32) + 32
#        colY = (colY * 32) + 32
#        coordinate_grid2 = identity_grid - driving.view(1, 15, 1, 1, 2)
#        i3 = coordinate_grid2[0][k][int(colY), int(colX)][0]
#        i4 = coordinate_grid2[0][k][int(colY), int(colX)][1]
#
#        xen3 = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_" + str(k) + "_N2.png")
#        xen3Y, xen3X = np.where(np.all(xen3 == color_arr[k], axis=2))
#        xen3Y = (np.max(xen3Y) + np.min(xen3Y)) / 2
#        xen3X = (np.max(xen3X) + np.min(xen3X)) / 2
#        PTT3 = [xen3X, xen3Y]
#        # print(str(xen2X) + " " + str(xen2Y))
#        xen3Y = (xen3Y - 127.5) / 127.5
#        xen3X = (xen3X - 127.5) / 127.5
#        deter = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(i) + "_" + str(k) + "_N2.png")
#        col = color_arr[k]
#        colY, colX = np.where(np.all(deter == color_arr[k], axis=2))
#        colY = (np.max(colY) + np.min(colY)) / 2
#        colX = (np.max(colX) + np.min(colX)) / 2
#        PTS3 = [colX, colY]
#        colX = (colX - 127.5) / 127.5
#        colY = (colY - 127.5) / 127.5
#        colX = (colX * 32) + 32
#        colY = (colY * 32) + 32
#        coordinate_grid3 = identity_grid - driving.view(1, 15, 1, 1, 2)
#        i5 = coordinate_grid3[0][k][int(colY), int(colX)][0]
#        i6 = coordinate_grid3[0][k][int(colY), int(colX)][1]
#        PTS = np.float32([PTS1,PTS2,PTS3])
#        PTT = np.float32([PTT1,PTT2,PTT3])
#        x = xenX
#        y = xenY
#        x2 = xen2X
#        y2 = xen2Y
#        x3 = xen3X
#        y3 = xen3Y
#        #print(source.shape)
#        s1 = np.asarray(source[0,k,0,0,0].cpu())
#        s2 = np.asarray(source[0,k,0,0,1].cpu())
#        i1 = np.asarray(i1.cpu())
#        i2 = np.asarray(i2.cpu())
#        i3 = np.asarray(i3.cpu())
#        i4 = np.asarray(i4.cpu())
#        i5 = np.asarray(i5.cpu())
#        i6 = np.asarray(i6.cpu())
#        #print(x*127.5 + 127.5)
#        #print(y*127.5 + 127.5)
#        #print(x2*127.5 + 127.5)
#        #print(y2*127.5 + 127.5)
#        #print(s1*127.5 + 127.5)
#        #print(s2*127.5 + 127.5)
#        #print(i1)
#        #print(i2)
#        #print(i3)
#        #print(i4)
#        PTS = (PTS-127.5)/127.5
#        PTT = (PTT - 127.5) / 127.5
#        M = cv2.getAffineTransform(PTS, PTT)
#        print(M)
#        a1 = M[0,0]
#        a2 = M[0, 1]
#        a3 = M[0, 2]
#        a4 = M[1, 0]
#        a5 = M[1, 1]
#        a6 = M[1, 2]
#        X = (X-127.5)/127.5
#        Y =  (Y-127.5)/127.5
#        X_DASH = (X_DASH - 127.5) / 127.5
#        Y_DASH =  (Y_DASH-127.5)/127.5
#        #v1 = ((((X_DASH * a1) + (Y_DASH * a2) + a3) * i2) - (s1 * i2) - (i4 * ((X * a1) + (Y * a2) + a3 - s1))) / (
#        #            (i3 * i2) - (i1 * i4))
#        #print(v1)
#        #v2 =((X*a1) + (Y*a2) + a3 - s1 - (v1*i1))/i2
#        #print(v2)
#        #v3 = ((((X_DASH*a4) + (Y_DASH*a5) + a6)*i2) - (s2*i2) - (i4*( (X*a4) + (Y*a5) + a6 - s2))) / ( (i3*i2) - (i1*i4) )
#        #print(v3)
#        #v4 = ((X*a4) + (Y*a5) + a6 - s2 - (v3*i1))/i2
##
#        ##v1 =((i2*(x3+x2)) - (2*s1*i2) - (x*i4) - (x*i6) + (s1*i4) + (s1*i6))/ ((i3*i2) + (i5*i2) -(i1*i4) - (i1*i6))
#        ##v2 = ((x*((i3*i2) + (i5*i2) - (i1*i4) - (i1*i6))) - (s1*( (i3*i2) + (i5*i2) - (i1*i4) - (i1*i6))) - (i1*( (i2*(x3+x2)) - (2*s1*i2) - (x*i4) - (x*i6) + (s1*(i4+i6)))) ) / (i2*( (i3*i2) + (i5*i2) - (i1*i4) - (i1*i6)))
#        ##v3 = (i2*(y3+y2) - 2*s2*i2 - y*i4 - y*i6 + s2*i4 + s2*i6)/ (i3*i2 + i5*i2 - i1*i4 - i1*i6  )
#        ##v4 = ((y*((i3*i2) + (i5*i2) - (i1*i4) - (i1*i6))) - (s2*( (i3*i2) + (i5*i2) - (i1*i4) - (i1*i6))) - (i1*( (i2*(y3+y2)) - (2*s2*i2) - (y*i4) - (y*i6) + (s2*(i4+i6)))) ) / (i2*( (i3*i2) + (i5*i2) - (i1*i4) - (i1*i6)))
#        #jacob_arr[0][k, 0, 0] = v1
#        #jacob_arr[0][k,0,1] = v2
#        #jacob_arr[0][k, 1, 0] = v3
#        #jacob_arr[0][k, 1, 1] = v4
#        #np.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\synthhesized_jacobs/" + str(i) + ".npy",jacob_arr)


##        v1 = ((x2*i2)-(s1*i2)-(i4*x)+(s1*i4))/((i3*i2)-(i1*i4))
##        v2 = ((x * i3) - (s1 * i3) - (i1 * x2) + (s1 * i1)) / ((i3 * i2) - (i1 * i4))
##        v3 = ((s2 * i4) - (y * i4) + (i2 * y2) - (i2 * s2)) / ((i2 * i3) - (i4 * i1))
##        v4 = ((y * i3) - (s2 * i3) - (y2 * i1) + (s2 * i1)) / ((i2 * i3) - (i4 * i1))
##        ##print("GG: "+ str(i) + " " + str(k))
##        ##print(str(xenX) + " " + str(xenY))
##        print(v1)
##        print(v2)
##        print(v3)
##        print(v4)
##        jacob_arr[0][k,0,0] = v1
##        jacob_arr[0][k,0,1] = v2
##        jacob_arr[0][k, 1, 0] = v3
##        jacob_arr[0][k, 1, 1] = v4
##        np.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\synthhesized_jacobs/" + str(i) + ".npy",jacob_arr)





#import math
#from scipy.io import loadmat
#import numpy as np
#from scipy.io import savemat
#mat = loadmat(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\data\my - Copy/mattyN.mat")
#VERN = np.zeros(shape=((len(mat['vertices']) + 30), 3), dtype=np.float32)
#ver0 =mat['vertices'][12571]
#xn = ver0[0]+0.05
#yn = ver0[1]+0.11
#zn = ver0[2]
#print(len(mat['vertices']))
#VERN[:len(mat['vertices'])] = mat['vertices']
#VERN[(len(mat['vertices'])+  0),0] =xn
#VERN[(len(mat['vertices']) + 0), 1] = yn
#VERN[(len(mat['vertices']) + 0), 2] = zn
#ver0 = mat['vertices'][12572]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 1), 0] = xn
#VERN[(len(mat['vertices']) + 1), 1] = yn
#VERN[(len(mat['vertices']) + 1), 2] = zn
#ver0 = mat['vertices'][12573]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 2), 0] = xn
#VERN[(len(mat['vertices']) + 2), 1] = yn
#VERN[(len(mat['vertices']) + 2), 2] = zn
#ver0 = mat['vertices'][12574]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 3), 0] = xn
#VERN[(len(mat['vertices']) + 3), 1] = yn
#VERN[(len(mat['vertices']) + 3), 2] = zn
#ver0 = mat['vertices'][12575]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 4), 0] = xn
#VERN[(len(mat['vertices']) + 4), 1] = yn
#VERN[(len(mat['vertices']) + 4), 2] = zn
#ver0 = mat['vertices'][12576]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 5), 0] = xn
#VERN[(len(mat['vertices']) + 5), 1] = yn
#VERN[(len(mat['vertices']) + 5), 2] = zn
#ver0 = mat['vertices'][12577]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 6), 0] = xn
#VERN[(len(mat['vertices']) + 6), 1] = yn
#VERN[(len(mat['vertices']) + 6), 2] = zn
#ver0 = mat['vertices'][12578]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 7), 0] = xn
#VERN[(len(mat['vertices']) + 7), 1] = yn
#VERN[(len(mat['vertices']) + 7), 2] = zn
#ver0 = mat['vertices'][12579]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 8), 0] = xn
#VERN[(len(mat['vertices']) + 8), 1] = yn
#VERN[(len(mat['vertices']) + 8), 2] = zn
#ver0 = mat['vertices'][12580]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 9), 0] = xn
#VERN[(len(mat['vertices']) + 9), 1] = yn
#VERN[(len(mat['vertices']) + 9), 2] = zn
#ver0 = mat['vertices'][12581]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 10), 0] = xn
#VERN[(len(mat['vertices']) + 10), 1] = yn
#VERN[(len(mat['vertices']) + 10), 2] = zn
#ver0 = mat['vertices'][12582]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 11), 0] = xn
#VERN[(len(mat['vertices']) + 11), 1] = yn
#VERN[(len(mat['vertices']) + 11), 2] = zn
#ver0 = mat['vertices'][12583]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 12), 0] = xn
#VERN[(len(mat['vertices']) + 12), 1] = yn
#VERN[(len(mat['vertices']) + 12), 2] = zn
#ver0 = mat['vertices'][12584]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 13), 0] = xn
#VERN[(len(mat['vertices']) + 13), 1] = yn
#VERN[(len(mat['vertices']) + 13), 2] = zn
#ver0 = mat['vertices'][12585]
#xn = ver0[0] + 0.05
#yn = ver0[1] + 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 14), 0] = xn
#VERN[(len(mat['vertices']) + 14), 1] = yn
#VERN[(len(mat['vertices']) + 14), 2] = zn
#
#ver0 =mat['vertices'][12571]
#xn = ver0[0]-0.05
#yn = ver0[1]-0.11
#zn = ver0[2]
#print(len(mat['vertices']))
#VERN[:len(mat['vertices'])] = mat['vertices']
#VERN[(len(mat['vertices'])+  15),0] =xn
#VERN[(len(mat['vertices']) + 15), 1] = yn
#VERN[(len(mat['vertices']) + 15), 2] = zn
#ver0 = mat['vertices'][12572]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 16), 0] = xn
#VERN[(len(mat['vertices']) + 16), 1] = yn
#VERN[(len(mat['vertices']) + 16), 2] = zn
#ver0 = mat['vertices'][12573]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 17), 0] = xn
#VERN[(len(mat['vertices']) + 17), 1] = yn
#VERN[(len(mat['vertices']) + 17), 2] = zn
#ver0 = mat['vertices'][12574]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 18), 0] = xn
#VERN[(len(mat['vertices']) + 18), 1] = yn
#VERN[(len(mat['vertices']) + 18), 2] = zn
#ver0 = mat['vertices'][12575]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 19), 0] = xn
#VERN[(len(mat['vertices']) + 19), 1] = yn
#VERN[(len(mat['vertices']) + 19), 2] = zn
#ver0 = mat['vertices'][12576]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 20), 0] = xn
#VERN[(len(mat['vertices']) + 20), 1] = yn
#VERN[(len(mat['vertices']) + 20), 2] = zn
#ver0 = mat['vertices'][12577]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 21), 0] = xn
#VERN[(len(mat['vertices']) + 21), 1] = yn
#VERN[(len(mat['vertices']) + 21), 2] = zn
#ver0 = mat['vertices'][12578]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 22), 0] = xn
#VERN[(len(mat['vertices']) + 22), 1] = yn
#VERN[(len(mat['vertices']) + 22), 2] = zn
#ver0 = mat['vertices'][12579]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 23), 0] = xn
#VERN[(len(mat['vertices']) + 23), 1] = yn
#VERN[(len(mat['vertices']) + 23), 2] = zn
#ver0 = mat['vertices'][12580]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 24), 0] = xn
#VERN[(len(mat['vertices']) + 24), 1] = yn
#VERN[(len(mat['vertices']) + 24), 2] = zn
#ver0 = mat['vertices'][12581]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 25), 0] = xn
#VERN[(len(mat['vertices']) + 25), 1] = yn
#VERN[(len(mat['vertices']) + 25), 2] = zn
#ver0 = mat['vertices'][12582]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 26), 0] = xn
#VERN[(len(mat['vertices']) + 26), 1] = yn
#VERN[(len(mat['vertices']) + 26), 2] = zn
#ver0 = mat['vertices'][12583]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 27), 0] = xn
#VERN[(len(mat['vertices']) + 27), 1] = yn
#VERN[(len(mat['vertices']) + 27), 2] = zn
#ver0 = mat['vertices'][12584]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 28), 0] = xn
#VERN[(len(mat['vertices']) + 28), 1] = yn
#VERN[(len(mat['vertices']) + 28), 2] = zn
#ver0 = mat['vertices'][12585]
#xn = ver0[0] - 0.05
#yn = ver0[1] - 0.11
#zn = ver0[2]
#VERN[(len(mat['vertices']) + 29), 0] = xn
#VERN[(len(mat['vertices']) + 29), 1] = yn
#VERN[(len(mat['vertices']) + 29), 2] = zn
#
#mat['vertices'] = VERN
#savemat(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_manual_calc/mattyN.mat", mat)



#x = 124
#y = 208
#x = (x-127.5)/127.5
#y = (y-127.5)/127.5
#x = (x*32)+32
#y = (y*32)+32
#source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#source =torch.tensor(source).cuda()
#print(source.shape)
#print(source[0][8])
#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/15.npy")
#jacobian = torch.tensor(jacobian).cuda()
#identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#identity_grid = torch.tensor(identity_grid).cuda()
#driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/15.npy")
#driving = torch.tensor(driving).cuda()
##print(identity_grid*32+32)
#coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
#print(coordinate_grid[0][8][46,33])
##print(coordinate_grid[0][8]*32+32)
##print(coordinate_grid[0][8][int(y),int(x)])
#
#jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
##print(jacobian.shape)
##print(jacobian[0][8])
##print(coordinate_grid.shape)
#
#coordinate_gridN = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#coordinate_gridN = coordinate_gridN.squeeze(-1)
##driving_to_source = coordinate_grid + source.view(1, 15, 1, 1, 2)
#for i in range(64):
#    for j in range(64):
#       if j >=int(y) and i>=int(x):
#        print("GG")
#        print(coordinate_grid[0][8][j,i])
#        print(jacobian[0][8][j,i])
#        print(coordinate_gridN[0][8][j,i]*127.5 + 127.5)
#
#
#
#
#
#
#coordinate_grid = coordinate_grid.squeeze(-1)
#driving_to_source = coordinate_grid + source.view(1, 15, 1, 1, 2)
#
##print(coordinate_grid[0][8][int(y),int(x)][0] *127.5 + 127.5)
##print(coordinate_grid[0][8][int(y),int(x)][1]*127.5 + 127.5)
##print(coordinate_grid[0][0])
##print(driving[0][0])




#source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#source =torch.tensor(source).cuda()
#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/15.npy")
#print(jacobian[0][8])
#jacobian = torch.tensor(jacobian).cuda()
#identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#identity_grid = torch.tensor(identity_grid).cuda()
#driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/15.npy")
#driving = torch.tensor(driving).cuda()
#coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
#FIRST_POINT = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/15_8.png")
#YSP,XSP = np.where(np.all(FIRST_POINT == [255, 63,255], axis=2))
#midx = (np.max(XSP) + np.min(XSP))/2
#midy = (np.max(YSP) + np.min(YSP))/2
#print("FIRST POINT: ")
#print(midx)
#print(midy)
#X = midx
#Y = midy
#x = (X-127.5)/127.5
#y = (Y-127.5)/127.5
#x = (x*32)+32
#y = (y*32)+32
#i1 = coordinate_grid[0][8][int(y),int(x)][0]
#i2 = coordinate_grid[0][8][int(y),int(x)][1]
#jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
#coordinate_gridN = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#coordinate_gridN = coordinate_gridN.squeeze(-1)
#driving_to_source = coordinate_gridN + source.view(1, 15, 1, 1, 2)
#xn =driving_to_source[0][8][int(y),int(x)][0]
#yn = driving_to_source[0][8][int(y),int(x)][1]
#print("FIRST POINT NETWORK PREDICT: ")
#print(xn*127.5+127.5)
#print(yn*127.5+127.5)
#FIRST_POINT_PREDICT = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_8.png")
#YSP,XSP = np.where(np.all(FIRST_POINT_PREDICT == [255, 63,255], axis=2))
#midx = (np.max(XSP) + np.min(XSP))/2
#midy = (np.max(YSP) + np.min(YSP))/2
#print("FIRST POINT PREDICT: ")
#print(midx)
#print(midy)
#FIRST_POINTX_PREDICT = (midx-127.5)/127.5
#FIRST_POINTY_PREDICT = (midy-127.5)/127.5
#
#x = np.asarray(xn.cpu())
#y = np.asarray(yn.cpu())
#s1 = np.asarray(source[0,8,0,0,0].cpu())
#s2 = np.asarray(source[0,8,0,0,1].cpu())
#i1 = np.asarray(i1.cpu())
#i2 = np.asarray(i2.cpu())
#SECOND_POINT = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/15_8_N.png")
#YSP,XSP = np.where(np.all(SECOND_POINT == [255, 63,255], axis=2))
#midx = (np.max(XSP) + np.min(XSP))/2
#midy = (np.max(YSP) + np.min(YSP))/2
#print("SECOND POINT: ")
#print(midx)
#print(midy)
#SECOND_POINTX = (midx-127.5)/127.5
#SECOND_POINTY = (midy-127.5)/127.5
#SECOND_POINTX = SECOND_POINTX*32+32
#SECOND_POINTY = SECOND_POINTY*32+32
#SECOND_POINT_PREDICT = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_8_N.png")
#YSP,XSP = np.where(np.all(SECOND_POINT_PREDICT == [255, 63,255], axis=2))
#midx = (np.max(XSP) + np.min(XSP))/2
#midy = (np.max(YSP) + np.min(YSP))/2
#print("SECOND POINT PREDICT: ")
#print(midx)
#print(midy)
#SECOND_POINTX_PREDICT = (midx-127.5)/127.5
#SECOND_POINTY_PREDICT = (midy-127.5)/127.5
#
#for i in range(64):
#    for j in range(64):
#       if j ==int(SECOND_POINTY) and i==int(SECOND_POINTX):
#        #print("GG")
#        #print(coordinate_grid[0][8][j,i])
#        #print(jacobian[0][8][j,i])
#        #print(coordinate_gridN[0][8][j,i]*127.5 + 127.5)
#        xn2 = driving_to_source[0][8][j,i][0]
#        yn2 = driving_to_source[0][8][j, i][1]
#        print("SECOND POINT NETWORK PREDICT: ")
#        print(xn2 * 127.5 + 127.5)
#        print(yn2 * 127.5 + 127.5)
#        i3 = coordinate_grid[0][8][int(j), int(i)][0]
#        i4 = coordinate_grid[0][8][int(j), int(i)][1]
#        x2 = np.asarray(xn2.cpu())
#        y2 = np.asarray(yn2.cpu())
#        i3 = np.asarray(i3.cpu())
#        i4 = np.asarray(i4.cpu())
#        #v1 = ((x2*i2)-(s1*i2)-(i4*x)+(s1*i4))/((i3*i2)-(i1*i4))
#        #v2 = ((x * i3) - (s1 * i3) - (i1 * x2) + (s1 * i1)) / ((i3 * i2) - (i1 * i4))
#        #v3 = ((s2 * i4) - (y * i4) + (i2 * y2) - (i2 * s2)) / ((i2 * i3) - (i4 * i1))
#        #v4 = ((y * i3) - (s2 * i3) - (y2 * i1) + (s2 * i1)) / ((i2 * i3) - (i4 * i1))
#        #print(v1)
#        #print(v2)
#        #print(v3)
#        #print(v4)
#        #xn2 = SECOND_POINTX_PREDICT
#        #yn2 = SECOND_POINTY_PREDICT
#        #x2 = xn2
#        #y2 = yn2
#        #x = FIRST_POINTX_PREDICT
#        #Y = FIRST_POINTY_PREDICT
#        #v1 = ((x2 * i2) - (s1 * i2) - (i4 * x) + (s1 * i4)) / ((i3 * i2) - (i1 * i4))
#        #v2 = ((x * i3) - (s1 * i3) - (i1 * x2) + (s1 * i1)) / ((i3 * i2) - (i1 * i4))
#        #v3 = ((s2 * i4) - (y * i4) + (i2 * y2) - (i2 * s2)) / ((i2 * i3) - (i4 * i1))
#        #v4 = ((y * i3) - (s2 * i3) - (y2 * i1) + (s2 * i1)) / ((i2 * i3) - (i4 * i1))
#        #print(v1)
#        #print(v2)
#        #print(v3)
#        #print(v4)


#col_arr =  [ [255,255,255],[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 127], [255, 127, 255], [127, 255, 255]
#     ,[255, 255, 63], [255, 63, 255], [63, 255, 255], [0, 0, 63], [0, 63, 0],[63, 0, 0], [0, 0, 127], [0, 127, 0], ]
#for aa in range(45):
#    jac_arr = np.zeros(shape=(1,15,2,2),dtype=np.float32)
#    for bb in range(15):
#        print("INDEX: " + str(aa) + " " + "KEYPOINT NO: " + str(bb))
#        source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#        source = torch.tensor(source).cuda()
#        jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/" + str(aa) + ".npy")
#        print(jacobian.shape)
#        print("FROM FILE: ")
#        print(jacobian[0][bb])
#        jacobian = torch.tensor(jacobian).cuda()
#        identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#        identity_grid = torch.tensor(identity_grid).cuda()
#        driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/"+ str(aa) +".npy")
#        driving = torch.tensor(driving).cuda()
#        coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
#        FIRST_POINT = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/"+ str(aa) + "_" + str(bb)+ ".png")
#        YSP, XSP = np.where(np.all(FIRST_POINT == col_arr[bb], axis=2))
#        midx = (np.max(XSP) + np.min(XSP)) / 2
#        midy = (np.max(YSP) + np.min(YSP)) / 2
#        print("FIRST POINT: ")
#        print(midx)
#        print(midy)
#        X = midx
#        Y = midy
#        x = (X - 127.5) / 127.5
#        y = (Y - 127.5) / 127.5
#        x = (x * 32) + 32
#        y = (y * 32) + 32
#        i1 = coordinate_grid[0][bb][int(y), int(x)][0]
#        i2 = coordinate_grid[0][bb][int(y), int(x)][1]
#        jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#        jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
#        coordinate_gridN = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#        coordinate_gridN = coordinate_gridN.squeeze(-1)
#        driving_to_source = coordinate_gridN + source.view(1, 15, 1, 1, 2)
#        xn = driving_to_source[0][bb][int(y), int(x)][0]
#        yn = driving_to_source[0][bb][int(y), int(x)][1]
#        print("FIRST POINT NETWORK PREDICT: ")
#        print(xn * 127.5 + 127.5)
#        print(yn * 127.5 + 127.5)
#        FIRST_POINT_PREDICT = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_" + str(bb)+".png")
#        YSP, XSP = np.where(np.all(FIRST_POINT_PREDICT == col_arr[bb], axis=2))
#        midx = (np.max(XSP) + np.min(XSP)) / 2
#        midy = (np.max(YSP) + np.min(YSP)) / 2
#        print("FIRST POINT PREDICT: ")
#        print(midx)
#        print(midy)
#        FIRST_POINTX_PREDICT = (midx - 127.5) / 127.5
#        FIRST_POINTY_PREDICT = (midy - 127.5) / 127.5
#        x = np.asarray(xn.cpu())
#        y = np.asarray(yn.cpu())
#        s1 = np.asarray(source[0, bb, 0, 0, 0].cpu())
#        s2 = np.asarray(source[0, bb, 0, 0, 1].cpu())
#        i1 = np.asarray(i1.cpu())
#        i2 = np.asarray(i2.cpu())
#        SECOND_POINT = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/" + str(aa) + "_" +str(bb)+  "_N.png")
#        YSP, XSP = np.where(np.all(SECOND_POINT == col_arr[bb], axis=2))
#        midx = (np.max(XSP) + np.min(XSP)) / 2
#        midy = (np.max(YSP) + np.min(YSP)) / 2
#        print("SECOND POINT: ")
#        print(midx)
#        print(midy)
#        SECOND_POINTX = (midx - 127.5) / 127.5
#        SECOND_POINTY = (midy - 127.5) / 127.5
#        SECOND_POINTX = SECOND_POINTX * 32 + 32
#        SECOND_POINTY = SECOND_POINTY * 32 + 32
#        SECOND_POINT_PREDICT = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/0_" + str(bb)+ "_N.png")
#        YSP, XSP = np.where(np.all(SECOND_POINT_PREDICT == col_arr[bb], axis=2))
#        midx = (np.max(XSP) + np.min(XSP)) / 2
#        midy = (np.max(YSP) + np.min(YSP)) / 2
#        print("SECOND POINT PREDICT: ")
#        print(midx)
#        print(midy)
#        SECOND_POINTX_PREDICT = (midx - 127.5) / 127.5
#        SECOND_POINTY_PREDICT = (midy - 127.5) / 127.5
#        for i in range(64):
#            for j in range(64):
#                if j == int(SECOND_POINTY) and i == int(SECOND_POINTX):
#                    # print("GG")
#                    # print(coordinate_grid[0][8][j,i])
#                    # print(jacobian[0][8][j,i])
#                    # print(coordinate_gridN[0][8][j,i]*127.5 + 127.5)
#                    xn2 = driving_to_source[0][bb][j, i][0]
#                    yn2 = driving_to_source[0][bb][j, i][1]
#                    print("SECOND POINT NETWORK PREDICT: ")
#                    print(xn2 * 127.5 + 127.5)
#                    print(yn2 * 127.5 + 127.5)
#                    i3 = coordinate_grid[0][bb][int(j), int(i)][0]
#                    i4 = coordinate_grid[0][bb][int(j), int(i)][1]
#                    x2 = np.asarray(xn2.cpu())
#                    y2 = np.asarray(yn2.cpu())
#                    i3 = np.asarray(i3.cpu())
#                    i4 = np.asarray(i4.cpu())
#                    #v1 = ((x2*i2)-(s1*i2)-(i4*x)+(s1*i4))/((i3*i2)-(i1*i4))
#                    #v2 = ((x * i3) - (s1 * i3) - (i1 * x2) + (s1 * i1)) / ((i3 * i2) - (i1 * i4))
#                    #v3 = ((s2 * i4) - (y * i4) + (i2 * y2) - (i2 * s2)) / ((i2 * i3) - (i4 * i1))
#                    #v4 = ((y * i3) - (s2 * i3) - (y2 * i1) + (s2 * i1)) / ((i2 * i3) - (i4 * i1))
#                    #print("FROM NETWORK: ")
#                    #print(v1)
#                    #print(v2)
#                    #print(v3)
#                    #print(v4)
#                    #xn2 = SECOND_POINTX_PREDICT
#                    #yn2 = SECOND_POINTY_PREDICT
#                    #x2 = xn2
#                    #y2 = yn2
#                    #x = FIRST_POINTX_PREDICT
#                    #Y = FIRST_POINTY_PREDICT
#                    #v1 = ((x2 * i2) - (s1 * i2) - (i4 * x) + (s1 * i4)) / ((i3 * i2) - (i1 * i4))
#                    #v2 = ((x * i3) - (s1 * i3) - (i1 * x2) + (s1 * i1)) / ((i3 * i2) - (i1 * i4))
#                    #v3 = ((s2 * i4) - (y * i4) + (i2 * y2) - (i2 * s2)) / ((i2 * i3) - (i4 * i1))
#                    #v4 = ((y * i3) - (s2 * i3) - (y2 * i1) + (s2 * i1)) / ((i2 * i3) - (i4 * i1))
#                    #print("OWN: ")
#                    #print(v1)
#                    #print(v2)
#                    #print(v3)
#                    #print(v4)
#                    #jac_arr[0][bb][0][0] =v1
#                    #jac_arr[0][bb][0][1] = v2
#                    #jac_arr[0][bb][1][0] = v3
#                    #jac_arr[0][bb][1][1] = v4
#                    ##print(jac_arr[0][bb])
#                    #np.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\synthhesized_jacobs/" + str(aa) + ".npy",jac_arr)


#import matplotlib.pyplot as plt
#col_arr =  [ [255,255,255],[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 127], [255, 127, 255], [127, 255, 255]
#     ,[255, 255, 63], [255, 63, 255], [63, 255, 255], [0, 0, 63], [0, 63, 0],[63, 0, 0], [0, 0, 127], [0, 127, 0], ]
#plot_arrRx = []
#plot_arrBx = []
#plot_arrRy = []
#plot_arrBy = []
#ks_arrRx = []
#ks_arrBx = []
#ks_arrRy = []
#ks_arrBy = []
#for aa in range(15,16):
#    jac_arr = np.zeros(shape=(1,15,2,2),dtype=np.float32)
#    for bb in range(14,15):
#        print("INDEX: " + str(aa) + " " + "KEYPOINT NO: " + str(bb))
#        source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#        source = torch.tensor(source).cuda()
#        jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/" + str(aa) + ".npy")
#        #print(jacobian.shape)
#        #print("FROM FILE: ")
#        print(jacobian[0][bb])
#        jacobian[0][bb][0][0] = 1.05276
#        jacobian[0][bb][0][1] = -0.0526
#        jacobian[0][bb][1][0] = -0.05263
#        jacobian[0][bb][1][1] = 1.05263
#        jacobian = torch.tensor(jacobian).cuda()
#        identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#        identity_grid = torch.tensor(identity_grid).cuda()
#        driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings_network/"+ str(aa) +".npy")
#        #print(driving[0][14]*127.5+127.5)
#        x = driving[0][bb,0]
#        y = driving[0][bb, 1]
#        #print(x*127.5+127.5)
#        #print(y*127.5+127.5)
#        driving = torch.tensor(driving).cuda()
#        coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
#        #FIRST_POINT = cv2.imread(r"F:\PRESERVE_SPACE_PROJECTS\pytorch3d-renderer\cloud_plot/"+ str(aa) + "_" + str(bb)+ ".png")
#        #YSP, XSP = np.where(np.all(FIRST_POINT == col_arr[bb], axis=2))
#        #midx = (np.max(XSP) + np.min(XSP)) / 2
#        #midy = (np.max(YSP) + np.min(YSP)) / 2
#        #print("FIRST POINT: ")
#        #print(midx)
#        #print(midy)
#        #X = midx
#        #Y = midy
#        #x = (X - 127.5) / 127.5
#        #y = (Y - 127.5) / 127.5
#        x = (x * 32) + 32
#        y = (y * 32) + 32
#        i1 = coordinate_grid[0][bb][int(y), int(x)][0]
#        i2 = coordinate_grid[0][bb][int(y), int(x)][1]
#        jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#        jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
#        #print("GG")
#        #print("FIRST")
#        #print(jacobian[0,bb,int(y),int(x)])
#        #print(coordinate_grid.unsqueeze(-1)[0,bb,int(y),int(x)])
#        cord_x1 = coordinate_grid.unsqueeze(-1)[0,bb,int(y),int(x)][0]
#        cord_y1 = coordinate_grid.unsqueeze(-1)[0, bb, int(y), int(x)][1]
#        cord_x2 = coordinate_grid.unsqueeze(-1)[0, bb, int(y)+1, int(x)+1][0]
#        cord_y2 = coordinate_grid.unsqueeze(-1)[0, bb, int(y)+1, int(x)+1][1]
#        coordinate_gridN = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#        ##print(coordinate_gridN[0,bb,int(y),int(x)]*32+32)
#        #pt1x =coordinate_gridN[0,bb,int(y),int(x)][0]
#        #pt1y = coordinate_gridN[0,bb,int(y),int(x)][1]
#        #pt1 = coordinate_gridN[0,bb,int(y),int(x)]*32+32
#        ##pt2 = (((coordinate_gridN[0,bb,int(y),int(x)]*32+32)+1) - 32)/32
#        #pt2 = coordinate_gridN[0,bb,int(y)+1,int(x)+1]
#        #pt2x = pt2[0]
#        #pt2y = pt2[1]
#        #v1 = ((pt2x*cord_y1) - (cord_y2*pt1x))/ ((cord_x2*cord_y1) - (cord_x1*cord_y2))
#        #v2 = ((pt1x*((cord_x2*cord_y1) - (cord_x1*cord_y2  ))) - (cord_x1*((pt2x*cord_y1)  - (cord_y2*pt1x))))/ (((cord_x2*cord_y1) - (cord_x1*cord_y2))*cord_y1)
#        #v3 = ((pt2y * cord_y1) - (cord_y2 * pt1y)) / ((cord_x2 * cord_y1) - (cord_x1 * cord_y2))
#        #v4 = ((pt1y * ((cord_x2 * cord_y1) - (cord_x1 * cord_y2))) - (cord_x1 * ((pt2y * cord_y1) - (cord_y2 * pt1y)))) / (((cord_x2 * cord_y1) - (cord_x1 * cord_y2)) * cord_y1)
#        #print(v1)
#        #print(v2)
#        #print(v3)
#        #print(v4)
#        #jac_arr[0][bb][0][0] = v1
#        #jac_arr[0][bb][0][1] = v2
#        #jac_arr[0][bb][1][0] = v3
#        #jac_arr[0][bb][1][1] = v4
#        ##print(jac_arr[0][bb])
#        #np.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\synthhesized_jacobs/" + str(aa) + ".npy",jac_arr)
#        ##print("SECOND")
#        ##print(jacobian[0, bb, int(y)+1, int(x)+1])
#        ##print(coordinate_grid.unsqueeze(-1)[0, bb, int(y)+1, int(x)+1])
#        ##print(coordinate_gridN[0, bb, int(y)+1, int(x)+1]*32+32)
#        ##a = coordinate_gridN[0][bb][int(y)][int(x)] * 32 + 32
#        ##b = coordinate_gridN[0][bb][int(y) + 1][int(x) + 1] * 32 + 32
#        ##print(int(y))
#        ##print(int(x))
#        ##print(a)
#        ##print(b)
#        ##coordinate_gridN = coordinate_gridN.squeeze(-1)
#        ##driving_to_source = coordinate_gridN + source.view(1, 15, 1, 1, 2)
#        ##print(source[0][bb])
#        ##print(driving_to_source[0][bb][int(y)][int(x)] *32 +32 )
#        ##print(driving_to_source[0][bb][int(y)+1][int(x)+1] * 32 + 32)
#        ##a = driving_to_source[0][bb][int(y)][int(x)] *32 +32
#        ##b = driving_to_source[0][bb][int(y)+1][int(x)+1] * 32 + 32
#        ##print((driving_to_source[0][bb][int(y)+5][int(x)+5][0] * 32 + 32)  - (driving_to_source[0][bb][int(y)][int(x)][0] *32 +32)  )
#        ##print((driving_to_source[0][bb][int(y) + 5][int(x) + 5][1] * 32 + 32) - (driving_to_source[0][bb][int(y)][int(x)][1] * 32 + 32))
#        ##xpoints = [np.asarray(i1.cpu())]
#        ##ypoints = [np.asarray(i2.cpu())]
#        ###print(xpoints[0])
#        ##diff1 = xpoints[0]
#        ###plt.subplot(1, 2, 1)
#        ##plot_arrBx.append(xpoints[0])
#        ##plot_arrBy.append(ypoints[0])
#        ##plt.scatter(xpoints, ypoints, c='blue')
#        xpoints = []
#        ypoints = []
#        for zimmer in range(int(y)-2,int(y)+2):
#          for kimmer in range(int(x)-2,int(x)+2):
#           xpoints.append(coordinate_gridN[0][bb][zimmer, kimmer][0].cpu())
#           ypoints.append(coordinate_gridN[0][bb][zimmer, kimmer][1].cpu())
#           if kimmer==25 and zimmer==36:
#               print(coordinate_gridN[0][bb][zimmer, kimmer][0].cpu())
#               print(coordinate_gridN[0][bb][zimmer, kimmer][1].cpu())
#        xpoints =  np.array(xpoints)
#        ypoints = np.array(ypoints)
#        #plot_arrRx.append(xpoints[0])
#        #plot_arrRy.append(ypoints[0])
#        ##print(xpoints[0])
#        #diff1 = abs(diff1-xpoints[0])
#        plt.scatter(xpoints, ypoints, c='red')
#        plt.title("FIRST")
#        #plt.show()
#        #i1 = coordinate_grid[0][bb][int(y)+1, int(x)+1][0]
#        #i2 = coordinate_grid[0][bb][int(y)+1, int(x)+1][1]
#        #xpoints = [np.asarray(i1.cpu())]
#        #ypoints = [np.asarray(i2.cpu())]
#        #plt.subplot(1, 2, 2)
#        xpoints = []
#        ypoints = []
#        print(str(int(x)))
#        print(str(int(y)))
#        for zimmer in range(int(y) - 2, int(y) + 2):
#            for kimmer in range(int(x) - 2, int(x) + 2):
#                xpoints.append(coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][0].cpu())
#                ypoints.append(coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][1].cpu())
#                plt.text(x=coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][0].cpu(), y=coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][1].cpu(), s=str(kimmer) + " " + str(zimmer))
#                if kimmer == 25 and zimmer == 36:
#                    print(coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][0].cpu())
#                    print(coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][1].cpu())
#                    #plt.text(x=plot_arrBx[i], y=plot_arrBy[i], s=str(i))
#        xpoints = np.array(xpoints)
#        ypoints = np.array(ypoints)
#        plt.scatter(xpoints, ypoints, c='blue')
#        #xpoints = np.array([coordinate_gridN[0][bb][int(y)+1, int(x)+1][0].cpu()])
#        #ypoints = np.array([coordinate_gridN[0][bb][int(y)+1, int(x)+1][1].cpu()])
#        #plt.scatter(xpoints, ypoints, c='red')
#        #plt.title("SECOND")
#        #i1 = coordinate_grid[0][bb][int(y) - 4, int(x) - 4][0]
#        #i2 = coordinate_grid[0][bb][int(y) - 4, int(x) - 4][1]
#        #xpoints = [np.asarray(i1.cpu())]
#        #ypoints = [np.asarray(i2.cpu())]
#        #plt.subplot(1, 2, 2)
#        #plt.scatter(xpoints, ypoints, c='blue')
#        #xpoints = np.array([coordinate_gridN[0][bb][int(y) - 4, int(x) - 4][0].cpu()])
#        #ypoints = np.array([coordinate_gridN[0][bb][int(y) - 4, int(x) - 4][1].cpu()])
#        #plt.scatter(xpoints, ypoints, c='red')
#        #plt.title("THIRD")
#        plt.show()
#        #fill0 = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd_network/frame0.npy")
#        #fill = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd_network/frame" + str(aa) + ".npy")
#        ##print(fill0.shape)
#        #xpoints = [fill0[0][bb][0]]
#        #ypoints = [fill0[0][bb][1]]
#        ##print(xpoints[0])
#        #diff2 = xpoints[0]
#        ##xpoints = [fill0[0][bb][0]*32+32]
#        ##ypoints = [fill0[0][bb][1]*32+32]
#        ##xpoints = [(xpoints[0] - 32) / 32]
#        ##ypoints = [(ypoints[0] - 32) / 32]
#        ##plt.subplot(1, 2, 2)
#        #plot_arrBx.append(xpoints[0])
#        #plot_arrBy.append(ypoints[0])
#        #plt.scatter(xpoints, ypoints, c='blue')
#        #xpoints = [fill[0][bb][0]]
#        #ypoints =[fill[0][bb][1]]
#        ##print(xpoints[0])
#        #diff2 = abs(diff2-xpoints[0])
#        #rel = diff2-diff1
#        #print(rel)
#        ##xpoints = [fill[0][bb][0]*32+32]
#        ##ypoints =[fill[0][bb][1]*32+32]
#        ##xpoints = [(xpoints[0] - 32) / 32]
#        ##ypoints = [(ypoints[0] - 32) / 32]
#        #plot_arrRx.append(xpoints[0])
#        #plot_arrRy.append(ypoints[0])
#        #plt.scatter(xpoints, ypoints, c='red')
#        ##plt.title("SECOND")
#        ##plt.show()
##for i in range(len(plot_arrRx)):
##    plt.text(x=plot_arrRx[i], y = plot_arrRy[i], s = str(i))
##    plt.text(x=plot_arrBx[i], y=plot_arrBy[i], s=str(i))
##plt.scatter(plot_arrRx, plot_arrRy, c='red')
##plt.scatter(plot_arrBx, plot_arrBy, c='blue')
##plt.title("FIRST")
##plt.show()


import matplotlib.pyplot as plt
col_arr =  [ [255,255,255],[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 127], [255, 127, 255], [127, 255, 255]
     ,[255, 255, 63], [255, 63, 255], [63, 255, 255], [0, 0, 63], [0, 63, 0],[63, 0, 0], [0, 0, 127], [0, 127, 0], ]
plot_arrRx = []
plot_arrBx = []
plot_arrRy = []
plot_arrBy = []
ks_arrRx = []
ks_arrBx = []
ks_arrRy = []
ks_arrBy = []
for aa in range(17,18):
    jac_arr = np.zeros(shape=(1,15,2,2),dtype=np.float32)
    for bb in range(14,15):
        print("INDEX: " + str(aa) + " " + "KEYPOINT NO: " + str(bb))
        source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
        source = torch.tensor(source).cuda()
        jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/" + str(aa) + ".npy")
        print(jacobian[0][bb])
        print(source[0,bb])

        #jacobian[0][bb][0][0] =1.0602689448717666
        #jacobian[0][bb][0][1] =-0.0602689448717642
        #jacobian[0][bb][1][0] =-0.05287996135730154
        #jacobian[0][bb][1][1] =1.0528799613573019
        jacobian = torch.tensor(jacobian).cuda()
        identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
        identity_grid = torch.tensor(identity_grid).cuda()
        driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings_network/"+ str(aa) +".npy")
        print(driving[0][bb, 0])
        print(driving[0][bb, 1])
        x = driving[0][bb,0]
        y = driving[0][bb, 1]
        driving = torch.tensor(driving).cuda()
        coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
        x = (x * 32) + 32
        y = (y * 32) + 32
        i1 = coordinate_grid[0][bb][int(y), int(x)][0]
        i2 = coordinate_grid[0][bb][int(y), int(x)][1]
        jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
        jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
        #cord_x1 = coordinate_grid.unsqueeze(-1)[0,bb,int(y),int(x)][0]
        #cord_y1 = coordinate_grid.unsqueeze(-1)[0, bb, int(y), int(x)][1]
        #cord_x2 = coordinate_grid.unsqueeze(-1)[0, bb, int(y)+1, int(x)+1][0]
        #cord_y2 = coordinate_grid.unsqueeze(-1)[0, bb, int(y)+1, int(x)+1][1]
        coordinate_gridN = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
        #print(int(x))
        #print(int(y))
        #print(coordinate_grid.unsqueeze(-1)[0][bb][int(y), int(x)])
        #print(coordinate_gridN[0][bb][int(y), int(x)])
        n1 = np.asarray(coordinate_gridN[0][bb][int(y), int(x)][0].cpu())[0]
        n2 = np.asarray(coordinate_gridN[0][bb][int(y), int(x)][1].cpu())[0]
        n3 = n1+0.0002
        n4 = n2+0.0002
        g1 = np.asarray(coordinate_grid.unsqueeze(-1)[0,bb,int(y),int(x)][0].cpu())[0]
        g2 = np.asarray(coordinate_grid.unsqueeze(-1)[0,bb,int(y),int(x)][1].cpu())[0]
        g3 = g1+0.0002
        g4 = g2+0.0002
        #print("N1: " + str(n1))
        #print("N2: " + str(n2))
        #print("N3: " + str(n3))
        #print("N4: " + str(n4))
        #print("G1: "+ str(g1))
        #print("G2: " + str(g2))
        #print("G3: " + str(g3))
        #print("G4: " + str(g4))
        v1 = ((n1 *((g1*g4) - (g2*g3))) - (n3*g1*g2) + (g3*n1*g2)) / (((g1*g4) - (g2*g3))*g1)
        v2 = ((n3*g1) - (g3*n1)) / ((g1*g4) - (g2*g3))
        v3 = ((n2 *((g1*g4) - (g2*g3))) - (n4*g1*g2) + (g3*n2*g2)) / (((g1*g4) - (g2*g3))*g1)
        v4 = ((n4 * g1) - (g3 * n2)) / ((g1 * g4) - (g2 * g3))
        jac_arr[0][bb][0][0] = v1
        jac_arr[0][bb][0][1] = v2
        jac_arr[0][bb][1][0] = v3
        jac_arr[0][bb][1][1] = v4
        #np.save(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\synthhesized_jacobs/" + str(aa) + ".npy",jac_arr)
        #print(v1)
        #print(v2)
        #print(v3)
        #print(v4)
        xpoints = []
        ypoints = []
        for zimmer in range(int(y)-2,int(y)+2):
          for kimmer in range(int(x)-2,int(x)+2):
           xpoints.append(coordinate_gridN[0][bb][zimmer, kimmer][0].cpu())
           ypoints.append(coordinate_gridN[0][bb][zimmer, kimmer][1].cpu())
           #if kimmer==25 and zimmer==36:
               #print(coordinate_gridN[0][bb][zimmer, kimmer][0].cpu())
               #print(coordinate_gridN[0][bb][zimmer, kimmer][1].cpu())
        xpoints =  np.array(xpoints)
        ypoints = np.array(ypoints)
        plt.scatter(xpoints, ypoints, c='red')
        plt.title("FIRST")
        xpoints = []
        ypoints = []
        for zimmer in range(int(y) - 2, int(y) + 2):
            for kimmer in range(int(x) - 2, int(x) + 2):
                xpoints.append(coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][0].cpu())
                ypoints.append(coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][1].cpu())
                plt.text(x=coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][0].cpu(), y=coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][1].cpu(), s=str(kimmer) + " " + str(zimmer))
                #if kimmer == 25 and zimmer == 36:
                    #print(coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][0].cpu())
                    #print(coordinate_grid.unsqueeze(-1)[0][bb][zimmer, kimmer][1].cpu())
        xpoints = np.array(xpoints)
        ypoints = np.array(ypoints)
        plt.scatter(xpoints, ypoints, c='blue')
        #plt.savefig(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\plts/syn" + str(aa) + ".png")
        #plt.close()
        plt.show()
