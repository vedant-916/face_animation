#drive_sorx = sourcex+ jk(ident-kptx)
#drive_sorx = sourcex+ (v1*idkptdiffx) + (v2*idkptdiffy)
#drive_sory = sourcey+ (v3*idkptdiffx) + (v3*idkptdiffy)

#drive_sorx2 = sourcex+ (v1*idkptdiffx2) + (v2*idkptdiffy2)
#drive_sory2 = sourcey+ (v3*idkptdiffx2) + (v3*idkptdiffy2)

#  v2 =   (drive_sorx- sourcex-(v1*idkptdiffx))/idkptdiffy
# drive_sorx2 =  sourcex+ (v1*idkptdiffx2) + (  ((drive_sorx- sourcex-(v1*idkptdiffx))/idkptdiffy)  *idkptdiffy2)



#import numpy as np
#import torch
#file = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/frame35.npy")
##print(file)
#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/35.npy")
#jacobian = torch.tensor(jacobian).cuda()
#identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#identity_grid = torch.tensor(identity_grid).cuda()
#driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\kp_drivings/35.npy")
#driving = torch.tensor(driving).cuda()
#source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#x = source[0][1][0][0][0]
#y = source[0][1][0][0][1]
#x =0.01821449
#y =0.30644906
##x22 = 0.01991449
##y22 = 0.32644906
#xv = (x*127.5)+127.5
#yv = (y*127.5)+127.5
##print(str(xv) + " " + str(yv))
#x = (x*32)+32
#y = (y*32)+32
##x = x+5
##y = y+5
#source= torch.tensor(source).cuda()
#coordinate_grid = identity_grid - driving
##print(coordinate_grid[0][1][int(y),int(x)])
#
#i1 = coordinate_grid[0][1][int(y),int(x)][0]
#i2 = coordinate_grid[0][1][int(y),int(x)][1]
#
#jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#jacobian = jacobian.repeat(1, 1, 64, 64, 1, 1)
#coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#coordinate_grid = coordinate_grid.squeeze(-1)
#driving_to_source = coordinate_grid + source.view(1, 15, 1, 1, 2)
##print(driving_to_source.shape)
##print(driving_to_source[0][1][24,33])
#xn = driving_to_source[0][1][int(y),int(x)][0]
#yn = driving_to_source[0][1][int(y),int(x)][1]
##print(str(xn) + " " + str(yn))
#
#
#jacobian = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\jacobian_transforms/35.npy")
#print(jacobian[0][1])
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
##x = x+5
##y = y+5
#source= torch.tensor(source).cuda()
#coordinate_grid2 = identity_grid - driving
##print(coordinate_grid2[0][1][int(y),int(x)])
#
#i3 = coordinate_grid2[0][1][int(y),int(x)][0]
#i4 = coordinate_grid2[0][1][int(y),int(x)][1]
#
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
##print(str(xn2) + " " + str(yn2))
#
#
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
##print(x)
##print(y)
##print(x2)
##print(y2)
##print(s1)
##print(s2)
#
##v1 =  ((x2*i2) - (s1*i2) - (i4*x) + (s1*i4))/((i3*i2)-(i1*i4))
##v2 = ((x*i3) - (s1*i3) - (i1*x2) + (s1*i1)) / ((i3*i2) - (i1*i4))
##v3 = ((s2*i4) - (y*i4) + (i2*y2) -(i2*s2))/((i2*i3) - (i4*i1))
##v4 = ((y*i3) - (s2*i3) - (y2*i1) + (s2*i1))/((i2*i3) - (i4*i1))
##print(v1)
##print(v2)
##print(v3)
##print(v4)
#
#
#
#for i in range(1):
#    identity_grid = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/identity_grid.npy")
#    identity_grid = torch.tensor(identity_grid).cuda()
#    driving = np.load(r"F:\PRESERVE_SPACE_PROJECTS\ZoeDepth\KSd/" + str(i) + ".npy")
#    driving = torch.tensor(driving).cuda()
#    source = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS/kp_SOURCE.npy")
#    source = torch.tensor(source).cuda()
#    for k in range(15):
#        cor_x = source[0][k][0][0][0]
#        cor_y = source[0][k][0][0][1]
#        cor_x = (cor_x * 32) + 32
#        cor_y = (cor_y * 32) + 32
#        coordinate_grid = identity_grid - driving.view(1, 15, 1, 1, 2)
#        i1 = coordinate_grid[0][1][int(cor_y), int(cor_x)][0]
#        i2 = coordinate_grid[0][1][int(cor_y), int(cor_x)][1]
#
#
#
#        x = np.asarray(xn.cpu())
#        y = np.asarray(yn.cpu())
#        x2 = np.asarray(xn2.cpu())
#        y2 = np.asarray(yn2.cpu())
#        s1 = np.asarray(source[0, k, 0, 0, 0].cpu())
#        s2 = np.asarray(source[0, k, 0, 0, 1].cpu())
#        i1 = np.asarray(i1.cpu())
#        i2 = np.asarray(i2.cpu())
#        i3 = np.asarray(i3.cpu())
#        i4 = np.asarray(i4.cpu())
#
#        #v1 = ((x2 * i2) - (s1 * i2) - (i4 * x) + (s1 * i4)) / ((i3 * i2) - (i1 * i4))
#        #v2 = ((x * i3) - (s1 * i3) - (i1 * x2) + (s1 * i1)) / ((i3 * i2) - (i1 * i4))
#        #v3 = ((s2 * i4) - (y * i4) + (i2 * y2) - (i2 * s2)) / ((i2 * i3) - (i4 * i1))
#        #v4 = ((y * i3) - (s2 * i3) - (y2 * i1) + (s2 * i1)) / ((i2 * i3) - (i4 * i1))

import torch
import  numpy as np
feature = np.load(r"F:\PRESERVE_SPACE_PROJECTS\CVPR2022-DaGAN\SPARSE_MOTIONS\feature_maps/5.npy")
feature = torch.tensor(feature)
print(feature.shape)
pred_mask = torch.argmax(feature, dim=1)
print(pred_mask.shape)
print(pred_mask)
#pred_mask = pred_mask[..., tf.newaxis]
#print(pred_mask[0])