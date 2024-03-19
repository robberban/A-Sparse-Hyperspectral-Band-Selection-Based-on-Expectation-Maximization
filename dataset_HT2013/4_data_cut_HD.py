import numpy as np
from skimage import io
import cv2
# from libtiff import TIFF
import gc
import scipy.io as sio
import scipy.io


waterLabelPath_train = '/home/glk/datasets/HIC_dataset/2013_DFTC/'
waterImgRootPath_train = '/home/glk/datasets/HIC_dataset/2013_DFTC/'
waterImgRootPath_train_save = '/home/glk/datasets/HIC_dataset/2013_DFTC/image_test_cut_10_64/'#test#train
waterLabelPath_train_save = '/home/glk/datasets/HIC_dataset/2013_DFTC/label_test_cut_10_64/'#

image_name = 'Houston.mat'
label_name = 'Houston_gt_test_10.mat'

imgpath = waterImgRootPath_train
image_path = imgpath + image_name
label_path = waterLabelPath_train + label_name
image_session_name = image_name.split('.')[0]
label_session_name = label_name.split('.')[0]


#349 * 1905
gap = 8
num_x = (349-64) // 8
num_y = (1905-64) // 8
length = 64




#open_data

data = scipy.io.loadmat(image_path)
spectral_data = data[image_session_name]
Image = np.array(spectral_data)
print(Image.shape,"Image shape")

data = scipy.io.loadmat(label_path)
spectral_data = data[label_session_name]
label = np.array(spectral_data)
print(label.shape,"Label shape")

name = image_session_name
f = open('/home/glk/datasets/HIC_dataset/2013_DFTC/test_cut_10_64.txt', 'w')#train_cut_10_64.txt
times = 0
for j in range(num_x):
    for k in range(num_y):
        point_x = int(j * gap)
        point_y = int(k * gap)
        Image_tmp = Image[point_x:point_x+64,point_y:point_y+64,:]
        # print(Image_tmp.shape)
        Label_tmp = label[point_x:point_x+64,point_y:point_y+64]
        #save_data
        name_new = name + '_' + str(times)
        # print((Label_tmp!=0).sum()/(64*64))#
        if (Label_tmp!=0).sum() == 0:
            # print((Label_tmp!=0).sum()/(64*64))
            times += 1
            continue
        np.save(waterImgRootPath_train_save + name_new, Image_tmp)
        try:
            cv2.imwrite(waterLabelPath_train_save + name_new + '.png', Label_tmp)
        except:
            print(Label_tmp.shape)
            print(label.shape)
            # cv2.imwrite(waterLabelPath_train_save + name_new + '.png', Label_tmp)
        A = np.load(waterImgRootPath_train_save + name_new + '.npy')
        a,b,c = A.shape
        #sava_txt
        f.writelines(name_new + '.png' + '\n')
        print(name_new + '.png' + '\n')
        times += 1
        # break
    # break
f.close()