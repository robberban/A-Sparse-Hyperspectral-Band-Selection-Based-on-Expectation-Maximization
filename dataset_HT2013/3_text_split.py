import numpy as np
import scipy.io
import numpy as np

import scipy.io as sio

def randomize_pixels(img):
    img_copy_test = img.copy()  
    h, w = img_copy_test.shape  
    mask_all = (img_copy_test == -1)
    class_num = 15
    for i in range(class_num):
        i = i + 1
        mask = (img_copy_test == i)  
        indices = np.random.choice(np.flatnonzero(mask), int(np.sum(mask) * 0.1), replace=False)#
        mask_all.flat[indices] = True
    img_copy_test[mask_all] = 0

    return  img - img_copy_test, img_copy_test


data = scipy.io.loadmat('Houston_gt.mat')
spectral_data = data['Houston_gt']
spectral_data = np.array(spectral_data)
print(spectral_data.shape,'shape')
print(spectral_data.min(),'min')
print(spectral_data.max(),'max')

img_randomized_train_5, img_randomized_test_5 = randomize_pixels(spectral_data)  #
print(img_randomized_train_5.sum(),'train')
print(img_randomized_test_5.sum(),'test')

sio.savemat('Houston_gt_train_10.mat', {'Houston_gt_train_10': img_randomized_train_10})
sio.savemat('Houston_gt_test_10.mat', {'Houston_gt_test_10': img_randomized_test_10})




