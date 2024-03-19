from skimage import io
import numpy as np
import scipy.io as sio

houston_img = io.imread("HD_groun_truth.tif")
houston_arr = np.array(houston_img)
print(houston_arr.shape, houston_arr.dtype)

#houston_arr = houston_arr.transpose()
print(houston_arr.shape)

gt = np.zeros((houston_arr.shape[0], houston_arr.shape[1]))

for i in range(houston_arr.shape[0]):
    for j in range(houston_arr.shape[1]):
        if (houston_arr[i][j] == np.array([0, 0, 0])).all():  #background
            gt[i][j] = 0
        if (houston_arr[i][j] == np.array([0, 205, 0])).all(): #grass_healthy
            gt[i][j] = 1
        if (houston_arr[i][j] == np.array([127, 255, 0])).all(): #grass_stressed
            gt[i][j] = 2     
        if (houston_arr[i][j] == np.array([46, 139, 87])).all(): #grass_synthetic
            gt[i][j] = 3     
        if (houston_arr[i][j] == np.array([0, 139, 0])).all(): #tree
            gt[i][j] = 4    
        if (houston_arr[i][j] == np.array([160, 82, 45])).all(): #soil
            gt[i][j] = 5    
        if (houston_arr[i][j] == np.array([0, 255, 255])).all(): #water
            gt[i][j] = 6    
        if (houston_arr[i][j] == np.array([255, 255, 255])).all(): #residential
            gt[i][j] = 7    
        if (houston_arr[i][j] == np.array([216, 191, 216])).all(): #commercial
            gt[i][j] = 8    
        if (houston_arr[i][j] == np.array([255, 0, 0])).all(): # road
            gt[i][j] = 9    
        if (houston_arr[i][j] == np.array([139, 0, 0])).all(): #highway
            gt[i][j] = 10    
        if (houston_arr[i][j] == np.array([205, 205, 0])).all(): #railway
            gt[i][j] = 11    
        if (houston_arr[i][j] == np.array([255, 255, 0])).all(): #parking_lot1
            gt[i][j] = 12    
        if (houston_arr[i][j] == np.array([238, 154, 0])).all(): #parking_lot2
            gt[i][j] = 13    
        if (houston_arr[i][j] == np.array([85, 26, 139])).all(): #tennis_court
            gt[i][j] = 14    
        if (houston_arr[i][j] == np.array([255, 127, 80])).all(): #running_track
            gt[i][j] = 15    

nb_classes = int(gt.max())
cls, count = np.unique(gt, return_counts=True)
TOTAL_SIZE = np.sum(count[1:])
print(cls, count)
print('The class numbers of the HSI data is:', nb_classes)
print('The total size of the labeled data is:', TOTAL_SIZE)


import scipy.io as sio
sio.savemat('Houston_gt.mat', {'Houston_gt': gt})