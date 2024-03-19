from skimage import io
import numpy as np
import scipy.io as sio

houston_img = io.imread("2013_IEEE_GRSS_DF_Contest_CASI.tif")
houston_arr = np.array(houston_img)
print(houston_arr.shape, houston_arr.dtype)

houston = houston_arr#.transpose(2,0,1)
print(houston.shape)

sio.savemat('Houston.mat', {'Houston': houston})
