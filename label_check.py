# train label 기준 제일 작은 값 0 , 제일 큰값 170
# simulation의  폴더별로 보니까 max 값이 140 150 160 170으로 나뉘며 해당 max값이 외곽 테두리 값이며 생각보다 넓은영역임..
import glob
import cv2
import numpy as np
import skimage.io as iio
from scipy import fftpack

# A_pahts = sorted(glob.glob('simulation_data/Depth/*/*/*.png'))
A_pahts = sorted(glob.glob('simulation_data/SEM/*/*/*.png'))
# A_pahts = sorted(glob.glob('train/SEM/*/*/*.png'))
# A_pahts = sorted(glob.glob('test/*/*.png'))
# simulation_depth_paths = sorted(glob.glob('simulation_data/Depth/*'))

Avals=[]
Bvals=[]

## normalize를 -1 ~ 1 로 해주기 때문에 똑같이 진행
# for idx in range(len(A_pahts)):
#
#     aval = cv2.imread(A_pahts[idx])
#
#     Avals.append(np.max(aval))
#     Bvals.append(np.min(aval))
#     # if np.max(aval)>140:
#     #     print(A_pahts[idx])
#
#     if idx % 5000==0:
#         print(str(idx)+' END!')
#         print(np.max(np.array(Avals)))
#         print(np.min(np.array(Bvals)))
#
#
# print(np.unique(np.array(Avals), return_counts=True))
# print(np.unique(np.array(Bvals)))
# print(np.max(np.array(Avals)))

# print(np.min(np.array(Bvals)))

vals = []

#
# for f_paths in simulation_depth_paths:
#     paths = sorted(glob.glob(f_paths+'/*/*'))
#
#     cls_val=[]
#     for i, path in enumerate(paths):
#         sem_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         cls_val.append(np.mean(sem_img))
#
#         if i%1000==0:
#             print(str(i)+' END!')
#     print(np.mean(np.array(cls_val)), np.min(np.array(cls_val)), np.max(np.array(cls_val)))
#     vals.append(cls_val)
#
# print ('cccccccc')


#
# def noisy(image):
#     row,col= image.shape
#     mean = 0
#     var = 0.1
#     sigma = var**0.5
#     gauss = np.random.normal(mean,sigma,(row,col))
#     gauss = gauss.reshape(row,col)
#     noisy = image + gauss*50
#     return noisy
#
for idx in range(len(A_pahts)):

    image = iio.imread(A_pahts[idx], as_gray=True)
    # image = noisy(image)
    # cv2.imwrite('kkk4_4.png', image)
    # im_fft = fftpack.fft2(image)
    # keep_fraction = 0.15
    # im_fft2 = im_fft.copy()
    # r, c = im_fft2.shape
    # im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    # im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    # im_new = fftpack.ifft2(im_fft2).real

    denoise1 = cv2.bilateralFilter(image, 5, 50, 50)

    clahe = cv2.createCLAHE(clipLimit=1.333, tileGridSize=(3,3))
    img2 = clahe.apply(denoise1)

    # sem_img = iio.imread(A_pahts[idx], as_gray=True)

    # denoise1 = cv2.GaussianBlur(image,(7,7),1)
    cv2.imwrite('kkkk55.png',denoise1)
    cv2.imwrite('kkkk11.png', img2)


