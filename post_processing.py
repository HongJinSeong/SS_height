import cv2
import numpy as np
import glob


A_pahts = sorted(glob.glob('outputs/unet_attention_aug/submit/*.png'))

for idx in range(len(A_pahts)):
    image = cv2.imread(A_pahts[idx], cv2.IMREAD_GRAYSCALE)
    h,w = image.shape
    thre_out = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 15, 2)

    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(thre_out)

    # num_labels, labels_im = cv2.connectedComponents(image)


    print('ccc')



