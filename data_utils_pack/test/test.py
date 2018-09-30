import cv2
import numpy as np
import matplotlib.pyplot as plt


img_dir = 'D:\herschel\changrong\data\sequential_ocr\\letter_coarse_test.jpg'
img = cv2.imread('D:\herschel\changrong\data\sequential_ocr\\letter_coarse.jpg', 0)
img_with_frame = cv2.imread(img_dir)
gray = cv2.cvtColor(img_with_frame,cv2.COLOR_BGR2GRAY)
# img_bg = np.where(gray<10, 1, 0).astype(np.uint8)
kernel = np.ones((1, 8), np.uint8)
img_h_bg = 1 - cv2.dilate(gray, kernel, iterations=5) / 255
split_h_index = np.where(np.sum(img_h_bg, axis=1)>50)[0].tolist()
split_h_index.append(gray.shape[0])
kernel = np.ones((8, 1), np.uint8)
img_w_bg = 1 - cv2.dilate(gray, kernel, iterations=5) / 255
split_w_index = np.where(np.sum(img_w_bg, axis=0)> 50)[0].tolist()
split_w_index.append(gray.shape[1])
h_item_prev = 0
w_item_prev = 0
count = 0
for h_item in split_h_index:
    if abs(h_item - h_item_prev) < 10:
        continue
    else:
        for w_item in split_w_index:
            if abs(w_item - w_item_prev) < 10:
                continue
            else:
                crop_item = 255 - img[h_item_prev:h_item, w_item_prev:w_item]
                cv2.imwrite('{}.png'.format(count), crop_item)
                count += 1
            w_item_prev = w_item
    w_item_prev = 0
    h_item_prev = h_item


