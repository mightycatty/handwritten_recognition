import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from data_utils_pack.data_utils import get_file_list
from tqdm import tqdm
img_f = 'C:\\Users\\v5-Server\Desktop\digit_train_model-master-c21b2c4f8394d66050e80f3a97973a0c71a00f4a\digit_train_model-master-c21b2c4f8394d66050e80f3a97973a0c71a00f4a\data\merge_data'
data = []
label = []
img_f_list = []
for i in range(10):
    img_f_item = os.path.join(img_f, str(i))
    img_f_list += get_file_list(img_f_item)
    label += [i]*len(get_file_list(img_f_item))
bar = tqdm(total=len(img_f_list))
for item in img_f_list:
    img_data = cv2.imread(item, 0)
    img_data = cv2.threshold(img_data, 0, 1, cv2.THRESH_OTSU+cv2.THRESH_BINARY)[-1]
    data.append(img_data)
    bar.update()
np.random.seed(0)
np.random.shuffle(data)
np.random.seed(0)
np.random.shuffle(label)
data = np.stack(data, axis=0)
label = np.array(label, dtype=np.uint8)
for img_item, label_item in zip(data, label):
    img_item = np.squeeze(img_item)
    plt.imshow(img_item)
    plt.title(str(label_item))
    plt.show()
a = 1

