# -*- coding:utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from PIL import Image
from keras.layers import Input
from keras.models import Model
from build_model import dense_cnn_fn
from config.config_digit import ModelConfig
import cv2
from data_utils_pack.data_utils import get_file_list
import random
nclass = 11

input = Input(shape=(ModelConfig.img_h, None, 1), name='the_input')
y_pred = dense_cnn_fn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)
basemodel_dir = 'D:\herschel\changrong\sequential_ocr\models\digit_model\digit-0.53-0.76.h5'
modelPath = os.path.join(os.getcwd(), basemodel_dir)
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and (
                (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(str(pred_text[i]))
    return u''.join(char_list)


def predict(img):
    img = img.crop(img.getbbox())
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / ModelConfig.img_h
    width = int(width / scale)

    img = img.resize([width, ModelConfig.img_h], Image.ANTIALIAS)

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0
    img[img > 0.1] = 1
    # plt.imshow(img)
    X = img.reshape([1, ModelConfig.img_h, width, 1])

    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)

    return out


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[-1]
    return img


def evaluation():
    test_img_folder = 'D:\herschel\changrong\sequential_ocr\\test_img\\real\digit_coarse'
    test_img_list = get_file_list(test_img_folder)
    random.shuffle(test_img_list)
    false_count = 0.0
    for item in test_img_list:
        im = Image.open(item)
        label = os.path.basename(item).split('_')[0]
        # img_data = preprocessing(np.array(im))
        # im = Image.fromarray(img_data)
        result = predict(im)
        if result != label:
            false_count += 1.
    print ('acc for coarse: {}'.format(1. - false_count/len(test_img_list)))
    test_img_folder = 'D:\herschel\changrong\sequential_ocr\\test_img\\real\digit_fine'
    test_img_list = get_file_list(test_img_folder)
    random.shuffle(test_img_list)
    false_count = 0.0
    for item in test_img_list:
        im = Image.open(item)
        label = os.path.basename(item).split('_')[0]
        # img_data = preprocessing(np.array(im))
        # im = Image.fromarray(img_data)
        result = predict(im)
        if result != label:
            false_count += 1.
    print('acc for fine: {}'.format(1. - false_count / len(test_img_list)))
    return


if __name__ == '__main__':
   evaluation()