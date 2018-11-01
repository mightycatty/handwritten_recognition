# -*- coding:utf-8 -*-
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from PIL import Image
from keras.layers import Input
from keras.models import Model
from build_model import dense_cnn_fn
from config.config_digit import ModelConfig
import cv2
from data_utils_pack.data_utils import get_file_list
import random
import matplotlib.pyplot as plt
import shutil


class ModelServer(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._model_initilizer()
    
    def _model_initilizer(self):
        nclass = 11
        input = Input(shape=(ModelConfig.img_h, None, 1), name='the_input')
        y_pred = dense_cnn_fn(input, nclass)
        self.basemodel = Model(inputs=input, outputs=y_pred)
        self.basemodel_dir = self.model_dir
        modelPath = os.path.join(os.getcwd(), self.basemodel_dir)
        if os.path.exists(modelPath):
            self.basemodel.load_weights(modelPath)
        print (self.basemodel.summary())
        self.basemodel.trainable = False
        self.basemodel.save('sequential_digit')
        return

    def predict_from_dir(self, img_dir):
        img = preprocessing_from_dir_test(img_dir)
        X = np.expand_dims(img, axis=0)
        X = np.expand_dims(X, axis=-1)
        y_pred = self.basemodel.predict(X)
        y_pred = np.squeeze(y_pred, axis=0)
        out = ModelServer._decode(y_pred)
        # for debug
        # plt.title(out)
        # plt.subplot(211)
        # plt.imshow(img)
        # plt.subplot(212)
        # plt.imshow(y_pred.T)
        # plt.show()
        return out

    def batch_evaluation(self, img_folder):
        fail_saved_f = 'fail_digit'
        # try:
        #     os.remove(fail_saved_f)
        # except:
        #     pass
        os.makedirs(fail_saved_f, exist_ok=True)
        test_img_list = get_file_list(img_folder)
        random.shuffle(test_img_list)
        false_count = 0.0
        for item in test_img_list:
            label = os.path.basename(item).split('_')[0]
            result = self.predict_from_dir(item)
            # plt.title(result)
            # plt.show()
            if result != label:
                false_count += 1.
                # src = os.path.join(fail_saved_f, result+'.png')
                # while os.path.exists(src):
                #     src = os.path.join(fail_saved_f, result+'_'+str(random.randint(0, 100))+'.png')
                # shutil.copy(item, src)
        print('acc: {}'.format(1. - false_count / len(test_img_list)))
        return

    @staticmethod
    def _decode(pred, nclass=11):
        char_list = []
        pred_text = pred.argmax(axis=-1).tolist()
        for i in range(len(pred_text)):
            if pred_text[i] != nclass - 1 and (
                    (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
                char_list.append(str(pred_text[i]))
        return u''.join(char_list)


# TODO:会导致程序崩溃
def preprocessing_from_dir(img_dir):
    # 预二值化
    # plt.subplot(121)
    img = cv2.imread(img_dir, 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
    # plt.imshow(img)
    img = Image.fromarray(img)
    # 最小边框裁剪，保持高度resize
    img = img.crop(img.getbbox())
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / ModelConfig.img_h
    width = int(width / scale)
    img = img.resize([width, ModelConfig.img_h], Image.ANTIALIAS)
    img = np.array(img)
    # 重新二值化
    img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
    # plt.subplot(122)
    # plt.imshow(img)
    return img


def preprocessing_from_dir_test(img_dir):
    # 预二值化
    # plt.subplot(121)
    img = cv2.imread(img_dir, 0) / 255
    return img


if __name__ == '__main__':
    model_dir = 'D:\herschel\changrong\sequential_ocr\models\sequential_digit'
    model_server = ModelServer(model_dir)
    test_img_f = 'D:\herschel\changrong\sequential_ocr\\test_img\\train\digit\evaluation'
    # test_img_f = 'D:\herschel\changrong\sequential_ocr\\test_img\\train\digit\\emnist'
    model_server.batch_evaluation(test_img_f)
