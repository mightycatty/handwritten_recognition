"""
data utils functions
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def get_file_list(folder_dir):
    """
    遍历目录下的所有文件
    :param folder_dir:
    :return:
    """
    file_list = []
    for root, dirs, files in os.walk(folder_dir, topdown=False):
        for name in files:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
        for name in dirs:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
    return file_list


def resize_gray_with_padding(im, desired_size):
    """
    通过黑色填充边框的形式保持长宽比例得到一个正方形图
    :param im:
    :param desired_size:
    :return:
    """
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("L", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))

    return im


def resize_width(img, width=128):
    """
    将图片通过填充黑色到指定宽度
    :param img:
    :param width:
    :return:
    """
    import cv2
    h = img.shape[0]
    if img.shape[1] > width:
        img = cv2.resize(img.astype(np.uint8), (width, h))
        mask = img
    else:
        mask = np.zeros((h, width), dtype=img.dtype)
        mask[:h, :img.shape[1]] = img
    return mask


def inference_img_prepare(img_dir, img_h):
    """
    将图像读取并为inference做准备，输出数据可以直接输入神经网络
    :param img_dir:
    :param img_h:
    :return:
    """
    img = Image.open(img_dir)
    img = np.array(img)
    img = np.where(img > 50, 255, 0)
    img = Image.fromarray(img)
    img = img.crop(img.getbbox())
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / img_h
    width = int(width / scale)
    img = img.resize([width, img_h], Image.ANTIALIAS)
    img = np.array(img).astype(np.float32) / 255.0
    # plt.imshow(img)
    img = np.where(img > 0.1, 1, 0)
    X = img.reshape([1, img_h, width, 1])
    return X


if __name__ == '__main__':
    im = Image.open('D:\herschel\changrong\sequential_ocr\\test_img\synthesis\\52540865-5bf3-4639-b56b-98d0bb02d9e6.png')
    im = resize_gray_with_padding(im, 48)
    im.show()
