from scipy import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import string


def read_emnist(data_type='letters'):
    """
    注意，图片需要reshape后转置.T
    :data_type: ['letters', 'digits', 'mnist']
    :return:
    """
    test_dir = 'D:\herschel\changrong\data\eng\matlab\\emnist-byclass.mat'
    mat = io.loadmat(test_dir)
    labels = mat['dataset']['train'][0][0]['labels'][0][0]
    labels = np.squeeze(labels)
    images = mat['dataset']['train'][0][0]['images'][0][0]
    if data_type == 'letters':
        images = images[labels >= 10]
        labels = labels[labels >= 10]
        images = images[labels < 36]
        labels = labels[labels < 36] - 10
    elif data_type == 'digits':
        images = images[labels < 10]
        labels = labels[labels < 10]
    return images, labels


if __name__ == '__main__':
    images, labels = read_emnist_test('letters')
    import string
    import matplotlib.pyplot as plt
    for img_item, labels_item in zip(images, labels):
        img_item = img_item.reshape(28, 28).T
        plt.imshow(img_item)
        plt.title(string.ascii_uppercase[labels_item])
        plt.show()