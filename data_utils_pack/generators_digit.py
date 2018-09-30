import os
import random
import string

import cv2
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha

from data_utils_pack.data_synthesis import SequentialDigit
from data_utils_pack.data_utils import get_file_list
from data_utils_pack.letter_utils import read_emnist


def generator_with_cap(batch_size, width=256, height=64, max_len=6):
    """
    利用开源二维码生成器生成数据
    :param batch_size:
    :return:
    """
    # fonts = None
    fonts_size = (int(0.75*height), int(0.75*height))
    image = ImageCaptcha(width=width, height=height, font_sizes=fonts_size)
    while True:
        # 标签
        x_batch = []
        labels = np.ones([batch_size, max_len]) * 10000
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        for i in range(batch_size):
            N = random.randint(1, max_len)
            y = ''.join(random.SystemRandom().choice(string.digits) for _ in range(N))
            x = image.generate_image(y)
            x = np.array(x)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.threshold(x, 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[-1]
            x_batch.append(x)
            input_length[i] = width // 8
            labels[i, :len(y)] = [int(k) for k in y]
            label_length[i] = len(y)
        x_batch = np.stack(x_batch, axis=0)
        x_batch = np.expand_dims(x_batch, axis=-1)
        inputs = {'the_input': x_batch,
                  'the_labels': labels.astype(np.float32),
                  'input_length': input_length.astype(np.int64),
                  'label_length': label_length.astype(np.int64),
                  }
        outputs = {'ctc': np.zeros([batch_size])}
        yield inputs, outputs


def generator_with_folder_img(batch_size, width=256, height=64, max_len=6):
    """
    从文件夹中的图片生成序列数字，主要来源于cr提供的用户数据
    :param batch_size:
    :return:
    """
    # fonts = None
    fonts_size = int(0.75*height)
    seq = SequentialDigit(width=width, height=height, font_sizes=fonts_size)
    resource_folder = 'D:\herschel\changrong\data\digit\digital\split\split_red'
    img_dir_list = get_file_list(resource_folder)
    random.shuffle(img_dir_list)
    while True:
        # 标签
        x_batch = []
        labels = np.ones([batch_size, max_len]) * 10000
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        for i in range(batch_size):
            select_index = np.random.randint(0, len(img_dir_list), (random.randint(1, max_len)))
            img_list_s = [img_dir_list[item] for item in select_index]
            y = ''.join(item.split('\\')[-2] for item in img_list_s)
            img_data_list = [Image.open(item) for item in img_list_s]
            x = seq.generate_image(img_data_list)
            x = np.array(x)
            # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.threshold(x, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
            x_batch.append(x)
            input_length[i] = width // 8
            labels[i, :len(y)] = [int(k) for k in y]
            label_length[i] = len(y)
        x_batch = np.stack(x_batch, axis=0)
        x_batch = np.expand_dims(x_batch, axis=-1)
        inputs = {'the_input': x_batch,
                  'the_labels': labels.astype(np.float32),
                  'input_length': input_length.astype(np.int64),
                  'label_length': label_length.astype(np.int64),
                  }
        outputs = {'ctc': np.zeros([batch_size])}
        yield inputs, outputs


def generator_with_legacy_data(batch_size, width=256, height=64, max_len=6):
    """
    利用历史30w数据生成序列数字，mnist与中国数据混合
    :param batch_size:
    :return:
    """
    # fonts = None
    fonts_size = int(0.75*height)
    seq = SequentialDigit(width=width, height=height, font_sizes=fonts_size)
    legacy_npy_dir = 'D:\herschel\changrong\data\digit\old'
    file_name = ['train_data.npy', 'labels.npy']
    data = np.load(os.path.join(legacy_npy_dir, file_name[0])) * 255
    data = data.reshape((-1, 32, 32))
    label = np.load(os.path.join(legacy_npy_dir, file_name[1]))
    label = np.argmax(label, axis=-1).astype(np.uint8)
    while True:
        # 标签
        x_batch = []
        labels = np.ones([batch_size, max_len]) * 10000
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        for i in range(batch_size):
            select_index = np.random.randint(0, data.shape[0], (random.randint(1, max_len)))
            img_data_list = [Image.fromarray(np.squeeze(item)) for item in data[select_index]]
            y = ''.join(str(item) for item in label[select_index])
            x = seq.generate_image(img_data_list)
            x = np.array(x)
            # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.threshold(x, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
            x_batch.append(x)
            input_length[i] = width // 8
            labels[i, :len(y)] = [int(k) for k in y]
            label_length[i] = len(y)
        x_batch = np.stack(x_batch, axis=0)
        x_batch = np.expand_dims(x_batch, axis=-1)
        inputs = {'the_input': x_batch,
                  'the_labels': labels.astype(np.float32),
                  'input_length': input_length.astype(np.int64),
                  'label_length': label_length.astype(np.int64),
                  }
        outputs = {'ctc': np.zeros([batch_size])}
        yield inputs, outputs


def generator_with_mnist(batch_size, width=256, height=64, max_len=6):
    """
    利用历史30w数据生成序列数字，mnist与中国数据混合
    :param batch_size:
    :return:
    """
    # fonts = None
    fonts_size = int(0.75*height)
    seq = SequentialDigit(width=width, height=height, font_sizes=fonts_size)
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    data = x_train
    data = data.reshape((-1, 28, 28))
    label = y_train
    while True:
        # 标签
        x_batch = []
        labels = np.ones([batch_size, max_len]) * 10000
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        for i in range(batch_size):
            select_index = np.random.randint(0, data.shape[0], (random.randint(1, max_len)))
            img_data_list = [Image.fromarray(np.squeeze(item)) for item in data[select_index]]
            y = ''.join(str(item) for item in label[select_index])
            x = seq.generate_image(img_data_list)
            x = np.array(x)
            # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.threshold(x, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
            x_batch.append(x)
            input_length[i] = width // 8
            labels[i, :len(y)] = [int(k) for k in y]
            label_length[i] = len(y)
        x_batch = np.stack(x_batch, axis=0)
        x_batch = np.expand_dims(x_batch, axis=-1)
        inputs = {'the_input': x_batch,
                  'the_labels': labels.astype(np.float32),
                  'input_length': input_length.astype(np.int64),
                  'label_length': label_length.astype(np.int64),
                  }
        outputs = {'ctc': np.zeros([batch_size])}
        yield inputs, outputs


def generator_with_emnist(batch_size, width=256, height=64, max_len=6):
    """
    :param batch_size:
    :return:
    """
    # fonts = None
    fonts_size = int(0.75*height)
    seq = SequentialDigit(width=width, height=height, font_sizes=fonts_size)
    x_train, y_train = read_emnist('digits')
    data = x_train
    data = data.reshape((-1, 28, 28))
    label = y_train
    while True:
        # 标签
        x_batch = []
        labels = np.ones([batch_size, max_len]) * 10000
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        for i in range(batch_size):
            select_index = np.random.randint(0, data.shape[0], (random.randint(1, max_len)))
            img_data_list = [Image.fromarray(np.squeeze(item).T) for item in data[select_index]]
            y = ''.join(str(item) for item in label[select_index])
            x = seq.generate_image(img_data_list)
            x = np.array(x)
            # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.threshold(x, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
            x_batch.append(x)
            input_length[i] = width // 8
            labels[i, :len(y)] = [int(k) for k in y]
            label_length[i] = len(y)
        x_batch = np.stack(x_batch, axis=0)
        x_batch = np.expand_dims(x_batch, axis=-1)
        inputs = {'the_input': x_batch,
                  'the_labels': labels.astype(np.float32),
                  'input_length': input_length.astype(np.int64),
                  'label_length': label_length.astype(np.int64),
                  }
        outputs = {'ctc': np.zeros([batch_size])}
        yield inputs, outputs


def merge_generator(batch_size, width=256, height=64, max_len=6):
    per = [0, 3, 1, 1]
    cap_gen = generator_with_cap(int(batch_size*per[0]/sum(per))+1, width, height, max_len)
    folder_gen = generator_with_folder_img(int(batch_size*per[1]/sum(per))+1, width, height, max_len)
    legacy_gen = generator_with_legacy_data(int(batch_size*per[2]/sum(per))+1, width, height, max_len)
    mnist_gen = generator_with_emnist(int(batch_size*per[3]/sum(per))+1, width, height, max_len)
    actual_batchsize = int(batch_size*per[0]/sum(per))+ int(batch_size*per[1]/sum(per))+ \
                       int(batch_size*per[2]/sum(per)) + int(batch_size*per[3]/sum(per)) + 4
    gen_list = [cap_gen, folder_gen, legacy_gen, mnist_gen]
    while True:
        the_input = []
        the_label = []
        input_len = []
        label_len = []
        for gen_item in gen_list:
            x, y = next(gen_item)
            the_input.append(x['the_input'])
            the_label.append(x['the_labels'])
            input_len.append(x['input_length'])
            label_len.append(x['label_length'])
        the_input = np.concatenate(the_input, axis=0)
        the_label = np.concatenate(the_label, axis=0)
        input_len = np.concatenate(input_len, axis=0)
        label_len = np.concatenate(label_len, axis=0)
        inputs = {'the_input': the_input,
                  'the_labels': the_label.astype(np.float32),
                  'input_length': input_len.astype(np.int64),
                  'label_length': label_len.astype(np.int64),
                  }
        outputs = {'ctc': np.zeros([actual_batchsize])}
        yield inputs, outputs


def evaluation_generator(batch_size, width=256, height=64, max_len=6):
    from data_utils_pack.data_utils import inference_img_prepare, resize_width
    test_img_list = []
    test_img_folder = 'D:\herschel\changrong\sequential_ocr\\test_img\\real\digit_coarse'
    test_img_list += get_file_list(test_img_folder)
    test_img_folder = 'D:\herschel\changrong\sequential_ocr\\test_img\\real\digit_fine'
    test_img_list += get_file_list(test_img_folder)
    random.shuffle(test_img_list)
    img_data = []
    label = []
    for item in test_img_list:
        img_item = inference_img_prepare(item, height)
        img_item = resize_width(np.squeeze(img_item), width)
        img_item = np.expand_dims(img_item, -1)
        label_item = os.path.basename(item).split('_')[0]
        img_data.append(img_item)
        label.append(label_item)
        # plt.imshow(np.squeeze(img_item))
        # plt.title(label_item)
        # plt.show()
    while True:
        # 标签
        x_batch = []
        labels = np.ones([batch_size, max_len]) * 10000
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        for i in range(batch_size):
            data_item = img_data.pop()
            label_item = label.pop()
            img_data.insert(0, data_item)
            label.insert(0, label_item)
            x_batch.append(data_item)
            input_length[i] = width // 8
            labels[i, :len(label_item)] = [int(k) for k in label_item]
            label_length[i] = len(label_item)
        x_batch = np.stack(x_batch, axis=0)
        inputs = {'the_input': x_batch,
                  'the_labels': labels.astype(np.float32),
                  'input_length': input_length.astype(np.int64),
                  'label_length': label_length.astype(np.int64),
                  }
        outputs = {'ctc': np.zeros([batch_size])}
        yield inputs, outputs


def legacy_test():
    import uuid
    import matplotlib.pyplot as plt
    legacy_npy_dir = 'D:\herschel\changrong\data\digit\old'
    file_name = ['train_data.npy', 'labels.npy']
    data = np.load(os.path.join(legacy_npy_dir, file_name[0]))
    data = data.reshape((-1, 32, 32))
    label = np.load(os.path.join(legacy_npy_dir, file_name[1]))
    label = np.argmax(label, axis=-1).astype(np.uint8)
    saved_folder = 'H:\\test\\legacy'
    os.makedirs(saved_folder, exist_ok=True)
    count_limit = None
    count = 0
    while 1:
        for x_item, y_item in zip(data, label):
            x_item = np.squeeze(x_item) * 255
            x_item = np.uint8(x_item)
            y_item = y_item[y_item < 1000].astype(np.uint8)
            y_item = ''.join(str(x) for x in y_item)
            saved_name = '{}_{}.png'.format(count, y_item)
            saved_name = os.path.join(saved_folder, saved_name)
            cv2.imwrite(saved_name, x_item)
            count += 1
        if count_limit is not None:
            if count > count_limit:
                break
    return


if __name__ == '__main__':
    import uuid
    import matplotlib.pyplot as plt
    gen = merge_generator(128, 128, 48)
    saved_folder = 'H:\\test\\digit'
    os.makedirs(saved_folder, exist_ok=True)
    count_limit = 1000
    count = 0
    while 1:
        x, y = next(gen)
        for x_item, y_item in zip(x['the_input'], x['the_labels']):
            x_item = np.squeeze(x_item) * 255
            x_item = np.uint8(x_item)
            y_item = y_item[y_item < 1000].astype(np.uint8)
            y_item = ''.join(str(x) for x in y_item)
            saved_name = '{}_{}.png'.format(y_item, str(uuid.uuid4()))
            saved_name = os.path.join(saved_folder, saved_name)
            cv2.imwrite(saved_name, x_item)
            count += 1
        if count_limit is not None:
            if count > count_limit:
                break
            # plt.imshow(x_item)
            # plt.title(y_item)
            # plt.show()
    # legacy_test()