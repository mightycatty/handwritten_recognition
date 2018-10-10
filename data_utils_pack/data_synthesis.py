# coding: utf-8
"""
利用单字数字生成数字序列
"""
import os
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
import numpy as np
import random
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
try:
    from wheezy.captcha import image as wheezy_captcha
except ImportError:
    wheezy_captcha = None
from data_utils_pack.data_utils import resize_gray_with_padding

import cv2
table  =  []
for  i  in  range( 256 ):
    table.append( i * 1.97 )


class SynthesisConfig(object):
    noise_curve_per = 1
    noise_dots_per = 1


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destionation.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


class SequentialDigit(_Captcha):
    def __init__(self, width=160, height=60, font_sizes=None):
        self._width = width
        self._height = height
        self._font_sizes = font_sizes or int(height*0.75)

    @staticmethod
    def create_noise_curve(image, color):
        if not random.randint(0, SynthesisConfig.noise_curve_per):
            w, h = image.size
            x1 = random.randint(0, int(w / 5))
            x2 = random.randint(w - int(w / 5), w)
            y1 = random.randint(int(h / 5), h - int(h / 5))
            y2 = random.randint(y1, h - int(h / 5))
            points = [x1, y1, x2, y2]
            end = random.randint(160, 200)
            start = random.randint(0, 20)
            Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def create_noise_dots(image, color, width=3, number=30):
        if not random.randint(0, SynthesisConfig.noise_dots_per):
            draw = Draw(image)
            w, h = image.size
            while number:
                x1 = random.randint(0, w)
                y1 = random.randint(0, h)
                draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
                number -= 1
        return image

    def create_captcha_image(self, images):
        image = Image.new('L', (self._width, self._height), (0, ))
        images = [image_distortion_test(item, self._font_sizes) for item in images]
        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(images))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_image(self, image_list):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        # background = random_color(238, 255)
        # color = random_color(0, 200, random.randint(220, 255))
        im = self.create_captcha_image(image_list)
        im = self.create_noise_dots(im, (255,))
        im = self.create_noise_curve(im, (255,))
        im = im.filter(ImageFilter.SMOOTH)
        return im


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)


def image_distortion(im, resize_shape):
    # rotate
    im = im.crop(im.getbbox())
    im = im.resize((resize_shape, resize_shape))
    im = im.rotate(random.uniform(-15, 15), Image.BILINEAR, expand=1)
    w, h = im.size
    # im.show()
    # warp
    dx = w * random.uniform(0.1, 0.3)
    dy = h * random.uniform(0.2, 0.3)
    x1 = int(random.uniform(-dx, dx))
    y1 = int(random.uniform(-dy, dy))
    x2 = int(random.uniform(-dx, dx))
    y2 = int(random.uniform(-dy, dy))
    w2 = w + abs(x1) + abs(x2)
    h2 = h + abs(y1) + abs(y2)
    data = (
        x1, y1,
        -x1, h2 - y2,
        w2 + x2, h2 + y2,
        w2 - x2, -y1,
    )
    im = im.resize((w2, h2))
    im = im.transform((w, h), Image.QUAD, data)
    # img_data = np.array(im)
    # plt.imshow(img_data)
    # plt.show()
    # im.show()
    return im


def image_distortion_test(im, resize_shape):
    # rotate
    resize_shape = int(resize_shape * random.uniform(0.75, 1.25))
    im = im.crop(im.getbbox())
    im = resize_gray_with_padding(im, resize_shape)
    if random.randint(0, 1):
        im = im.rotate(random.uniform(-5, 5), Image.BILINEAR, expand=1)
    if random.randint(0, 1):
        w, h = im.size
        # warp
        dx = w * random.uniform(0.1, 0.3)
        dy = h * random.uniform(0.2, 0.3)
        x1 = int(random.uniform(-dx, dx))
        y1 = int(random.uniform(-dy, dy))
        x2 = int(random.uniform(-dx, dx))
        y2 = int(random.uniform(-dy, dy))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        data = (
            x1, y1,
            -x1, h2 - y2,
            w2 + x2, h2 + y2,
            w2 - x2, -y1,
        )
        im = im.resize((w2, h2))
        im = im.transform((w, h), Image.QUAD, data)
    # # erode or dilate
    # if not random.randint(0, 10):
    #     img_data = np.array(im)
    #     kernel = np.ones((3, 3), np.uint8)
    #     img_data = cv2.dilate(img_data, kernel, iterations=random.randint(0, 1))
    #     img_data = cv2.erode(img_data, kernel, iterations=random.randint(0, 1))
    #     im = Image.fromarray(img_data)
    return im


if __name__ == '__main__':
    resource_folder = 'D:\herschel\changrong\data\digit\digital\split\split_red'
    from data_utils_pack.data_utils import get_file_list
    img_dir_list = get_file_list(resource_folder)
    seq = SequentialDigit(256, 48, 30)
    import random
    while 1:
        select_index = np.random.randint(0, len(img_dir_list), (random.randint(1, 10)))
        img_list_s = [img_dir_list[item] for item in select_index]
        img_data_list = [Image.open(item) for item in img_list_s]
        img = seq.generate_image(img_data_list)
        img.show()