"""
building dense model for sequential digital number recognition
usage sample:
    input_shape=(32, 280, 1)
    class_num=11
    keras_model = dense_cnn_model(input_shape, class_num)
"""
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, Flatten, Lambda
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
import keras.backend as K


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if (dropout_rate):
        x = Dropout(dropout_rate)(x)

    if (pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif (pooltype == 1):
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif (pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter


def dense_cnn_fn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)
    return y_pred


def dense_cnn_model(input_shape=(32, 280, 1), class_num=11):
    input = Input(shape=input_shape, name='the_input')
    output = dense_cnn_fn(input, class_num)
    dense_model = Model(input, output)
    return dense_model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def compile_model(img_h, nclass):
    basemodel = dense_cnn_model((img_h, None, 1), nclass)
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([basemodel.output, labels, input_length, label_length])

    model = Model(inputs=[basemodel.input, labels, input_length, label_length], outputs=loss_out)
    # from keras.utils import multi_gpu_model
    # model = multi_gpu_model(model, 2)
    from keras.optimizers import SGD
    # opt = SGD(lr=1e-5, momentum=0.9)
    opt = 'adam'
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt, metrics=['accuracy'])
    return basemodel, model