# -*- coding:utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from build_model import compile_model
from data_utils_pack.generators_digit import generator_with_cap, generator_with_folder_img, merge_generator, evaluation_generator
from config.config_digit import ModelConfig, TrainingConfig


def get_session(gpu_fraction=1.0):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if __name__ == '__main__':
    # K.set_session(get_session())
    gpu = 1
    basemodel, model = compile_model(ModelConfig.img_h, 10+1)
    modelPath = TrainingConfig.weight_dir
    if os.path.exists(modelPath):
        print("Loading model weights...")
        model.load_weights(modelPath, skip_mismatch=True)
        print('done!')
        # init_epoch = int(modelPath.split('-')[-2])
    else:
        init_epoch = 0
    init_epoch = 0
    train_loader = merge_generator(ModelConfig.batch_size, ModelConfig.img_w, ModelConfig.img_h)
    test_loader = evaluation_generator(ModelConfig.batch_size, ModelConfig.img_w, ModelConfig.img_h)
    checkpoint = ModelCheckpoint(filepath='./models/digit_model/digit-{loss:.2f}-{val_acc:.2f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)
    # lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    # learning_rate = np.array([lr_schedule(i) for i in range(10)])
    # changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs/digit', write_graph=True)
    # multi gpus
    # model = multi_gpu_model(model, 2)
    print('-----------Start training-----------')
    print (model.summary())
    while True:
        model.fit_generator(train_loader,
                            steps_per_epoch=TrainingConfig.steps_per_epoch,
                            epochs=1000,
                            initial_epoch=init_epoch,
                            validation_data=test_loader,
                            validation_steps=TrainingConfig.validation_steps,
                            callbacks=[earlystop, tensorboard, checkpoint])
        # # export model
        # if gpu > 1:
        #     model_se = model.get_layer('model_2')
        #     # print (model_se.summary())
        # else:
        #     model_se = model
        # model.save('./models/train_model_{}'.format(time.ctime()).replace(' ', '_').replace(':', '-'))
        # weights = model_se.get_weights()
        # basemodel.set_weights(weights)
        # basemodel.save('./models/base_model_{}'.format(time.ctime()).replace(' ', '-').replace(':', '-'))
        # print('saved model:{}'.format(time.ctime()))
