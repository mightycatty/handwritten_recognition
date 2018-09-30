class ModelConfig(object):
    img_h = 48
    img_w = 128
    batch_size = 128


class TrainingConfig(object):
    weight_dir = 'D:\herschel\changrong\sequential_ocr\models\digit_model\\base-0.812-0.927.h5'
    steps_per_epoch = 1000
    epochs = 1000
    initial_epoch = 0
    validation_steps = 2