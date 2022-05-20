import tensorflow as tf
import os


def get_test_info():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)

    all_devices_dict = tf.config.experimental.list_physical_devices()
    gpus = tf.config.experimental.list_physical_devices('GPU')

    print('\nAll available devices are: {}'.format(all_devices_dict))
    print('GPU devieces are: {}'.format(gpus))

    if tf.test.gpu_device_name():
        print('\nDefault GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print('No GPU device')