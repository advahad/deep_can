# Seed value (can actually be different for each attribution step)

def set_seeds(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    print('setting global seeds')
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value) # tensorflow 2.x
    # tf.set_random_seed(seed_value) # tensorflow 1.x

    # # # 5. Configure a new global `tensorflow` session
    # from tensorflow.keras import backend as K
    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.get_default_graph(), config_classes=session_conf)
    # K.set_session(sess)

    # # 5. Configure a new global `tensorflow` session
    from tensorflow.keras import backend as K

    tf.compat.v1.ConfigProto()
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # K.set_session(sess)
    tf.compat.v1.keras.backend.set_session(sess)