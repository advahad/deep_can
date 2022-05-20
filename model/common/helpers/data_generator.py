import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf


def batch_generation(samples_path, series_files_names, non_features_cols, remove_features, keep_features, seq_len, le,
                     should_reshape=True):
    'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
    samples_ctr = 0
    samples_dfs_dict = {}
    batch_df = pd.DataFrame([])

    for file_name in series_files_names:
        sample_df = pd.read_csv(samples_path + file_name)
        sample_df = sample_df.drop(remove_features, axis=1)
        samples_dfs_dict[samples_ctr] = sample_df
        samples_ctr += 1
    all_batch_samples = list(samples_dfs_dict.values())
    batch_df = batch_df.append(all_batch_samples)

    X = batch_df.drop(non_features_cols, axis=1)
    if keep_features != 'all':
        X = batch_df[keep_features]
    # X = X.drop(remove_features, axis=1)

    # if np.shape(X)[1] > 18:
    #     print('wait')
    # scale X
    # X = scaler.transform(X)

    # adjustments to architecture
    if should_reshape:
        X = reshape_x(X, seq_len)


    # if not is_auto_encoder:
    target_col = 'aug.road_type'
    y = batch_df[target_col]
    if should_reshape:
        y = reshape_y(y, seq_len)
    # categorical to numaric
    y_encoded = le.transform(y)
    return X, y_encoded
    # else:
    #     return X, X

def reshape_y(y, seq_len):
    # y = y.values.reshape((int(y.shape[0] / seq_len), seq_len))[:, 0]
    # return y
    return np.reshape(y.values, (int(y.shape[0] / seq_len), seq_len))[:, 0]


def reshape_x(x, seq_len):
    # x = x.values.reshape((int(x.shape[0] / seq_len), seq_len, x.shape[1]))
    # return x
    if type(x) == pd.core.frame.DataFrame:
        x = x.values.reshape((int(x.shape[0] / seq_len), seq_len, x.shape[1]))
    else:
        x - np.reshape(x, (int(x.shape[0] / seq_len), seq_len, x.shape[1]))
    return x


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, samples_path, non_features_cols, remove_features, label_encoder, batch_size=32, seq_len=50,
                 shuffle=True, is_ae=False):
        self.list_IDs = list_IDs
        self.samples_path = samples_path
        self.non_features_cols = non_features_cols
        self.remove_features = remove_features
        self.le = label_encoder
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.on_epoch_end()
        self.is_ae = is_ae

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = batch_generation(self.samples_path, list_IDs_temp, self.non_features_cols, self.remove_features, self.seq_len, self.le)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

