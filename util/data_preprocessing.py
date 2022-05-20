from util.data_loader.data_loader_shrp2 import LSTMDataContainer
import numpy as np


# first order derivative for gyro features only
# Use the first order difference of the following features
# Absolute Orientation features dont make sense to predict surface
def first_order_derivative(gyro_features, feat_cols, feat_array):
    for dc in gyro_features:
        iia = feat_cols.index(dc)
        np_arr = feat_array[:, :, iia]
        roll_arr = np.copy(np_arr)
        roll_arr[:, 1:] = roll_arr[:, :-1]
        np_arr = np_arr - roll_arr
        feat_array[:, :, iia] = np_arr
        return feat_array


# Normalize non gyro features
# Normalize each 128-pt sample to ensure there is no group related information left in the samples
def norm_each_series(feat_cols, gyro_features, feat_array):
    norm_cols = [x for x in feat_cols if x not in gyro_features]
    num_meas = np.shape(feat_array)[1]
    # norm_cols.remove('file_id')
    for norm in norm_cols:
        iia = feat_cols.index(norm)
        np_arr = feat_array[:, :, iia]
        mean_arr = np.mean(np_arr, 1)
        mean_arr = np.expand_dims(mean_arr, 1)
        mean_arr = np.repeat(mean_arr, num_meas, 1)
        np_arr = np_arr - mean_arr
        feat_array[:, :, iia] = np_arr
        return feat_array

# private func
def absfft(x):
    return np.abs(np.fft.rfft(x))

# Fourier transformation for non gyro features, creates new data set with a subset of the remaining features
def fourier_transform(feat_array):
    feat_fft_array = np.copy(feat_array[:, :, 3:])
    feat_fft_array = np.apply_along_axis(absfft, 1, feat_fft_array)

    return feat_fft_array


# Further normalization across the entire dataset to ensure NN inputs are zero-mean and unit standard deviation
# normalize all features
def normalize_all(feat_array):
    num_sensor = feat_array.shape[2]
    for i in range(num_sensor):
        mean_s = np.mean(feat_array[:, :, i])
        sd_s = np.std(feat_array[:, :, i])
        feat_array[:, :, i] = (feat_array[:, :, i] - mean_s) / sd_s
        return feat_array