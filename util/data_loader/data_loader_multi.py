import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# TODO:change to parameters
def calc_test_size(seq_len, data):
    # calc percentage
    test_percentage = 0.2
    total_amount = data.shape[0]
    train_amount = test_percentage * total_amount
    reminder = train_amount % seq_len
    if reminder != 0:
        train_amount = train_amount + reminder
        test_amount = total_amount - train_amount
        test_percentage = 1 - (test_amount / total_amount)
    return test_percentage


class BaseDataContainer:
    def __init__(self, csv_path, target_columns, seq_len, remove_cols=None, is_LSTM=True, should_scale=True,
                 should_reshape=True):
        data = pd.read_csv(csv_path)
        # data = data[data['aug.road_type'].isnull() == False]
        original_data = data
        num_of_taregets = len(target_columns)

        # drop unclassified class
        # print(data.isnull().any())
        # filling missing values
        # TODO: complete data before loading!!!!! european data only
        for col in data.columns:
            data[col].interpolate(method='linear', inplace=True)
        # TODO: remove 'sid', 'fid_x' from european data, to european data to work should uncomment following lines
        # if is_LSTM:
        #     data = data.drop(['series_num', 'sid', 'fid_x'], axis=1)
        # else:uncl
        #     data = data.drop(['sid', 'fid_x'], axis=1)
        if remove_cols:
            data = data.drop(remove_cols, axis=1)
        # if is_LSTM:
        #     data = data.drop(['series_num'], axis=1)
        # create a dict for every target
        targets_dict = {}
        for target in target_columns:
            targets_dict[target] = data[target].unique()


        features_size = data.shape[1] - num_of_taregets
        x = data.drop(target_columns, axis=1)
        if should_scale:
            # normalize features
            scaler = MinMaxScaler(feature_range=(0, 1))
            x_scaled = scaler.fit_transform(x.values)
            x = pd.DataFrame(x_scaled)

        if should_reshape:
            y = data[target_columns]
            x = x.values.reshape((int(x.shape[0] / seq_len), seq_len, features_size))
            y = y.values.reshape((int(y.shape[0] / seq_len), seq_len, num_of_taregets))
            y = y[:, 0, :]  # get only the first column
        else:
            # TODO: check if OK
            y = data.groupby(['series_num']).first()[target_columns]

        # split train test
        test_size = calc_test_size(seq_len, data)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
                                                            # stratify=y)

        # encode to numbers
        # TODO: save classes for evaluation phase
        le_d = {}
        # TODO check if working for one label
        y_train_encoded = pd.DataFrame([])
        y_test_encoded = pd.DataFrame([])
        for idx, target in enumerate(target_columns):
            le = preprocessing.LabelEncoder()
            le_d[target] = le
            le.fit(y[:, idx])
            y_train_encoded[target] = le.transform(y_train[:, idx])
            y_test_encoded[target] = le.transform(y_test[:, idx])
            # y_train_encoded = np.append(y_train_encoded, le.transform(y_train[:, idx]), axis=1)
            # y_test_encoded = np.append(y_test_encoded, le.transform(y_test[:, idx]), axis=1)
            # np.save('classes.npy', le.classes_)

        self.original_data = original_data
        self.x = x
        self.y = y
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_encoded = y_train_encoded
        self.y_test_encoded = y_test_encoded
        # TODO: should remove and change usage in code, leave only dict
        self.le_dict = le_d
        self.targets_dict = targets_dict
        self.seq_len = seq_len
        self.features_size = features_size


class BaselineDataContainer(BaseDataContainer):
    def __init__(self, csv_path, target_column):
        super().__init__(csv_path, target_column, is_LSTM=False, should_scale=False, should_reshape=False)
        # preprocess_data(self, csv_path, is_LSTM=False, should_scale=False, should_reshape=False)


class LSTMDataContainer(BaseDataContainer):
    def __init__(self, csv_path, target_columns, seq_len, remove_cols):
        super().__init__(csv_path, target_columns, seq_len, remove_cols, is_LSTM=True, should_scale=True,
                         should_reshape=True)
