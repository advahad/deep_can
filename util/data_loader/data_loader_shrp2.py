import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
from sklearn.utils import class_weight
from util import pickle_util


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



def calc_class_weights(y_train):
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)
    class_labels = np.unique(y_train)
    class_weights_dict = {}
    for idx, label in enumerate(class_labels):
        class_weights_dict[label] = class_weights[idx]
    return class_weights_dict

def reshape_y(y, seq_len):
    y = y.values.reshape((int(y.shape[0] / seq_len), seq_len))[:, 0]
    return y

def reshape_x(x, seq_len):
    x = x.values.reshape((int(x.shape[0] / seq_len), seq_len, x.shape[1]))
    # x = x.values.reshape((int(x.shape[0] / seq_len), x.shape[1], seq_len))
    return x

def split_data_by_group(data_df, target, seq_len, should_reshape=True, seed=5):
    FILE_ID = 'file_id'
    EVENT_ID = 'event_id'
    SERIES_NUM = 'series_num'
    file_id_target_df = data_df[[target, FILE_ID]]
    unique_file_ids, indices = np.unique(file_id_target_df['file_id'].values, return_index=True)
    set_for_spliiting = file_id_target_df.iloc[indices]

    test_set_size = 0.1
    val_set_size = 0.2
    try:
        train_val_set_file_ids, test_set_file_ids, train_val_set_target_col, test_set_target_col = train_test_split(
            set_for_spliiting, set_for_spliiting[target], stratify=set_for_spliiting[target],
            test_size=test_set_size, shuffle=True, random_state=seed)
    except Exception as e:
        print(e)
        train_val_set_file_ids, test_set_file_ids, train_val_set_target_col, test_set_target_col = train_test_split(
            set_for_spliiting, set_for_spliiting[target], test_size=test_set_size, shuffle=True, random_state=seed)

    # create validation set from train set
    try:
        print("split val set from train set")
        train_set_file_ids, val_set_file_ids, train_set_target_col, val_set_target_col = \
            train_test_split(train_val_set_file_ids, train_val_set_target_col, stratify=train_val_set_target_col,
                             test_size=val_set_size, shuffle=True, random_state=seed)
    except Exception as e:
        print(e)
        train_set_file_ids, val_set_file_ids, train_set_target_col, val_set_target_col = \
            train_test_split(train_val_set_file_ids, train_val_set_target_col,
                             test_size=val_set_size, shuffle=True, random_state=seed)

    # train test data
    train_set = data_df[data_df[FILE_ID].isin(train_set_file_ids[FILE_ID])]
    test_set = data_df[data_df[FILE_ID].isin(test_set_file_ids[FILE_ID])]
    val_set = data_df[data_df[FILE_ID].isin(val_set_file_ids[FILE_ID])]


    # validate no leakage
    train_set_file_ids_unique = set(train_set[FILE_ID])
    test_set_file_ids_unique = set(test_set[FILE_ID])
    val_set_file_ids_unique = set(val_set[FILE_ID])

    common_file_ids_train_test = train_set_file_ids_unique.intersection(test_set_file_ids_unique)
    common_file_ids_train_val = train_set_file_ids_unique.intersection(val_set_file_ids_unique)

    len_common_train_test = len(common_file_ids_train_test)
    len_common_train_val = len(common_file_ids_train_val)
    print("splitted train test sets common file_ids size: {}".format(len_common_train_test))
    print("splitted train val sets common file_ids size: {}".format(len_common_train_val))

    # split to X and y for each set
    drop_cols = [target, FILE_ID, EVENT_ID, SERIES_NUM]
    X_train = train_set.drop(drop_cols, axis=1)
    X_test = test_set.drop(drop_cols, axis=1)
    X_val = val_set.drop(drop_cols, axis=1)

    y_train = train_set[target]
    y_test = test_set[target]
    y_val = val_set[target]

    y_train_file_id = train_set[FILE_ID]
    y_test_file_id = test_set[FILE_ID]
    y_val_file_id = val_set[FILE_ID]

    y_train_event_id = train_set[EVENT_ID]
    y_test_event_id = test_set[EVENT_ID]
    y_val_event_id = val_set[EVENT_ID]


    y_train_series_num = train_set[SERIES_NUM]
    y_test_series_num = test_set[SERIES_NUM]
    y_val_series_num = val_set[SERIES_NUM]




    # reshape the data
    if should_reshape:
        X_train = reshape_x(X_train, seq_len)
        X_test = reshape_x(X_test, seq_len)
        X_val = reshape_x(X_val, seq_len)

        y_train = reshape_y(y_train, seq_len)
        y_test = reshape_y(y_test, seq_len)
        y_val = reshape_y(y_val, seq_len)


        # reshape file_id event_id
        y_train_file_id = reshape_y(y_train_file_id, seq_len)
        y_test_file_id = reshape_y(y_test_file_id, seq_len)
        y_val_file_id = reshape_y(y_val_file_id, seq_len)

        y_train_event_id = reshape_y(y_train_event_id, seq_len)
        y_test_event_id = reshape_y(y_test_event_id, seq_len)
        y_val_event_id = reshape_y(y_val_event_id, seq_len)


        y_train_series_num = reshape_y(y_train_series_num, seq_len)
        y_test_series_num = reshape_y(y_test_series_num, seq_len)
        y_val_series_num = reshape_y(y_val_series_num, seq_len)

    file_id_event_id_dict = {
        'y_train': {FILE_ID: y_train_file_id, EVENT_ID: y_train_event_id, SERIES_NUM: y_train_series_num},
        'y_test': {FILE_ID: y_test_file_id, EVENT_ID: y_test_event_id, SERIES_NUM: y_test_series_num},
        'y_val': {FILE_ID: y_val_file_id, EVENT_ID: y_val_event_id, SERIES_NUM: y_val_series_num}
    }
    # return return_sets
    return X_train, X_test, X_val, y_train, y_test, y_val, file_id_event_id_dict

def split_data_random(data_df, target_col, seq_len, should_reshape=True):
    data_df = data_df.drop(['file_id'], axis=1)

    test_size = calc_test_size(seq_len, data_df)

    x = data_df.drop([target_col], axis=1)
    y = data_df[[target_col]]

    if should_reshape:
        x = x.values.reshape((int(x.shape[0] / seq_len), seq_len, x.shape[1]))
        y = y.values.reshape((int(y.shape[0] / seq_len), seq_len))[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=5, stratify=y)



    return X_train, X_test, y_train, y_test


class BaseDataContainer:
    def __init__(self, csv_path, target_column, seq_len, remove_cols=None, is_LSTM=True, should_scale=True,
                 should_reshape=True):
        GROUP_BY_FIELD = 'file_id'
        EVENT_ID = 'event_id'
        SERIES_NUM = 'series_num'
        data = pd.read_csv(csv_path)
        # remove rowa
        # groups = data.groupby(['series_num', target_column])
        # for name, df in groups:
        #     redundent = (df.shape[0])%seq_len
        #     if redundent > 0:
        #         print(name)
        # remove minor class from data
        # minor_class = data[target_column].value_counts(ascending=True).index[5]
        # # minor_class_1 = data[target_column].value_counts(ascending=True).index[1]
        # # minor_class_2 = data[target_column].value_counts(ascending=True).index[2]
        # print('Removing class {}'.format(minor_class))
        # # print('Removing class {}'.format(minor_class_1))
        # # print('Removing class {}'.format(minor_class_2))
        #
        # data = data[data[target_column] != minor_class]
        # # data = data[data[target_column] != minor_class_1]
        # # data = data[data[target_column] != minor_class_2]
        # data.reset_index(inplace=True)


        original_data = data

        # mini pre processing- getting only data for spesific target
        # for col in data.columns:
        #     data[col].interpolate(method='linear', inplace=True)
        if remove_cols:
            data = data.drop(remove_cols, axis=1)


        # save important meta data
        classes = data[target_column].unique()
        # rescaling X data
        X = data.drop(target_column, axis=1)
        if should_scale:
            # normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            X = pd.DataFrame(X_scaled, columns=data.columns.drop(target_column))
            X[GROUP_BY_FIELD] = original_data[GROUP_BY_FIELD]
            X[target_column] = original_data[target_column]
            X[EVENT_ID] = original_data[EVENT_ID]
            X[SERIES_NUM] = original_data[SERIES_NUM]
            data = X
            print(data.describe())
        y = data[target_column]

        # split train test
        test_size = calc_test_size(seq_len, data)

        X_train, X_test, X_val, y_train, y_test, y_val, file_id_event_id_dict = split_data_by_group(data, target_column, seq_len,
                                                                                                    should_reshape=True, seed=31)

        # X_train, X_test, y_train, y_test = split_data_with_leakage(data, target_column, seq_len)

        features_size = X_train.shape[2]

        # TODO: save classes for evaluation phase
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)

        y_val_encoded = le.transform(y_val)
        np.save('classes.npy', le.classes_)


        # y_train_encoded = pd.get_dummies(y_train)
        # y_test_encoded = pd.get_dummies(y_test)
        # y_val_encoded = pd.get_dummies(y_val)

        # class_weights = calc_class_weights(y_train_encoded)

        class_weights = {}
        self.original_data = original_data
        self.x = X
        self.y = y
        self.x_train = X_train
        self.x_test = X_test
        self.x_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.y_train_encoded = y_train_encoded
        self.y_test_encoded = y_test_encoded
        self.y_val_encoded = y_val_encoded
        self.classes = classes
        self.seq_len = seq_len
        self.features_size = features_size
        self.class_weights_encoded = class_weights
        self.file_id_event_id_dict = file_id_event_id_dict
        self.le = le


class BaselineDataContainer(BaseDataContainer):
    def __init__(self, csv_path, target_column):
        super().__init__(csv_path, target_column, is_LSTM=False, should_scale=False, should_reshape=False)
        # preprocess_data(self, csv_path, is_LSTM=False, should_scale=False, should_reshape=False)


class LSTMDataContainer(BaseDataContainer):
    def __init__(self, csv_path, target_column, seq_len, remove_cols):
        super().__init__(csv_path, target_column, seq_len, remove_cols, is_LSTM=True, should_scale=True,
                         should_reshape=True)


class KaggleDataContainer(BaseDataContainer):
    def __init__(self, csv_path, target_column, seq_len, remove_cols):
        super().__init__(csv_path, target_column, seq_len, remove_cols, is_LSTM=True, should_scale=True,
                         should_reshape=True)
