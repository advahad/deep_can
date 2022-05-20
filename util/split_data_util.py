import numpy as np
from sklearn.model_selection import train_test_split


def split_data_no_leakage(data_df, target_name, should_reshape=True, seq_len=None, val_set=True, seed=5, drop_cols=None):
    LEAKAGE_FIELD = 'file_id'
    file_id_target_df = data_df[[target_name, LEAKAGE_FIELD]]
    unique_file_ids, indices = np.unique(file_id_target_df['file_id'].values, return_index=True)
    set_for_spliiting = file_id_target_df.iloc[indices]

    test_set_size = 0.1
    val_set_size = 0.2

    try:
        train_val_set_file_ids, test_set_file_ids, train_val_set_target_col, test_set_target_col = train_test_split(
            set_for_spliiting, set_for_spliiting[target_name], stratify=set_for_spliiting[target_name],
            test_size=test_set_size, shuffle=True, random_state=seed)
    except Exception as e:
        print(e)
        train_val_set_file_ids, test_set_file_ids, train_val_set_target_col, test_set_target_col = train_test_split(
            set_for_spliiting, set_for_spliiting[target_name], test_size=test_set_size, shuffle=True, random_state=seed)

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
    train_set = data_df[data_df[LEAKAGE_FIELD].isin(train_set_file_ids[LEAKAGE_FIELD])]
    test_set = data_df[data_df[LEAKAGE_FIELD].isin(test_set_file_ids[LEAKAGE_FIELD])]
    val_set = data_df[data_df[LEAKAGE_FIELD].isin(val_set_file_ids[LEAKAGE_FIELD])]

    # validate no leakage
    train_set_file_ids_unique = set(train_set[LEAKAGE_FIELD])
    test_set_file_ids_unique = set(test_set[LEAKAGE_FIELD])
    val_set_file_ids_unique = set(val_set[LEAKAGE_FIELD])

    common_file_ids_train_test = train_set_file_ids_unique.intersection(test_set_file_ids_unique)
    common_file_ids_train_val = train_set_file_ids_unique.intersection(val_set_file_ids_unique)

    len_common_train_test = len(common_file_ids_train_test)
    len_common_train_val = len(common_file_ids_train_val)
    print("splitted train test sets common file_ids size: {}".format(len_common_train_test))
    print("splitted train val sets common file_ids size: {}".format(len_common_train_val))

    # split to X and y for each set
    if drop_cols == None:
        drop_cols = [target_name, LEAKAGE_FIELD]
    X_train = train_set.drop(drop_cols, axis=1)
    X_test = test_set.drop(drop_cols, axis=1)
    X_val = val_set.drop(drop_cols, axis=1)

    y_train = train_set[target_name]
    y_test = test_set[target_name]
    y_val = val_set[target_name]

    # reshape the data
    if should_reshape:
        X_train = X_train.values.reshape((int(X_train.shape[0] / seq_len), seq_len, X_train.shape[1]))
        X_test = X_test.values.reshape((int(X_test.shape[0] / seq_len), seq_len, X_test.shape[1]))
        X_val = X_val.values.reshape((int(X_val.shape[0] / seq_len), seq_len, X_val.shape[1]))

        y_train = y_train.values.reshape((int(y_train.shape[0] / seq_len), seq_len))[:, 0]
        y_test = y_test.values.reshape((int(y_test.shape[0] / seq_len), seq_len))[:, 0]
        y_val = y_val.values.reshape((int(y_val.shape[0] / seq_len), seq_len))[:, 0]

    # return return_sets
    return X_train, X_test, X_val, y_train, y_test, y_val