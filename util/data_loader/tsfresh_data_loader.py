import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TSfreshDataContainter:
    def __init__(self, tsfresh_data_path, series_nums_train, series_nums_test, series_nums_val, target,
                 drop_cols_ts_fresh):
        features_filtered_direct_df = pd.read_csv(tsfresh_data_path)
        # handle ts-fresh data
        SERIES_COL = 'series_num'
        # drop_cols_ts_fresh = [FILE_ID, EVENT_ID, CURRENT_TARGET, SERIES_COL]
        # SERIES_TRAIN = lstm_data.file_id_event_id_dict.get('y_train').get(SERIES_COL)
        #         # SERIES_TEST = lstm_data.file_id_event_id_dict.get('y_test').get(SERIES_COL)
        #         # SERIES_VAL = lstm_data.file_id_event_id_dict.get('y_val').get(SERIES_COL)

        X = features_filtered_direct_df.drop(drop_cols_ts_fresh, axis=1)

        # scale features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_scaled = scaler.fit_transform(X.values)
        X = pd.DataFrame(x_scaled, columns=X.columns)
        X[SERIES_COL] = features_filtered_direct_df[SERIES_COL]

        X_aggr = X.loc[X[SERIES_COL].isin(series_nums_train)].drop(SERIES_COL, axis=1)
        x_test = X.loc[X[SERIES_COL].isin(series_nums_test)].drop(SERIES_COL, axis=1)
        x_val = X.loc[X[SERIES_COL].isin(series_nums_val)].drop(SERIES_COL, axis=1)

        y = features_filtered_direct_df
        y_aggr = y.loc[y[SERIES_COL].isin(series_nums_train)][target]
        y_test = y.loc[y[SERIES_COL].isin(series_nums_test)][target]
        y_val = y.loc[y[SERIES_COL].isin(series_nums_val)][target]

        self.x_train = X_aggr
        self.x_test = x_test
        self.x_val = x_val

        self.y_train = y_aggr
        self.y_test = y_test
        self.y_val = y_val
