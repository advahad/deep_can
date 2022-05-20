from util.data_loader.data_loader_shrp2 import LSTMDataContainer
from util.data_loader.tsfresh_data_loader import TSfreshDataContainter



class HybridDataContainer:
    def __init__(self, data_path, tsfresh_data_path, current_target, seq_len, remove_cols_lstm, drop_cols_ts_fresh,
                 series_col):
        lstm_data = LSTMDataContainer(data_path, current_target, seq_len, remove_cols_lstm)

        # get relevant series numbers from lstm data container
        series_train = lstm_data.file_id_event_id_dict.get('y_train').get(series_col)
        series_test = lstm_data.file_id_event_id_dict.get('y_test').get(series_col)
        series_val = lstm_data.file_id_event_id_dict.get('y_val').get(series_col)

        tsfresh_data = TSfreshDataContainter(tsfresh_data_path, series_train, series_test, series_val, current_target,
                                             drop_cols_ts_fresh)

        self.lstm_data = lstm_data
        self.tsfresh_data = tsfresh_data
