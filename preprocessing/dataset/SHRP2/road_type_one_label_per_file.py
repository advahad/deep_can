import os
import pandas as pd

BASELINES_PATH_GPS = '../data_SHRP2/augmented_timeseries/'
RESULT_PATH = '../data_SHRP2/augmented_timeseries_one_roadtype_per_file/'


def create_one_class_per_baseline():
    ts_files_names = os.listdir(BASELINES_PATH_GPS)
    target_col = 'aug.road_type'
    for file in ts_files_names:
        df = pd.read_csv(BASELINES_PATH_GPS + file)
        value_c = df[target_col].value_counts()
        most_common_road_type = df[target_col].value_counts().index[0]
        df[target_col] = most_common_road_type
        df.to_csv(RESULT_PATH + file, index=False)
