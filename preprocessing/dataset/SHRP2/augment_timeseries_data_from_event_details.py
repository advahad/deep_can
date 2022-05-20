import os
import pandas as pd

BASELINES_PATH_GPS = '../data_SHRP2/augmented_timeseries_one_roadtype_per_file/'
BASELINES_PATH_NO_GPS = '../data_SHRP2/no_gps_timeseries/'
BASELINES_MULTI_PATH = '../data_SHRP2/multi_task_timeseries/'
EVENTS_DETAILS_PATH = '../data_SHRP2/events_details_joint_timeseries.csv'
aug_cols = ['surfaceCondition', 'lighting', 'relationToJunction', 'weather', 'locality']


def augment_ts_from_event_details(ts_dir_path):
    ts_files_names = os.listdir(ts_dir_path)
    events_details_df = pd.read_csv(EVENTS_DETAILS_PATH)
    for name in ts_files_names:
        # load file
        current_file_df = pd.read_csv(ts_dir_path + name)
        # get event record identifier file_id + event id
        splitted = name.split("_", 5)
        file_id = splitted[2]
        event_id = splitted[4].split(".")[0]
        augmentation_row = events_details_df.loc[(events_details_df['fileID'] == int(file_id)) &
                                                 (events_details_df[' eventid'] == int(event_id))]
        for aug_col in aug_cols:
            if augmentation_row.empty:
                aug_value = None
            else:
                aug_value = augmentation_row[aug_col].values[0]
            current_file_df['aug.' + aug_col] = aug_value

        current_file_df.to_csv(BASELINES_MULTI_PATH + name, index=False)


if __name__ == '__main__':
    augment_ts_from_event_details(BASELINES_PATH_GPS)
    augment_ts_from_event_details(BASELINES_PATH_NO_GPS)
