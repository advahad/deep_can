import pandas as pd
import numpy as np

#
# LABELED_DATA_PATH = "../results/labeled/labeled_joint.csv"
#
# RESULTS_PATH = "../results/labeled/labeled_joint_seq.csv"


def create_trips(LABELED_DATA_PATH, RESULTS_PATH):
    data = pd.read_csv(LABELED_DATA_PATH)
    # creating trips
    data_as_sequences = pd.DataFrame([])
    data_as_series_list = pd.DataFrame([])
    road_type_sid_groups = data.groupby(['road_type', 'sid'])
    seq_next_idx = 0
    seq_size = 32
    for key, group_df in road_type_sid_groups:
        # check if there is seq smaller than seq_size
        remainder = group_df.shape[0] % seq_size
        if remainder > 0: # add the needed reminder to proper trip deviation
            empty = np.empty(shape=(seq_size - remainder, group_df.shape[1]))
            empty[:] = np.nan
            rows_to_add = pd.DataFrame(empty, columns=group_df.columns)
            rows_to_add['road_type'] = key[0]
            group_df = group_df.append(rows_to_add)
            group_df = group_df.reset_index(drop=True)
            for signal_col in group_df.columns:
                group_df[signal_col].interpolate(method='linear', inplace=True)

        group_df = group_df.reset_index(drop=True)
        group_df.insert(0, 'series_num', (group_df.index / seq_size).astype(int) + seq_next_idx)
        seq_next_idx = group_df['series_num'].max() + 1

        data_as_sequences = data_as_sequences.append(group_df)
    data_as_sequences.reset_index(drop=True)
    data_as_sequences.to_csv(RESULTS_PATH, index=False)
    classes = data_as_sequences.road_type.unique()
    print("")