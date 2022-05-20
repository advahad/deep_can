import pandas as pd
import os
import math
from datetime import datetime, timedelta

import os
from pathlib import Path

# STEP_SIZE = 10
# SEQ_SIZE = 30
OLD_TARGET = 'road_type'
CURRENT_TARGET = 'aug.road_type'


# PATH = '../results/labeled/relevant_sessions_full_labeled/'
# RESULTS_PATH = '../results/windows/SEQ_' + str(SEQ_SIZE) + '_STEP_' + str(STEP_SIZE) + '/'
# Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
from util import pickle_util
rpm_map = {'motorway': 2196, 'residential': 1142, 'secondary': 1655}
from random import randrange
extanded = '_extended_union'
main_dir = '../data_SHRP2/windows/seq_{}_step_{}/'.format(210, 2)
META_DATA_DIR_PATH_reg = main_dir + 'meta/'.format(extanded)

rpm_col_name = 'vtti.engine_rpm_instant'
rpm_flag_col = 'aug.is_rpm_filled'
TARGET = 'aug.road_type'
keep = ['vtti.accel_x', 'vtti.accel_y', 'vtti.accel_z', 'vtti.gyro_x', 'vtti.gyro_y', 'vtti.gyro_z',
        'vtti.speed_network', 'vtti.prndl', 'vtti.temperature_interior', 'vtti.cruise_state',
        'vtti.pedal_brake_state']
categorical_features = ['vtti.prndl', 'vtti.cruise_state', 'vtti.pedal_brake_state']

import random
def fill_and_flag_rpm(df, class_name, rpm_regressor):
    cols_order = pickle_util.load_obj(META_DATA_DIR_PATH_reg + 'columns_order.pickle')
    is_no_rpm = (df[rpm_col_name] < 0).all()
    if is_no_rpm:
        # random.uniform(rpm_map[class_name]*0.9, rpm_map[class_name]*1.1)
        # df[rpm_col_name] = randrange(rpm_map[class_name]*0.9, rpm_map[class_name]*1.1)
        # rpm_val = random.normalvariate(rpm_map[class_name], 20)

        x = df[keep]
        for cat_features in categorical_features:
            col_as_dummies = pd.get_dummies(x[cat_features], prefix=cat_features)

            x = pd.concat([x, col_as_dummies], axis=1)
            x.drop(cat_features, axis=1, inplace=True)
        import numpy as np
        x_full = pd.DataFrame(0, columns=cols_order, index=np.arange(len(x)))
        for col in x.columns:
            x_full[col] = x[col]

        x_full = x_full.loc[:, cols_order]
        rpm_val = rpm_regressor.predict(x_full)
        df[rpm_col_name] = rpm_val
        df[rpm_flag_col] = 'true'
    else:
        df[rpm_flag_col] = 'false'

    return df

def create_one_label(df, rpm_regressor):
    value_counts_percentage = df['aug.road_type'].value_counts(normalize=True, dropna=False)
    number_of_classes = len(value_counts_percentage)
    there_is_nan = False
    nan_percent = 100
    num_of_labeled_points_consider_full = len(df)/10
    num_of_current_labels_points = len(df[CURRENT_TARGET].dropna())
    label_fullness = (num_of_current_labels_points/num_of_labeled_points_consider_full)*100
    nan_percent = 100 - label_fullness

    for idx, value in value_counts_percentage.items():
        if pd.isna(idx):
            there_is_nan = True
            number_of_classes -= 1



    # if not there_is_nan and number_of_classes == 1:
    #     df[[CURRENT_TARGET]] = df[CURRENT_TARGET].fillna(method='ffill')
    #     df[[CURRENT_TARGET]] = df[CURRENT_TARGET].fillna(method='bfill')
    #     return df
    if number_of_classes > 2:
        return None
    elif number_of_classes == 1:
        if nan_percent < 30:
            # df.loc[:, CURRENT_TARGET].fillna(method='ffill', inplace=True)
            # df.loc[:, CURRENT_TARGET].fillna(method='bfill', inplace=True)
            df[CURRENT_TARGET].fillna(method='ffill', inplace=True)
            df[CURRENT_TARGET].fillna(method='bfill', inplace=True)

            # add rpm col
            class_name = df[CURRENT_TARGET].values[0]
            df = fill_and_flag_rpm(df.copy(), class_name, rpm_regressor)
            return df
        else:
            return None

    # if len(value_counts_percentage) > 1:
    #     print()
    # else:
    #     return df


# TODO: make sure no degradation for SHRP2 dataset
def create_windows(df, seq_size, window, start_seq_idx, windowed_data_dict, rpm_regressor):
    vel_0_trips_counter = 0
    gap_trips_counter = 0
    nan_NA_road_type_ctr = 0

    seq_next_idx = start_seq_idx
    for i in range(0, df.shape[0] - seq_size + 1, window):
        seq_to_add = df.iloc[i:seq_size + i]
        if (seq_to_add['vtti.speed_network'] == 0).all():
            vel_0_trips_counter += 1
            # print("trip is all with velocity 0")
            continue
        if seq_to_add['aug.road_type'].isnull().all() or (seq_to_add['aug.road_type'] == 'NA').all():
            nan_NA_road_type_ctr += 1
            continue
        # check if sequence is consecutive, no rest time (gap greater than 5 minutes)
        # deltas = seq_to_add['vtti.timestamp'].diff()[1:]
        # gaps = deltas[deltas > timedelta(minutes=5)]
        #
        # if gaps.empty:
        seq_to_add = create_one_label(seq_to_add, rpm_regressor)
        if seq_to_add is not None:  # there is one label generated or completed for the sequence
            if not seq_to_add.isnull().any().any():  # there is no nan values in the sequence features or label
                seq_to_add.insert(0, 'series_num', seq_next_idx)
                # TODO: add sub session name as col?
                windowed_data_dict[seq_next_idx] = seq_to_add
                seq_next_idx += 1
            else:
                print("sequence wasn't added partial nans {}".format(seq_next_idx))
        # else:
    #     gap_trips_counter += 1
    #         # print("there is gap greater than 5 minutes in session:{}".format(df['sid'][0]))
    # if gap_trips_counter > 0:
    #     print("deleted {} trips because of gaps".format(gap_trips_counter))
    if vel_0_trips_counter > 0:
        print("deleted {} trips because of zero velocity".format(vel_0_trips_counter))
    if nan_NA_road_type_ctr > 0:
        print("deleted {} trips because of nan / NA".format(nan_NA_road_type_ctr))
    # windowed_data.reset_index(drop=True, inplace=True)
    return windowed_data_dict, seq_next_idx


def create_shrp2_sequences(labeled_data_path, seq_size, step_size, results_dir, cols_to_drop):
    results_file_name = results_dir + 'seq_size_' + str(seq_size) + '_step_' + str(step_size) + '.csv'
    sessions_files = os.listdir(labeled_data_path)
    num_of_session_files = len(sessions_files)
    all_sequences = pd.DataFrame([])
    next_seq_id = 0
    windowed_data_dict = {}
    for idx, session_file_name in enumerate(sessions_files):
        # if idx < 102:
        #     continue
        print("\nfile idx {}, create windows for file {}/{}".format(idx, session_file_name, num_of_session_files))

        # result_path = results_dir + "\\labeled_sequence_win_" + str(step_size) + "_seq_" + str(seq_size) + ".csv"
        # data = pd.read_csv(labeled_data_path, parse_dates=['t_x'])
        session_df = pd.read_csv(labeled_data_path + session_file_name, parse_dates=['t_x'])
        session_df = session_df.drop(cols_to_drop, axis=1)

        # road_type_groups = session_df.groupby(['road_type'])

        # for key, group_df in road_type_groups:
        #     group_df = group_df.reset_index(drop=True)
        # if group_df['road_type'][0] == 'unclassified' or group_df['road_type'][0] == 'trunk':
        #     print('unclassified or trunk dataframe')
        #     continue
        # for signal_col in group_df.columns:
        #     group_df[signal_col].interpolate(method='linear', inplace=True)

        windowed_data_dict, next_seq_id = create_windows(session_df, seq_size, step_size, next_seq_id,
                                                         windowed_data_dict)
        # print("number of total sequences generated: {}".format(len(windowed_data_dict)))
        # if len(windowed_data_dict) > 0:
    windowed_data_dict_as_list = list(windowed_data_dict.values())
    all_sequences = all_sequences.append(windowed_data_dict_as_list, ignore_index=True, sort=False)

    all_sequences['aug.road_type'] = all_sequences['road_type']
    all_sequences.drop(['road_type'], axis=1, inplace=True)
    # print('total num of sequences: {}'.format(next_seq_id))
    all_sequences.to_csv(results_file_name, index=False)


def create_shrp2_sequences_separate_files_results(labeled_data_path, seq_size, step_size, results_dir, cols_to_drop):
    # results_file_name = results_dir + 'seq_size_' + str(seq_size) + '_step_' + str(step_size) + '.csv'
    sessions_files = os.listdir(labeled_data_path)

    for idx, sub_session_file_name in enumerate(sessions_files):
        # if idx > 1:
        #     continue
        print("\nfile idx {}, create windows for file {}".format(idx, sub_session_file_name))
        all_sequences = pd.DataFrame([])
        next_seq_id = 0
        windowed_data_dict = {}

        session_df = pd.read_csv(labeled_data_path + sub_session_file_name, parse_dates=['t_x'])
        session_df = session_df.drop(cols_to_drop, axis=1)

        windowed_data_dict, next_seq_id = create_windows(session_df, seq_size, step_size, next_seq_id,
                                                         windowed_data_dict)
        # print("number of total sequences generated: {}".format(len(windowed_data_dict)))

        if len(windowed_data_dict) > 0:
            windowed_data_dict_as_list = list(windowed_data_dict.values())
            all_sequences = all_sequences.append(windowed_data_dict_as_list, ignore_index=True, sort=False)

            all_sequences['aug.road_type'] = all_sequences['road_type']
            all_sequences.drop(['road_type'], axis=1, inplace=True)
            # print('total num of sequences: {}'.format(next_seq_id))
            all_sequences.to_csv(results_dir + sub_session_file_name, index=False)



def create_shrp2_sequences_separate_files_per_series_results(labeled_data_path, seq_size, step_size, results_dir,
                                                             cols_to_drop, rpm_regressor):
    # results_file_name = results_dir + 'seq_size_' + str(seq_size) + '_step_' + str(step_size) + '.csv'
    sessions_files = os.listdir(labeled_data_path)

    for idx, sub_session_file_name in enumerate(sessions_files):
        # if idx > 1:
        #     continue
        print("\nfile idx {}, create windows for file {}".format(idx, sub_session_file_name))
        sequence_df = pd.DataFrame([])
        next_seq_id = 0
        windowed_data_dict = {}

        session_df = pd.read_csv(labeled_data_path + sub_session_file_name, parse_dates=['vtti.timestamp'])
        session_df = session_df.drop(cols_to_drop, axis=1)

        windowed_data_dict, next_seq_id = create_windows(session_df, seq_size, step_size, next_seq_id,
                                                         windowed_data_dict, rpm_regressor)
        print("number of total sequences generated: {}".format(len(windowed_data_dict)))

        if len(windowed_data_dict) > 0:
            sub_session = sub_session_file_name.split('.')[0]
            for key, sequence_df in windowed_data_dict.items():
                # sequence_df['aug.road_type'] = sequence_df['road_type']
                # sequence_df.drop(['road_type'], axis=1, inplace=True)
                # print('total num of sequences: {}'.format(next_seq_id))
                sequence_df.to_csv(results_dir + sub_session + '_series_' + str(key) + '.csv', index=False)