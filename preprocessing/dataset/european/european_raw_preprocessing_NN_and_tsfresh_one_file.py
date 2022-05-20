import os
from pathlib import Path

import pandas as pd
import numpy as np

from preprocessing.utils.concat_label import label_concat_util_sessions_refactored as concator
from preprocessing.utils.empty_values import fill_empty_values_refactored as filler
from preprocessing.utils.filtering import filter_util_2_refactored as filter
from preprocessing.utils.partitioning import data_partitioning_util_gss as partitioning
from preprocessing.utils.windows import window_sequences_european_seperated_refactored as sequencer
from util.stats import european_statistics_refactored as stator

timedelta = 60
timedelta_str = '_{}'.format(timedelta)
timedelta_str = ""

RAW_SESSIONS_PATH = '../results/flat/sessions/'
RAW_FULL_ON_CHANGE_PATH = '../results/raw_preprocessing/raw_full_on_change/'
SUB_SESSIONS_PATH = '../results/raw_preprocessing/sub_sessions/'
SUB_SESSIONS_PATH = '../results/raw_preprocessing/sub_sessions{}/'.format(timedelta_str)
SUB_SESSIONS_ALL_FULL_PATH = '../results/raw_preprocessing/sub_sessions_full{}/'.format(timedelta_str)
SUB_SESSIONS_ALL_FULL_LABELED_PATH = '../results/raw_preprocessing/sub_sessions_full_labeled{}/'.format(timedelta_str)
SUB_SESSIONS_ALL_FULL_LABELED_FFT_PATH = '../results/raw_preprocessing/sub_sessions_full_labeled_fft{}/'.format(
    timedelta_str)
SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_PATH = '../results/raw_preprocessing/sub_sessions_full_labeled_fft_fe{}/'.format(
    timedelta_str)
SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_3_CLS_PATH = '../results/raw_preprocessing/sub_sessions_full_labeled_fft_fe_3_cls{}/'.format(
    timedelta_str)

AUGMENTED_GPS_DATA_PATH = '../results/road_type/sessions/'

# RELEVANT_COLS = '../preprocessing/empty_values/relevant_cols.csv'
RELEVANT_COLS = '../../../preprocessing/empty_values/relevant_cols.csv'
SIGNALS_IMPORTANCE_PATH = '../util/signlas_importance.csv'
STATISTICS_RESULTS_DIR_PATH = 'statistics_results/raw_preprocessing{}/'.format(timedelta_str)

SEQ_SIZE = 200
STEP_SIZE = 25
FE = '_fe'
FE = ''

RAW_TARGET = 'road_type'
TARGET = 'aug.road_type'
MAIN_DIR = '../results/raw_preprocessing/windows/seq_{}_step_{}/'.format(SEQ_SIZE, STEP_SIZE)
META_DATA_DIR_PATH = MAIN_DIR + 'meta/'
META_DATA_MERGED_DIR_PATH = MAIN_DIR + 'meta_merged_labels{}/'.format(FE)
META_DATA_MERGED_3_CLS_DIR_PATH = MAIN_DIR + 'meta_merged_labels_3_cls{}{}/'.format(FE, timedelta_str)

WINDOWS_DIR_PATH = MAIN_DIR + 'samples/'
WINDOWS_MERGED_DIR_PATH = MAIN_DIR + 'samples_merged_labels{}/'.format(FE)
WINDOWS_MERGED_3_CLS_DIR_PATH = MAIN_DIR + 'samples_merged_3_cls_labels{}{}/'.format(FE, timedelta_str)

WINDOWS_ONE_FILE_DIR_PATH = MAIN_DIR + 'samples_one_file/'
WINDOWS_ONE_FILE_NAME = 'samples_one_file_dict.csv'
WINDOWS_ONE_FILE_MERGED_LABELS_NAME = 'samples_one_file_merged_labels.csv'
classes_merge_mapping = {'trunk': 'motorway', 'primary': 'secondary', 'unclassified': 'tertiary'}
classes_merge_mapping = {'trunk': 'motorway', 'primary': 'secondary', 'tertiary': 'secondary'}

TEST_SETS_MERGRED_DIR = MAIN_DIR + '/test_sets/mereged/'




# WINDOWS_DIR_PATH = '../results/raw_preprocessing/windows/seq_size_{}_step_size_{}/'.format(SEQ_SIZE, STEP_SIZE)
# META_DATA_DIR_PATH = '../results/raw_preprocessing/windows/seq_size_{}_step_size_{}_meta/'.format(SEQ_SIZE, STEP_SIZE)

# WINDOWS_DIR_PATH = '../results/raw_preprocessing/windows/seq_size_{}_step_size_{}/'.format(SEQ_SIZE, STEP_SIZE)
#
# META_DATA_DIR_PATH = '../results/raw_preprocessing/windows/seq_size_{}_step_size_{}_meta/'.format(SEQ_SIZE, STEP_SIZE)
#
# ONE_FILE_WINDOWS_DIR_PATH = '../results/raw_preprocessing/windows/seq_size_{}_step_size_{}_one_file/'.format(SEQ_SIZE, STEP_SIZE)
#
# ONE_FILE_WINDOWS_FILE_NAME = 'seq_{}_step_{}.csv'.format(SEQ_SIZE, STEP_SIZE)


def create_folders():
    # create folders
    Path(MAIN_DIR).mkdir(parents=True, exist_ok=True)
    Path(SUB_SESSIONS_PATH).mkdir(parents=True, exist_ok=True)
    Path(SUB_SESSIONS_ALL_FULL_PATH).mkdir(parents=True, exist_ok=True)
    Path(SUB_SESSIONS_ALL_FULL_LABELED_PATH).mkdir(parents=True, exist_ok=True)
    # Path(SUB_SESSIONS_ALL_FULL_LABELED_FFT_PATH).mkdir(parents=True, exist_ok=True)
    Path(SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_PATH).mkdir(parents=True, exist_ok=True)
    Path(SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_3_CLS_PATH).mkdir(parents=True, exist_ok=True)
    Path(STATISTICS_RESULTS_DIR_PATH).mkdir(parents=True, exist_ok=True)

    Path(META_DATA_DIR_PATH).mkdir(parents=True, exist_ok=True)
    Path(WINDOWS_DIR_PATH).mkdir(parents=True, exist_ok=True)
    Path(WINDOWS_ONE_FILE_DIR_PATH).mkdir(parents=True, exist_ok=True)
    Path(WINDOWS_MERGED_DIR_PATH).mkdir(parents=True, exist_ok=True)
    Path(WINDOWS_MERGED_3_CLS_DIR_PATH).mkdir(parents=True, exist_ok=True)

    Path(META_DATA_MERGED_DIR_PATH).mkdir(parents=True, exist_ok=True)
    Path(TEST_SETS_MERGRED_DIR).mkdir(parents=True, exist_ok=True)
    Path(SUB_SESSIONS_ALL_FULL_LABELED_FFT_PATH).mkdir(parents=True, exist_ok=True)
    Path(SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_PATH).mkdir(parents=True, exist_ok=True)
    Path(SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_3_CLS_PATH).mkdir(parents=True, exist_ok=True)


class SubSessionsLabelsStat:
    def __init__(self):
        self.road_type_nan_sessions = []
        self.partial_road_type_nan_sessions = []
        self.road_type_NA_sessions = []
        self.partial_road_type_NA_sessions = []

    def create_csv_to_one_list(self, list_to_save, csv_name):
        road_type_nan_sessions = [s.strip('.csv') for s in list_to_save]
        nans = pd.DataFrame(road_type_nan_sessions, columns=['sid'])
        nans.to_csv(csv_name, index=False)

    def create_csv(self):
        self.create_csv_to_one_list(self.road_type_nan_sessions, 'road_type_nan_sessions.csv')
        self.create_csv_to_one_list(self.partial_road_type_nan_sessions, 'partial_road_type_nan_sessions.csv')
        self.create_csv_to_one_list(self.road_type_NA_sessions, 'road_type_NA_sessions.csv')
        self.create_csv_to_one_list(self.partial_road_type_NA_sessions, 'partial_road_type_NA_sessions.csv')


def complete_only_on_change_empty_values():
    files = os.listdir(RAW_SESSIONS_PATH)
    for session_file in files:
        filler.fill_empty_values_by_type(RAW_SESSIONS_PATH, session_file, RAW_FULL_ON_CHANGE_PATH, RELEVANT_COLS,
                                         on_change_only=True)


def create_sub_sessions():
    files = os.listdir(RAW_FULL_ON_CHANGE_PATH)
    for session_file in files:
        filter.create_relevant_sessions(RAW_FULL_ON_CHANGE_PATH, session_file, SUB_SESSIONS_PATH, timedelta_sec=60)


def complete_all_empty_values():
    files = os.listdir(SUB_SESSIONS_PATH)
    for session_file in files:
        filler.fill_empty_values_by_type(SUB_SESSIONS_PATH, session_file, SUB_SESSIONS_ALL_FULL_PATH, RELEVANT_COLS,
                                         on_change_only=False)


def concat_label():
    files = os.listdir(SUB_SESSIONS_ALL_FULL_PATH)
    sub_sessions_stat = SubSessionsLabelsStat()
    for session_file in files:
        concator.concat_label_to_can_bus(SUB_SESSIONS_ALL_FULL_PATH, session_file, AUGMENTED_GPS_DATA_PATH,
                                         SUB_SESSIONS_ALL_FULL_LABELED_PATH, RELEVANT_COLS, sub_sessions_stat)
    sub_sessions_stat.create_csv()


def calc_empty_cols_stat():
    stator.calc_sessions_fullness(SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_3_CLS_PATH, STATISTICS_RESULTS_DIR_PATH)
    colums_stats = stator.calc_columns_missing_values(STATISTICS_RESULTS_DIR_PATH, STATISTICS_RESULTS_DIR_PATH)
    print(colums_stats)
    stator.join_percent_empty_to_importance(STATISTICS_RESULTS_DIR_PATH, SIGNALS_IMPORTANCE_PATH)




def get_cols_to_drop():
    cols_stats = pd.read_csv(STATISTICS_RESULTS_DIR_PATH + 'percent_empty_and_signal_importance.csv')
    cols_to_drop = cols_stats.loc[cols_stats['percent_empty'] > 30]['signal_id'].astype(str)
    return cols_to_drop


def create_window_sequences_separate(src_dir, results_dir):
    # TODO: outer function get_cols_to_drop
    print('create windows from: {}'.format(src_dir))
    print('saving windows in: {}'.format(results_dir))
    cols_stats = pd.read_csv(STATISTICS_RESULTS_DIR_PATH + 'percent_empty_and_signal_importance.csv')
    cols_to_drop = cols_stats.loc[cols_stats['percent_empty'] > 30]['signal_id'].astype(str)
    Path(WINDOWS_DIR_PATH).mkdir(parents=True, exist_ok=True)
    sequencer.create_european_sequences_separate_files_per_series_results(
        labeled_data_path=src_dir,
        seq_size=SEQ_SIZE, step_size=STEP_SIZE,
        results_dir=results_dir, cols_to_drop=[])


def create_windows_meta_data(samples_dir, results_dir):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    num_of_files = len(os.listdir(samples_dir))
    # partitioning.create_meta_data_df(WINDOWS_MERGED_DIR_PATH, path + 'samples_meta_data.csv', 0, 5)
    partitioning.create_meta_data_df_shrp(samples_dir, results_dir + 'samples_meta_data.csv', 0, num_of_files + 1)


def create_one_windows_file(windows_path, one_file_dir_path, one_file_name, files_names_list=None):
    print('creating one file')
    Path(one_file_dir_path).mkdir(parents=True, exist_ok=True)
    if files_names_list != None:
        file_names = files_names_list
    else:
        file_names = os.listdir(windows_path)
    num_of_files = len(file_names)
    combined_samples_file = pd.DataFrame([])
    combined_samples_file_dict = {}
    relevant_cols = pd.read_csv(RELEVANT_COLS)['col_name']
    relevant_cols = relevant_cols.append(pd.Series(['series_num', 'fid_x', 't_x', 'aug.road_type']))
    for idx, file in enumerate(file_names):
        # print('adding file num {} out of {}'.format(idx, num_of_files))

        current_sample_file = pd.read_csv(windows_path + file)
        current_sample_file = current_sample_file.loc[:, current_sample_file.columns.isin(relevant_cols)]
        current_sample_file['series_num'] = idx
        current_sample_file['sub_sid'] = file.split('_series')[0]
        current_sample_file['sample_file_name'] = file
        combined_samples_file_dict[idx] = current_sample_file
        # combined_samples_file = combined_samples_file.append(current_sample_file)
    values_to_add = list(combined_samples_file_dict.values())
    combined_samples_file = combined_samples_file.append(values_to_add)
    combined_samples_file.reset_index(drop=True, inplace=True)
    results_path = one_file_dir_path + one_file_name
    print("saving one file in: {}".format(results_path))
    combined_samples_file.to_csv(results_path, index=False)


def merge_labels_for_windows():
    samples_files = os.listdir(WINDOWS_DIR_PATH)
    for idx, file in enumerate(samples_files):
        if idx % 100 == 0:
            print('file {} out of {}'.format(idx, len(samples_files)))
        sample_df = pd.read_csv(WINDOWS_DIR_PATH + file)
        current_target = sample_df[TARGET].unique()[0]
        if current_target in classes_merge_mapping:
            sample_df[TARGET] = classes_merge_mapping[current_target]
            sample_df.to_csv(WINDOWS_MERGED_DIR_PATH + file, index=False)
        else:
            sample_df.to_csv(WINDOWS_MERGED_DIR_PATH + file, index=False)


def merge_labels_for_sub_sessions_three_classes():
    src_path = SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_PATH
    results_path = SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_3_CLS_PATH
    sub_sessions = os.listdir(src_path)
    for idx, file in enumerate(sub_sessions):
        if idx % 100 == 0:
            print('file {} out of {}'.format(idx, len(sub_sessions)))
        sub_session_df = pd.read_csv(src_path + file)

        # remove unclassified class
        sub_session_df = sub_session_df.loc[sub_session_df[RAW_TARGET] != 'unclassified']
        # union classes of similar road to one road name
        sub_session_df = sub_session_df.replace({RAW_TARGET: classes_merge_mapping})

        sub_session_df.to_csv(results_path + file, index=False)


def merge_labeld_one_file():
    one_file = pd.read_csv(WINDOWS_ONE_FILE_DIR_PATH + WINDOWS_ONE_FILE_NAME)
    merged_labels_one_file = one_file.replace({TARGET: classes_merge_mapping})
    merged_labels_one_file.to_csv(WINDOWS_ONE_FILE_DIR_PATH + WINDOWS_ONE_FILE_MERGED_LABELS_NAME, index=False)


def remove_index():
    samples_files = os.listdir(WINDOWS_MERGED_DIR_PATH)
    for idx, file in enumerate(samples_files):
        print('file {} out of {}'.format(idx, len(samples_files)))
        sample_df = pd.read_csv(WINDOWS_DIR_PATH + file)
        sample_df.to_csv(WINDOWS_DIR_PATH + file, index=False)


def add_fft_t_sub_sessions_data(abs_acc=False):
    sub_sessions_files = os.listdir(SUB_SESSIONS_ALL_FULL_LABELED_PATH)
    cols_to_drop = get_cols_to_drop()
    relevant_cols = pd.read_csv(RELEVANT_COLS)
    cont_features = relevant_cols.loc[(relevant_cols['type'] == 'cont') | (relevant_cols['type'] == 'discrete')][
        'col_name'].values
    results_path = SUB_SESSIONS_ALL_FULL_LABELED_FFT_PATH

    for sub_file_name in sub_sessions_files:
        sub_file_df = pd.read_csv(SUB_SESSIONS_ALL_FULL_LABELED_PATH + sub_file_name)
        sub_file_df.drop(cols_to_drop, axis=1, inplace=True)

        # sample_df_cont_filter = sample_df.loc[:, sample_df.columns.isin(cont_features)]
        sub_file_fft_df = sub_file_df.loc[:, sub_file_df.columns.isin(cont_features)].apply(absfft)
        sub_file_fft_df.columns = sub_file_fft_df.columns.astype(str) + '_fft'

        target_temp = sub_file_df['road_type']

        sub_file_df.drop('road_type', axis=1, inplace=True)
        sub_file_df_reg_and_fft = pd.concat([sub_file_df, sub_file_fft_df, target_temp], axis=1)
        if abs_acc:
            sub_file_df_reg_and_fft['1159'] = sub_file_df_reg_and_fft['1159'].abs()
            sub_file_df_reg_and_fft['1160'] = sub_file_df_reg_and_fft['1160'].abs()
            results_path = SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_PATH
        sub_file_df_reg_and_fft.to_csv(results_path + sub_file_name, index=False)


def absfft(x):
    return np.abs(np.fft.fft(x, n=x.shape[0]))
    # # return res
    # fft = sp.fft.fft(x)
    # fft_abs = np.abs(fft)
    # # freq = sp.fftpack.fftfreq(len(fft_squre), 1. / len(x))
    # return fft_abs


def feature_engineering_on_sub_sessions():
    main_dir = SUB_SESSIONS_ALL_FULL_LABELED_FFT_PATH
    results_dir = SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_PATH
    sessions_files = os.listdir(main_dir)
    # relevnat_cols = pd.read_csv(RELEVANT_COLS)
    # relevnat_cols = relevnat_cols.loc[relevnat_cols['feature_engineernig'] != null][['col_name', 'feature_engineernig']]

    for file_name in sessions_files:
        session_df = pd.read_csv(main_dir + file_name)
        session_df['1159'] = session_df['1159'].abs()
        session_df['1160'] = session_df['1160'].abs()
        session_df.to_csv(results_dir + file_name, index=False)


# TODO: deleter, this process should come before windows creation
# def feature_engineering_for_windows():
#     print("adding abs to acceleration")
#     main_dir = WINDOWS_MERGED_DIR_PATH
#     results_dir = WINDOWS_MERGED_FE_DIR_PATH
#     sessions_files = os.listdir(main_dir)
#     # relevnat_cols = pd.read_csv(RELEVANT_COLS)
#     # relevnat_cols = relevnat_cols.loc[relevnat_cols['feature_engineernig'] != null][['col_name', 'feature_engineernig']]
#
#     for file_name in sessions_files:
#         session_df = pd.read_csv(main_dir + file_name)
#         session_df['1159'] = session_df['1159'].abs()
#         session_df['1160'] = session_df['1160'].abs()
#         session_df.to_csv(results_dir+ file_name, index=False)

if __name__ == '__main__':
    create_folders()
    complete_only_on_change_empty_values()
    create_sub_sessions()
    complete_all_empty_values()
    concat_label()
    add_fft_t_sub_sessions_data(abs_acc=True)
    merge_labels_for_sub_sessions_three_classes()
    calc_empty_cols_stat()

    create_window_sequences_separate(src_dir=SUB_SESSIONS_ALL_FULL_LABELED_FFT_FE_3_CLS_PATH,
                                     results_dir=WINDOWS_MERGED_3_CLS_DIR_PATH)
    meta_df = pd.read_csv(META_DATA_MERGED_DIR_PATH + 'samples_meta_data.csv')
    create_windows_meta_data(WINDOWS_MERGED_3_CLS_DIR_PATH, META_DATA_MERGED_3_CLS_DIR_PATH)
    create_one_windows_file(WINDOWS_MERGED_3_CLS_DIR_PATH, WINDOWS_ONE_FILE_DIR_PATH, WINDOWS_ONE_FILE_NAME)




