from preprocessing.utils.windows import window_sequences_shrp_seperated_refactored as sequencer
from util.stats import shrp_statistics_refactored as stator
import pandas as pd
import os
import numpy as np
from xgboost import XGBRegressor
from pathlib import Path
from preprocessing.utils.partitioning import data_partitioning_util_gss as partitioning
from preprocessing.utils.empty_values import fill_empty_values_refactored as filler
from util import pickle_util
from sklearn.model_selection import train_test_split

SEQ_SIZE = 100
STEP_SIZE = 1
WINDOWS_ONE_FILE_NAME = 'samples_one_file_dict.csv'
SAMPLES_RAW_PATH = '../data_SHRP2/augmented_timeseries_one_roadtype_per_file/'
SAMPLES_RAW_PATH = '../data_SHRP2/augmented_timeseries/'
RELEVANT_COLS = '../../../data_SHRP2/relevant_columns_extended.csv'
classes_merge_mapping = {'trunk': 'motorway', 'primary': 'secondary', 'tertiary': 'secondary'}
classes_merge_mapping = {'trunk': 'motorway', 'primary': 'secondary', 'tertiary': 'secondary', 'unclassified': 'residential'}
classes_merge_mapping = {'trunk': 'motorway', 'primary': 'secondary', 'tertiary': 'secondary'}

extanded = ''
extanded = '_extended_union'
CLEAN_DATA_PATH = '../data_SHRP2/clean_augmented_timeseries_road_type_only_new/'
CLEAN_DATA_PATH = '../data_SHRP2/clean_augmented_timeseries{}/'.format(extanded)

# results dirs
CLEAN_DATA_FFT = '../data_SHRP2/clean_fft/'
CLEAN_DATA_FFT = '../data_SHRP2/clean_fft_new{}/'.format(extanded)
CLEAN_DATA_FFT_ABC = '../data_SHRP2/clean_fft_abs/'
CLEAN_DATA_FFT_ABC = '../data_SHRP2/clean_fft_abs_new{}/'.format(extanded)

CLEAN_DATA_FFT_ABC_MERGED = '../data_SHRP2/clean_fft_abs_merged_3_cls_new{}/'.format(extanded)




STATISTICS_RESULTS_DIR_PATH = '../data_SHRP2/statistics/'
SIGNALS_IMPORTANCE_PATH = '{}signlas_importance.csv'.format(STATISTICS_RESULTS_DIR_PATH)

MAIN_DIR = '../data_SHRP2/windows/seq_{}_step_{}/'.format(SEQ_SIZE, STEP_SIZE)
META_DATA_DIR_PATH = MAIN_DIR + 'meta/'.format(extanded)
WINDOWS_ONE_FILE_DIR_PATH = MAIN_DIR + 'samples_one_file{}/'.format(extanded)

WINDOWS_MERGED_3_CLS_DIR_PATH = MAIN_DIR + 'samples{}/'.format(extanded)
WINDOWS_ONE_FILE_NAME = 'samples_one_file_dict.csv'



TARGET = 'aug.road_type'


def complete_all_empty_values():
    print("filling empty data from: {}, saving in: {}".format(SAMPLES_RAW_PATH, CLEAN_DATA_PATH))
    files = os.listdir(SAMPLES_RAW_PATH)
    for session_file in files:
        filler.fill_empty_values_by_type(SAMPLES_RAW_PATH, session_file, CLEAN_DATA_PATH, RELEVANT_COLS,
                                         on_change_only=False)

    print("clean data saved in: {}".format(CLEAN_DATA_PATH))

def merge_labels_for_sub_sessions_three_classes(src_path, results_path):
    print('merging labels')
    sub_sessions = os.listdir(src_path)
    for idx, file in enumerate(sub_sessions):
        if idx % 100 == 0:
            print('file {} out of {}'.format(idx, len(sub_sessions)))
        if file == 'File_ID_59483397_Index_113612791.csv':
            print('adva')
        sub_session_df = pd.read_csv(src_path + file)

        # remove unclassified class
        sub_session_df = sub_session_df.loc[sub_session_df[TARGET] != 'unclassified']
        sub_session_df = sub_session_df.loc[sub_session_df[TARGET] != 'residential']
        if sub_session_df.empty:
            continue
        # union classes of similar road to one road name
        sub_session_df = sub_session_df.replace({TARGET: classes_merge_mapping})

        sub_session_df.to_csv(results_path + file, index=False)
    print('save results in: {}'.format(results_path))


def create_folders():
    Path(CLEAN_DATA_PATH).mkdir(parents=True, exist_ok=True)
    Path(CLEAN_DATA_FFT_ABC).mkdir(parents=True, exist_ok=True)
    Path(CLEAN_DATA_FFT_ABC_MERGED).mkdir(parents=True, exist_ok=True)
    Path(STATISTICS_RESULTS_DIR_PATH).mkdir(parents=True, exist_ok=True)
    Path(WINDOWS_MERGED_3_CLS_DIR_PATH).mkdir(parents=True, exist_ok=True)
    Path(META_DATA_DIR_PATH).mkdir(parents=True, exist_ok=True)
    Path(WINDOWS_ONE_FILE_DIR_PATH).mkdir(parents=True, exist_ok=True)








def get_cols_to_drop():
    cols_stats = pd.read_csv(STATISTICS_RESULTS_DIR_PATH + 'percent_empty.csv')
    cols_to_drop = cols_stats.loc[cols_stats['percent_empty'] > 30]['col_name'].astype(str)
    # return cols_to_drop
    return []

def absfft(x):
    return np.abs(np.fft.fft(x, n=x.shape[0]))

def add_fft_and_abs_sub_sessions_data(abs_acc=False):
    sub_sessions_files = os.listdir(CLEAN_DATA_PATH)
    cols_to_drop = get_cols_to_drop()
    relevant_cols = pd.read_csv(RELEVANT_COLS)
    relevant_cols = relevant_cols.loc[relevant_cols['keep'] == 'yes']
    cont_features = relevant_cols.loc[(relevant_cols['type'] == 'cont') | (relevant_cols['type'] == 'discrete')][
        'col_name'].values
    results_path = CLEAN_DATA_FFT

    for sub_file_name in sub_sessions_files:
        sub_file_df = pd.read_csv(CLEAN_DATA_PATH + sub_file_name)
        sub_file_df.drop(cols_to_drop, axis=1, inplace=True)

        # sample_df_cont_filter = sample_df.loc[:, sample_df.columns.isin(cont_features)]
        sub_file_fft_df = sub_file_df.loc[:, sub_file_df.columns.isin(cont_features)].apply(absfft)
        sub_file_fft_df.columns = sub_file_fft_df.columns.astype(str) + '_fft'

        target_temp = sub_file_df['aug.road_type']

        sub_file_df.drop('aug.road_type', axis=1, inplace=True)
        sub_file_df_reg_and_fft = pd.concat([sub_file_df, sub_file_fft_df, target_temp], axis=1)
        if abs_acc:
            sub_file_df_reg_and_fft['vtti.accel_y'] = sub_file_df_reg_and_fft['vtti.accel_y'].abs()
            sub_file_df_reg_and_fft['vtti.accel_z'] = sub_file_df_reg_and_fft['vtti.accel_z'].abs()
            results_path = CLEAN_DATA_FFT_ABC
        sub_file_df_reg_and_fft.to_csv(results_path + sub_file_name, index=False)
    print('results saved in: {}'.format(results_path))


def calc_empty_cols_stat(data_path):
    print("calc empty stat")
    # CLEAN_DATA_FFT_ABC_MERGED
    stator.calc_sessions_fullness(data_path, STATISTICS_RESULTS_DIR_PATH, 'vtti.speed_network')
    percent_empty_per_col = stator.calc_columns_missing_values(STATISTICS_RESULTS_DIR_PATH)
    print(percent_empty_per_col)
    # stator.join_percent_empty_to_importance(STATISTICS_RESULTS_DIR_PATH, SIGNALS_IMPORTANCE_PATH)

def create_window_sequences_separate(src_dir, results_dir):
    # TODO: outer function get_cols_to_drop
    print('create windows from: {}'.format(src_dir))
    print('saving windows in: {}'.format(results_dir))
    # cols_stats = pd.read_csv(STATISTICS_RESULTS_DIR_PATH + 'percent_empty_and_signal_importance.csv')
    # cols_to_drop = cols_stats.loc[cols_stats['percent_empty'] > 30]['signal_id'].astype(str)
    Path(src_dir).mkdir(parents=True, exist_ok=True)
    rpm_regressor = pickle_util.load_obj(META_DATA_DIR_PATH_reg + 'linear.pickle')
    sequencer.create_shrp2_sequences_separate_files_per_series_results(
        labeled_data_path=src_dir,
        seq_size=SEQ_SIZE, step_size=STEP_SIZE,
        results_dir=results_dir, cols_to_drop=[], rpm_regressor=rpm_regressor)
    print('save results in:{}'.format(results_dir))


def create_windows_meta_data(samples_dir, results_path):
    # Path(results_dir).mkdir(parents=True, exist_ok=True)
    num_of_files = len(os.listdir(samples_dir))
    # partitioning.create_meta_data_df(WINDOWS_MERGED_DIR_PATH, path + 'samples_meta_data.csv', 0, 5)
    partitioning.create_meta_data_df_shrp(samples_dir, results_path, 0, num_of_files + 1)


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
    # relevant_cols = relevant_cols.append(pd.Series(['series_num', 'fid_x', 't_x', 'aug.road_type']))
    relevant_cols = relevant_cols.append(pd.Series(['series_num', 'vtti.timestamp', 'aug.road_type']))
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




main_dir_reg = '../data_SHRP2/windows/seq_{}_step_{}/'.format(210, 2)
WINDOWS_ONE_FILE_DIR_PATH_reg = main_dir_reg + 'samples_one_file{}/'.format(extanded)

META_DATA_DIR_PATH_reg = main_dir_reg + 'meta/'.format(extanded)

def build_rpm_augmentor():
    WINDOWS_ONE_FILE_DIR_PATH_reg = main_dir_reg + 'samples_one_file{}/'.format(extanded)
    one_file = pd.read_csv(WINDOWS_ONE_FILE_DIR_PATH_reg + WINDOWS_ONE_FILE_NAME)
    print('bla')
    main_dir = '../data_SHRP2/windows/seq_{}_step_{}/'.format(210, 2)
    WINDOWS_ONE_FILE_DIR_PATH_reg = main_dir + 'samples_one_file{}/'.format(extanded)
    one_file = pd.read_csv(WINDOWS_ONE_FILE_DIR_PATH_reg + WINDOWS_ONE_FILE_NAME)
    META_DATA_DIR_PATH_reg = main_dir + 'meta/'.format(extanded)

    meta_df = pd.read_csv(META_DATA_DIR_PATH_reg + 'samples_meta_data.csv')
    print(meta_df)
    save_samples = list(meta_df.loc[meta_df['aug.is_rpm_filled'] == False]['sample_file_name'])
    one_file_only_will_orig_rpm = one_file.loc[one_file['sample_file_name'].isin(save_samples)]
    print(one_file_only_will_orig_rpm.columns)
    rpm_col_name = "vtti.engine_rpm_instant"
    rpm_flag_col = 'aug.is_rpm_filled'
    one_file_y = one_file_only_will_orig_rpm[rpm_col_name]
    # dropcols = ['sub_sid', 'sample_file_name', 'series_num', 'vtti.timestamp', rpm_col_name, 'vtti.light_level', 'vtti.temperature_interior']
    # one_file_x = one_file_only_will_orig_rpm.drop(dropcols, axis=1)
    keep = ['vtti.accel_x', 'vtti.accel_y', 'vtti.accel_z', 'vtti.gyro_x', 'vtti.gyro_y', 'vtti.gyro_z',
            'vtti.speed_network', 'vtti.prndl', 'vtti.temperature_interior', 'vtti.cruise_state', 'vtti.pedal_brake_state']
    categorical_features = ['vtti.prndl', 'vtti.cruise_state', 'vtti.pedal_brake_state']
    one_file_x = one_file_only_will_orig_rpm[keep]
    for cat_features in categorical_features:
        col_as_dummies = pd.get_dummies(one_file_x[cat_features], prefix=cat_features)

        one_file_x = pd.concat([one_file_x, col_as_dummies], axis=1)
        one_file_x.drop(cat_features, axis=1, inplace=True)

    print(one_file_x.columns)
    pickle_util.save_obj(one_file_x.columns, META_DATA_DIR_PATH_reg + 'columns_order.pickle')
    x_train, x_test, y_train, y_test = train_test_split(one_file_x, one_file_y, test_size=0.25, random_state=0)
    linearRegr = XGBRegressor(max_depth=15, n_estimators=200)
    linearRegr.fit(x_train, y_train)
    predictions = linearRegr.predict(x_test)
    score = linearRegr.score(x_test, y_test)

    pickle_util.save_obj(linearRegr, META_DATA_DIR_PATH_reg + 'linear.pickle')
    print(score)
    print('')
    # one_file_rpm_target = one_file[["vtti.engine_rpm_instant", TARGET]]
    # rpm_per_target_groups = one_file_rpm_target.groupby(TARGET)
    # for group_id, group_df in rpm_per_target_groups:
    #     filtered_df = group_df[group_df["vtti.engine_rpm_instant"] > 0]
    #     missing_rpm = len(group_df[group_df["vtti.engine_rpm_instant"] < 0])/len(group_df)*100
    #     mean = group_df[group_df["vtti.engine_rpm_instant"] > 0]["vtti.engine_rpm_instant"].sum() / len(group_df[group_df["vtti.engine_rpm_instant"] > 0])
    #     median = group_df[group_df["vtti.engine_rpm_instant"] > 0]["vtti.engine_rpm_instant"].median()
    #     print(group_id)
    #     print(missing_rpm)
    #     print('mean: {}'.format(mean))
    #     print('median: {}'.format(median))
    #
    #     from sklearn.datasets import load_diabetes
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #
    #     sns.set()
    #     import pandas as pd
    #
    #     # Get Data
    #     # data = load_diabetes()
    #     # X, y_ = data.data, data.target
    #
    #     # Organize Data
    #     SR_y = pd.Series(filtered_df["vtti.engine_rpm_instant"], name="{}: engine_rpm_instant (Vector Distribution)".format(group_id))
    #
    #     # Plot Data
    #     fig, ax = plt.subplots()
    #     sns.distplot(SR_y, bins=25, color="g", ax=ax)
    #     plt.show()
if __name__ == '__main__':
    create_folders()
    # build_rpm_augmentor()
    meta_path = META_DATA_DIR_PATH + 'samples_meta_data.csv'
    RELEVANT_COLS = '../data_SHRP2/relevant_columns_extended.csv'
    # meta = pd.read_csv(meta_path)
    # print(meta.head())

    # calc_empty_cols_stat(CLEAN_DATA_PATH)
    # augmentor.augment_data()
    #TODO: delete
    # merger.create_one_class_per_baseline()

    # complete_all_empty_values()

    # add_fft_and_abs_sub_sessions_data(abs_acc=True)
    #
    #
    #
    src_path = CLEAN_DATA_FFT_ABC
    results_path = CLEAN_DATA_FFT_ABC_MERGED
    # merge_labels_for_sub_sessions_three_classes(src_path, results_path)

    #
    create_window_sequences_separate(src_dir=CLEAN_DATA_FFT_ABC_MERGED,
                                     results_dir=WINDOWS_MERGED_3_CLS_DIR_PATH)
    meta_path = META_DATA_DIR_PATH + 'samples_meta_data.csv'
    create_windows_meta_data(WINDOWS_MERGED_3_CLS_DIR_PATH, meta_path)
    create_one_windows_file(WINDOWS_MERGED_3_CLS_DIR_PATH, WINDOWS_ONE_FILE_DIR_PATH, WINDOWS_ONE_FILE_NAME)





# # statistics
# meta_path = 'samples_meta_data_merged.csv'
# #
# create_windows_meta_data(CLEAN_DATA_FFT_ABC_MERGED, meta_path)
#
# meta_df = pd.read_csv(meta_path)
# print(meta_df)