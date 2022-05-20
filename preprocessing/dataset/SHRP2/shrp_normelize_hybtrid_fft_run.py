import os
from pathlib import Path
import pandas as pd
from sklearn import preprocessing

from util import pickle_util
import numpy as np
import scipy as sp
import scipy.fftpack



SEQ_SIZE = 100
STEP_SIZE = 1



MAIN_DIR = '../data_SHRP2/windows/seq_{}_step_{}/'.format(SEQ_SIZE, STEP_SIZE)
RELEVANT_COLS = '../data_SHRP2/relevant_columns_extended.csv'

ENCODER_DIR_PATH = '../data_SHRP2/windows/encoder/'
WINDOWS_ONE_FILE_DIR_PATH = MAIN_DIR + 'samples_one_file_extended_union/'
ONE_FILE_PATH = '{}samples_one_file_dict.csv'.format(WINDOWS_ONE_FILE_DIR_PATH)

relevant_cols = pd.read_csv(RELEVANT_COLS)
relevant_cols = relevant_cols.loc[relevant_cols['keep'] == 'yes']
cont_features = relevant_cols.loc[(relevant_cols['type'] == 'cont') | (relevant_cols['type'] == 'discrete')][
    'col_name'].values
cont_features_with_fft = [s + '_fft' for s in cont_features]
cont_features_with_fft = np.concatenate([cont_features, cont_features_with_fft])
cont_features = cont_features_with_fft
categorical_features_raw = ['280']
categorical_features_raw = []
categorical_features_raw = list(relevant_cols.loc[(relevant_cols['type'] == 'categorical')]['col_name'].values)
categorical_features_after_preprocessing = ['280_0.0', '280_2.0', '280_3.0', '280_5.0']
categorical_features_after_preprocessing = []

meta_cols = list(relevant_cols.loc[(relevant_cols['type'] == 'meta')]['col_name'].values)
meta_cols = ['series_num', 'vtti.timestamp']
bool_cols = list(relevant_cols.loc[(relevant_cols['type'] == 'bool')]['col_name'].values)

target = 'aug.road_type'

Path(ENCODER_DIR_PATH).mkdir(parents=True, exist_ok=True)
# categorical_features.append('aug.road_type')

# prepare scaler
def prepare_scaler(scaler_initilizer_func, samples_files, src_path, cont_features):
    # categorical_features = relevant_cols.loc[(relevant_cols['type'] == 'categorical') | (relevant_cols['type'] == 'bool')]['col_name'].values

    encoders = {}

    scaler = scaler_initilizer_func()
    # fit scaler in batches
    for idx, file_name in enumerate(samples_files):
        if idx % 100 == 0:
            print('scaler idx {}/{}'.format(idx, len(samples_files)))
        sample_df = pd.read_csv(src_path + file_name)
        sample_df_cont = sample_df.loc[:, sample_df.columns.isin(cont_features)]
        scaler.partial_fit(sample_df_cont)

        # sample_df_categorical = sample_df.loc[:, sample_df.columns.isin(categorical_features)]

    Path(META_DATA_MERGED_DIR_PATH).mkdir(parents=True, exist_ok=True)
    pickle_util.save_obj(scaler, SCALER_PATH)
    return scaler


def prepare_encoder(categorical_features):
    # fit label ancoder on data in one file
    print("preparing encoder")
    data_for_categorical_encoder = pd.read_csv(ONE_FILE_PATH)
    print(data_for_categorical_encoder.columns)
    print(categorical_features)
    data_for_categorical_encoder = data_for_categorical_encoder.loc[:,
                                   data_for_categorical_encoder.columns.isin(categorical_features)]

    print(data_for_categorical_encoder.columns)
    data_for_categorical_encoder.rename(columns=str)
    print(data_for_categorical_encoder.columns)
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(data_for_categorical_encoder)

    pickle_util.save_obj(encoder, ENCODER_DIR_PATH + 'encoder.pickle')

    return encoder


def transform(src_data_dir_path, results_preprocessed_data_dir_path, encoder, categorical_features,
              cont_features, scaler=None, scaler_initilizer_func=None, fft=True,
              should_encode_cat=False, pattern_features=None):
    print('saving data in: {}'.format(results_preprocessed_data_dir_path))
    Path(results_preprocessed_data_dir_path).mkdir(parents=True, exist_ok=True)
    samples_files = os.listdir(src_data_dir_path)
    num_of_files = len(samples_files)

    print("converting stuff")
    for idx, file in enumerate(samples_files):
        if idx >= 43000:
            sample_df = pd.read_csv(src_data_dir_path + file)
            sample_df_cont_filter = sample_df.loc[:, sample_df.columns.isin(cont_features)]
            sample_df_categorical_filter = sample_df.loc[:, sample_df.columns.isin(categorical_features)]
            # transform
            if scaler != None:  # scale together
                # print('existing scaler')
                sample_cont = scaler.transform(sample_df_cont_filter)
            else:
                # std_ = np.nanstd(sample_df_cont_filter, axis=1, keepdims=True)
                # sample_cont_noa = sample_df_cont_filter - np.nanmean(sample_df_cont_filter, axis=1, keepdims=True)/std_
                scaler = scaler_initilizer_func()
                # print('scaler is:  {}'.format(scaler))
                sample_cont = scaler.fit_transform(sample_df_cont_filter)
            if fft == True:
                sample_cont_fft = np.apply_along_axis(absfft, 0, sample_cont)

                # convert cont data to dataframes
                cont_fft_df = pd.DataFrame(sample_cont_fft, columns=sample_df_cont_filter.columns.astype(str) + '_fft')
                cont_df = pd.DataFrame(sample_cont, columns=sample_df_cont_filter.columns)

                # name is wrong just for rest of the flow
                cont_plus_fft_df = pd.concat([cont_df, cont_fft_df], axis=1)

            # transforming categorical data
            if should_encode_cat == True:
                sample_categorical = encoder.transform(sample_df_categorical_filter).toarray()
                categorical_df = pd.DataFrame(sample_categorical, columns=encoder.get_feature_names(categorical_features))
            else:
                sample_categorical = sample_df_categorical_filter.values
                categorical_df = pd.DataFrame(sample_categorical)
            if fft:
                cont_df_to_add = cont_plus_fft_df
            else:
                cont_df_to_add = pd.DataFrame(sample_cont, columns=sample_df_cont_filter.columns)

            meta_df = sample_df[meta_cols]
            bool_df = sample_df.loc[:, sample_df.columns.isin(bool_cols)]
            target_df = sample_df[[target]]

            if pattern_features != None:
                patten_scaler = scaler_initilizer_func()
                patten_features_df = cont_df_to_add[pattern_features]
                pattern_features_nd = patten_scaler.fit_transform(patten_features_df)
                pattern_columns_names = patten_features_df.columns.astype(str) + '_pattern'
                patten_features_df = pd.DataFrame(pattern_features_nd, columns=pattern_columns_names)
                cont_df_to_add = pd.concat([cont_df_to_add, patten_features_df], axis=1)

            preprocessed_df = pd.concat([meta_df, cont_df_to_add, categorical_df, bool_df, target_df], axis=1)

            if idx % 1000 == 0:
                print('saving file at idx: {}/{}, file name: {}'.format(idx, num_of_files, file))
            preprocessed_df.to_csv(results_preprocessed_data_dir_path + file, index=False)


def preprocess_data_all_series_together(src_path, results_path, scaler_initilizer_func, fft, categorical_features,
                                        cont_features,
                                        should_encoder_categorical=True, pattern_features=None):
    print("preprocessing data together")
    if SHOULD_LOAD_SCALER:
        print("loading scaler")
        scaler = pickle_util.load_obj(SCALER_PATH)
    else:
        print("praparing scaler")
        sample_files = os.listdir(src_path)
        scaler = prepare_scaler(scaler_initilizer_func, sample_files, src_path, cont_features)
    if SHOULD_LOAD_ENCODER:
        print("loading encoder")
        encoder = pickle_util.load_obj(ENCODER_DIR_PATH + 'encoder.pickle')
    else:
        print("praparing encoder")
        encoder = prepare_encoder(categorical_features)
    transform(src_path, results_path, encoder, categorical_features, cont_features, scaler, fft=fft,
              scaler_initilizer_func=scaler_initilizer_func,
              should_encode_cat=should_encoder_categorical, pattern_features=pattern_features)


def preprocess_data_separate(src_path, results_path, scaler_initilizer_func, fft, categorical_features,
                             cont_features, should_encoder_categorical):
    print("preprocessing data separately")
    if SHOULD_LOAD_ENCODER:
        encoder = pickle_util.load_obj(ENCODER_DIR_PATH + 'encoder.pickle')
    else:
        encoder = prepare_encoder(categorical_features)
    transform(src_path, results_path, encoder, categorical_features=categorical_features,
              cont_features=cont_features_with_fft, scaler_initilizer_func=scaler_initilizer_func, fft=fft,
              should_encode_cat=should_encoder_categorical)


def preprocess_data_hybrid(src_path, separate_results_path, final_results_path, scaler_initilizer_func_separate,
                           scaler_initilizer_func_together,
                           fft_for_each_series,
                           categorical_features_for_separate,
                           categoricae_features_for_together,
                           should_process_separate=False):
    print("preprocessing data hybrid")
    if should_process_separate:
        preprocess_data_separate(src_path, separate_results_path, scaler_initilizer_func_separate,
                                 fft=fft_for_each_series, categorical_features=categorical_features_for_separate)
    preprocess_data_all_series_together(separate_results_path, final_results_path, scaler_initilizer_func_together,
                                        fft=False,
                                        categorical_features=categoricae_features_for_together,
                                        should_encoder_categorical=False)


def absfft(x):
    res = np.abs(np.fft.rfft(x, n=x.shape[0]))
    # return res
    fft = sp.fft.rfft(x)
    fft_squre = np.abs(fft) ** 2
    freq = sp.fftpack.fftfreq(len(fft_squre), 1. / len(x))
    return fft_squre


def init_min_max_scaler():
    scaler = preprocessing.MinMaxScaler()
    return scaler


def init_standard_scaler():
    scaler = preprocessing.StandardScaler()
    return scaler


scalers_init_map = {'min_max': init_min_max_scaler, 'standard': init_standard_scaler}



# # together
# RESULTS_PREPROCESSED_TOGETHER_MIN_MAX_FFT_DIR_PATH = MAIN_DIR + 'together/samples_min_max_fft/'
# RESULTS_PREPROCESSED_TOGETHER_STANDERD_DIR_PATH = MAIN_DIR + 'together/samples_standard_scaler/'
#
#
# # separate
# WINDOWS_PREPROCESSED_SEPARATE_STANDARD_FFT_DIR_PATH = MAIN_DIR + 'separate/samples_standard_scaler_fft/'
#
#
# # hybrid
# WINDOWS_PREPROCESSED_HYBRID_STANDARD_FFT_DIR_PATH = MAIN_DIR + 'hybrid/samples_standard_fft/'


SCALING_WIDE_TYPE = 'together'
SCALER_TYPE = 'min_max'
SHOULD_LOAD_SCALER = True
SHOULD_LOAD_ENCODER = True


pattern_features = None
pattern_features = ['1158', '1159', '1160', '11', '281', '452']
pattern_features = ['vtti.accel_x', 'vtti.accel_y', 'vtti.accel_z']

pattern_features_str = ""
if pattern_features != None:
    pattern_features_str = '_pattern_extended'

timedelta = 60
timedelta = ""

timedelta_str = ""
if timedelta != "":
    timedelta_str = '_{}'.format(timedelta)


# WINDOWS_SRC_MERGED_LABELS_DIR_PATH = MAIN_DIR + 'samples_merged_3_cls_labels_fe{}/'.format(timedelta_str)
WINDOWS_SRC_MERGED_LABELS_DIR_PATH = MAIN_DIR + 'samples_extended_union{}/'.format(timedelta_str)
RESULTS_PREPROCESSED_DIR = MAIN_DIR + '{}/samples_{}_fft_3_cls_fe{}{}/'.format(SCALING_WIDE_TYPE, SCALER_TYPE, pattern_features_str,
                                                                               timedelta_str)
META_DATA_MERGED_DIR_PATH = MAIN_DIR + 'meta_merged_labels{}{}/scaler/{}/{}/'.format(timedelta_str, pattern_features_str, SCALING_WIDE_TYPE, SCALER_TYPE)
SCALER_PATH = '{}scaler_SEQ_{}_STEP_{}.pickle'.format(META_DATA_MERGED_DIR_PATH, SEQ_SIZE,
                                                      STEP_SIZE)

print('running on: {} scaler type, on: {} data, saving results in: {}'.format(SCALER_TYPE, SCALING_WIDE_TYPE,
                                                                              RESULTS_PREPROCESSED_DIR))
print('src data in: {}'.format(WINDOWS_SRC_MERGED_LABELS_DIR_PATH))
print('feature engineering data')

if SCALING_WIDE_TYPE is 'together':
    preprocess_data_all_series_together(src_path=WINDOWS_SRC_MERGED_LABELS_DIR_PATH,
                                        results_path=RESULTS_PREPROCESSED_DIR,
                                        scaler_initilizer_func=scalers_init_map[SCALER_TYPE],
                                        fft=False,
                                        categorical_features=categorical_features_raw,
                                        cont_features=cont_features_with_fft,
                                        should_encoder_categorical=True,
                                        pattern_features=pattern_features)

elif SCALING_WIDE_TYPE is 'separate':
    preprocess_data_separate(src_path=WINDOWS_SRC_MERGED_LABELS_DIR_PATH,
                             results_path=RESULTS_PREPROCESSED_DIR,
                             scaler_initilizer_func=scalers_init_map[SCALER_TYPE],
                             fft=False,
                             categorical_features=categorical_features_raw,
                             cont_features=cont_features_with_fft,
                             should_encoder_categorical=True)
print('Done: running on: {} scaler type, on: {} data, saving results in: {}'.format(SCALER_TYPE, SCALING_WIDE_TYPE,
                                                                              RESULTS_PREPROCESSED_DIR))
# preprocess_data_hybrid(src_path=WINDOWS_SRC_MERGED_LABELS_DIR_PATH,
#                        separate_results_path=WINDOWS_PREPROCESSED_SEPARATE_STANDARD_FFT_DIR_PATH,
#                        final_results_path=WINDOWS_PREPROCESSED_HYBRID_STANDARD_FFT_DIR_PATH,
#                        scaler_initilizer_func_separate=init_standard_scaler,
#                        scaler_initilizer_func_together=init_standard_scaler,
#                        fft_for_each_series=True,
#                        categorical_features_for_separate=categorical_features_raw,
#                        categoricae_features_for_together=categorical_features_after_preprocessing,
#                        should_process_separate=False)
# preprocess_data_separate(WINDOWS_SRC_MERGED_LABELS_DIR_PATH, WINDOWS_PREPROCESSED_HYBRID_STANDARD_FFT_DIR_PATH,
#                          scaler_initilizer_func= init_standard_scaler, fft=True, categorical_features=categorical_features_raw)
