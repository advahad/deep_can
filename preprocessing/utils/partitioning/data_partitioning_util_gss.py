import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from util import pickle_util
from pathlib import Path
# import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

#
# SAMPLES_PATH = '../../../results/raw_preprocessing/windows/seq_size_50_step_size_50/'
# COLS_META_PATH = '../../../preprocessing/empty_values/relevant_cols.csv'
# META_DATA_DF_PATH = 'samples_meta_data.csv'
#
# series_files = os.listdir(SAMPLES_PATH)

FILE_COL = 'sample_file_name'


def create_meta_data_df_shrp(samples_path, meta_data_df_result_path, start_index, end_index):
    samples_files = os.listdir(samples_path)
    meta_data_cols = ['sample_file_name', 'sid', 'sub_sid', 'vid', 'aug.road_type']
    meta_data_df = pd.DataFrame([], columns=meta_data_cols)
    meta_data_df_dict = {}
    num_of_files = len(samples_files)
    print('total num of files {}'.format(num_of_files))
    for idx, file_name in enumerate(samples_files):
        if start_index <= idx < end_index:
            if idx % 100 == 0:
                print('create meta row for file {} at idx {} out of {}'.format(file_name, idx, num_of_files))
            series_df = pd.read_csv(samples_path + file_name)
            # meta_row = pd.DataFrame([], columns=meta_data_cols)
            time_col_name = 'vtti.timestamp'
            sid = file_name.split('File_ID_')[1].split('_')[0]
            empty = np.empty(shape=(1, len(meta_data_cols)))
            empty[:] = np.nan
            meta_row = pd.DataFrame(empty, columns=meta_data_cols)
            meta_row['sample_file_name'] = file_name
            meta_row['sid'] = sid
            meta_row['sub_sid'] = sid
            meta_row['start_t'] = series_df[time_col_name].iloc[0]
            meta_row['end_t'] = series_df[time_col_name].iloc[-1]
            meta_row['vid'] = 'NA'
            rpm_flag_col = 'aug.is_rpm_filled'
            meta_row[rpm_flag_col] = series_df[rpm_flag_col].values[0]
            meta_row['aug.road_type'] = series_df['aug.road_type']
            meta_data_df_dict[idx] = meta_row.iloc[0]
            # meta_data_df = meta_data_df.append(meta_row.iloc[0])
    meta_data_values = list(meta_data_df_dict.values())
    meta_data_df = meta_data_df.append(meta_data_values)
    meta_data_df.to_csv(meta_data_df_result_path, index=False)


def create_meta_data_df_european(samples_path, meta_data_df_result_path, start_index, end_index):
    samples_files = os.listdir(samples_path)
    meta_data_cols = ['sample_file_name', 'sid', 'sub_sid', 'vid', 'aug.road_type']
    meta_data_df = pd.DataFrame([], columns=meta_data_cols)
    meta_data_df_dict = {}
    num_of_files = len(samples_files)
    print('total num of files {}'.format(num_of_files))
    for idx, file_name in enumerate(samples_files):
        if start_index <= idx < end_index:
            print('create meta row for file {} at idx {} out of {}'.format(file_name, idx, num_of_files))
            series_df = pd.read_csv(samples_path + file_name, usecols=['sid', 'vid', 't_x', 'aug.road_type'])
            # meta_row = pd.DataFrame([], columns=meta_data_cols)

            empty = np.empty(shape=(1, len(meta_data_cols)))
            empty[:] = np.nan
            meta_row = pd.DataFrame(empty, columns=meta_data_cols)
            meta_row['sample_file_name'] = file_name
            meta_row['sid'] = series_df['sid']
            meta_row['sub_sid'] = file_name.split('_series')[0]
            meta_row['start_t'] = series_df['t_x'].iloc[0]
            meta_row['end_t'] = series_df['t_x'].iloc[-1]
            meta_row['vid'] = series_df['vid']
            meta_row['aug.road_type'] = series_df['aug.road_type']
            meta_data_df_dict[idx] = meta_row.iloc[0]
            # meta_data_df = meta_data_df.append(meta_row.iloc[0])
    meta_data_values = list(meta_data_df_dict.values())
    meta_data_df = meta_data_df.append(meta_data_values)
    meta_data_df.to_csv(meta_data_df_result_path, index=False)


def under_or_over_sample_set(df, target_col, is_over):
    # prepare dict of class name and number of istances
    print('Over sampling with strategy')
    value_counts = df[target_col].value_counts()
    major_class_idx = value_counts.argmax()
    major_size = value_counts[major_class_idx]
    major_class_name = value_counts.index[major_class_idx]

    minor_class_idx = value_counts.argmin()
    minor_size = value_counts[minor_class_idx]
    minor_class_name = value_counts.index[minor_class_idx]

    class_names = list(value_counts.index)
    class_names.remove(minor_class_name)
    class_names.remove(major_class_name)

    X = df.drop([target_col], axis=1)
    y = df[target_col]
    if is_over == 'over':
        sampling_strategy_over = {major_class_name: major_size, minor_class_name: major_size}
        for remaining_class in class_names:
            sampling_strategy_over[remaining_class] = major_size

        ros = RandomOverSampler(sampling_strategy=sampling_strategy_over)
        sampler = ros
        X_resampled, y_resampled = sampler.fit_sample(X, y)
        # rus = RandomOverSampler()

    elif is_over == 'under':
        print('Under sampling sets')
        rus = RandomUnderSampler()
        sampler = rus
        X_resampled, y_resampled = sampler.fit_sample(X, y)


    elif is_over == 'both':
        # over sampling
        sampling_strategy_over = {major_class_name: major_size, minor_class_name: int(minor_size * 2.5)}
        for remaining_class in class_names:
            sampling_strategy_over[remaining_class] = int(value_counts[remaining_class])
        ros = RandomOverSampler(sampling_strategy=sampling_strategy_over)

        # under sampling
        major_size_denominator = 3
        sampling_strategy_under = {major_class_name: int(major_size / major_size_denominator),
                                   minor_class_name: int(minor_size * 2.5)}
        for remaining_class in class_names:
            sampling_strategy_under[remaining_class] = int(major_size / major_size_denominator)
        rus = RandomUnderSampler(sampling_strategy_under)

        pipeline = Pipeline(steps=[('o', ros), ('u', rus)])
        X_resampled, y_resampled = pipeline.fit_resample(X, y)

    else:
        print("no sampling method selected, please select: over/under/both")

    df_balanced = pd.DataFrame(X_resampled)
    df_balanced.columns = df.columns.drop([target_col])
    df_balanced[target_col] = y_resampled
    return df_balanced

    # values_counts = df[target_col].value_counts(normalize=True)
    # class_to_under = values_counts.index[0]
    # # df
    # frac = values_counts[0] - values_counts[-1]
    # df = df.drop(df[df[target_col] == class_to_under].sample(frac=0.85).index)
    # return df

    # return df_subset
    # # under sampling train data
    # print('under sampling')
    # d = defaultdict(LabelEncoder)
    # df_transformed = df.apply(lambda x: d[x.name].fit_transform(x))
    # print('shape before under sampling {}'.format(df.shape))
    # undersample = NearMiss(version=2, n_neighbors=3)
    # X_over, y_over = undersample.fit_resample(df_transformed.drop([target_col], axis=1), df_transformed[target_col])
    # print('shape before under sampling {}'.format(X_over.shape))
    # df_balanced = pd.DataFrame(X_over)
    # df_balanced[target_col] = y_over
    # df_balanced.columns = df.columns
    #
    # df_balanced = df_balanced.apply(lambda x: d[x.name].inverse_transform(x))
    # return df_balanced


def get_meta_data(meta_data_dir_path, meta_data_file_name_prefix, should_parse_dates=False):
    meta_data_files = os.listdir(meta_data_dir_path)
    meta_df = pd.DataFrame([])
    for idx, meta_file_name in enumerate(meta_data_files):
        if meta_data_file_name_prefix in meta_file_name:
            print('appending meta file {} out of {}'.format(idx + 1, len(meta_data_files)))
            if should_parse_dates:
                current_meta_df = pd.read_csv(meta_data_dir_path + meta_file_name)
            else:
                current_meta_df = pd.read_csv(meta_data_dir_path + meta_file_name)
            meta_df = meta_df.append(current_meta_df)
    meta_df.reset_index(inplace=True, drop=True)
    return meta_df


def create_partitioning(meta_data_file_name_prefix,
                        meta_data_dir_path,
                        partitioning_col,
                        target_col,
                        test_set_size,
                        val_set_size,
                        results_path,
                        keep_classes_list,
                        should_under_sample=False,
                        under_sample_sets=['train'],
                        sample_frac=1,
                        over_under_type=False,
                        seed=43,
                        filter_rpm=False):
    print('partitioning col is {}'.format(partitioning_col))
    meta_df = get_meta_data(meta_data_dir_path, meta_data_file_name_prefix)
    number_of_groups = len(meta_df[partitioning_col].unique())
    print('\n##########Number of {}: {}##############'.format(partitioning_col, number_of_groups))
    print('distribution:\n{} \n'.format(meta_df[target_col].value_counts()))
    print('distribution:\n{} \n'.format(meta_df[target_col].value_counts(normalize=True)))

    if keep_classes_list == 'all':
        print('keeping all classes')
    else:
        meta_df = meta_df.loc[meta_df[target_col].isin(keep_classes_list)]

    # train-val test split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_set_size)
    X = meta_df.drop(target_col, axis=1)
    y = meta_df[target_col]
    groups = meta_df[partitioning_col]
    for train_val_idx, test_idx in gss.split(X, y, groups):
        print("")

    train_val_set = meta_df.iloc[train_val_idx]



    train_val_set.reset_index(drop=True, inplace=True)

    test_set = meta_df.iloc[test_idx]
    print('test size with rpm filled: {}'.format(len(test_set)))

    if filter_rpm:

        # # remove rpm cols that were filled
        test_set = test_set.loc[test_set['aug.is_rpm_filled'] == False]
        print('test size without rpm filled: {}'.format(len(test_set)))



        sids_in_test_residential = test_set.loc[test_set['aug.road_type'] == 'residential']['sid'].value_counts()
        if len(sids_in_test_residential) == 0:
            # manually take samples from train_val

        # print('sids in test before switching: {}'.format(sids_in_test.index[0]))

        # # perform switching
            sid_with_rpm_residential_1 = 98516201
            sid_with_rpm_residential_2 = 68712620

            import random
            sid_to_still = random.choice([sid_with_rpm_residential_1, sid_with_rpm_residential_2])
            # replace sid

            train_set_copy = train_val_set.copy()
            # remove sid from train_val
            train_val_set = train_val_set.loc[train_val_set['sid'] != sid_to_still]
            add_to_test = train_set_copy.loc[train_set_copy['sid'] == sid_to_still]
            # add sid to test
            test_set = test_set.append(add_to_test)


            test_set.reset_index(drop=True, inplace=True)
            sids_in_test = test_set.loc[test_set['aug.road_type'] == 'residential']['sid'].value_counts()
            print('sids in test after switching: {}'.format(sids_in_test.index[0]))




    # train val split
    X = train_val_set
    y = train_val_set[target_col]
    groups = train_val_set[partitioning_col]
    val_gss = GroupShuffleSplit(n_splits=1, test_size=val_set_size)
    for train_idx, val_idx in val_gss.split(X, y, groups):
        print("")

    train_set = train_val_set.iloc[train_idx]
    train_set.reset_index(drop=True, inplace=True)
    val_set = train_val_set.iloc[val_idx]
    val_set.reset_index(drop=True, inplace=True)

    sids_in_train_total = train_set[partitioning_col].unique()
    sids_in_test_total = test_set[partitioning_col].unique()
    sids_in_val_total = val_set[partitioning_col].unique()

    print('train_set sids: {}'.format(len(sids_in_train_total)))
    print('test_set sids: {}'.format(len(sids_in_test_total)))
    print('val_set sids: {}'.format(len(sids_in_val_total)))

    sids_in_train_resi = train_set[['aug.road_type', 'sid']][train_set[['aug.road_type', 'sid']]['aug.road_type'] == 'residential']['sid'].value_counts()
    sids_in_test_resi = test_set[['aug.road_type', 'sid']][test_set[['aug.road_type', 'sid']]['aug.road_type'] == 'residential']['sid'].value_counts()
    sids_in_val_resi = val_set[['aug.road_type', 'sid']][val_set[['aug.road_type', 'sid']]['aug.road_type'] == 'residential']['sid'].value_counts()

    print('train_set residential sids: {}'.format(len(sids_in_train_resi)))
    print('test_set residential sids: {}'.format(len(sids_in_test_resi)))
    print('val_set residential sids: {}'.format(len(sids_in_val_resi)))
    # train test data
    # train_set = meta_df[meta_df[partitioning_col].isin(train_set_ids[partitioning_col])]
    # test_set = meta_df[meta_df[partitioning_col].isin(test_set_ids[partitioning_col])]
    # val_set = meta_df[meta_df[partitioning_col].isin(val_set_ids[partitioning_col])]

    # validate no leakage
    train_set_ids_unique = set(train_set[partitioning_col])
    test_set_ids_unique = set(test_set[partitioning_col])
    val_set_ids_unique = set(val_set[partitioning_col])

    common_file_ids_train_test = train_set_ids_unique.intersection(test_set_ids_unique)
    common_file_ids_train_val = train_set_ids_unique.intersection(val_set_ids_unique)

    len_common_train_test = len(common_file_ids_train_test)
    len_common_train_val = len(common_file_ids_train_val)
    print("\nsplitted train test sets common ids size: {}".format(len_common_train_test))
    print("splitted train val sets common ids size: {}".format(len_common_train_val))

    print_partioning_stat(train_set, test_set, val_set, target_col, title='Befor under sampling')

    # create partitioning dict
    if sample_frac < 1:
        train_set_freq = train_set.groupby(target_col)[target_col].transform('count')
        test_set_freq = test_set.groupby(target_col)[target_col].transform('count')
        val_set_freq = val_set.groupby(target_col)[target_col].transform('count')

        # train_set = train_set.sample(frac=sample_frac, weights=train_set_freq)
        # test_set = test_set.sample(frac=sample_frac, weights=test_set_freq)
        # val_set = val_set.sample(frac=sample_frac, weights=val_set_freq)
        train_set = train_set.sample(frac=sample_frac)
        test_set = test_set.sample(frac=sample_frac)
        val_set = val_set.sample(frac=sample_frac)

        print_partioning_stat(train_set, test_set, val_set, target_col, 'After sample fraq')

    if should_under_sample:
        if 'train' in under_sample_sets:
            train_set = under_or_over_sample_set(train_set, target_col, is_over=over_under_type)
        if 'val' in under_sample_sets:
            val_set = under_or_over_sample_set(val_set, target_col, is_over=over_under_type)

    # update meta data before saving
    train_set_ids_unique = set(train_set[partitioning_col])
    test_set_ids_unique = set(test_set[partitioning_col])
    val_set_ids_unique = set(val_set[partitioning_col])

    train_set_ids = list(train_set[partitioning_col])

    partitioning_dict = {'train': list(train_set[FILE_COL].values), 'test': list(test_set[FILE_COL]),
                         'val': list(val_set[FILE_COL]),
                         'train_partitioning_col': train_set_ids_unique,
                         'test_partitioning_col': test_set_ids_unique,
                         'val_partitioning_col': val_set_ids_unique,
                         'train_partitioning_col_list': train_set_ids}

    # print("train size: {}\ntest size: {}\nval size: {}".format(len(train_set), len(test_set), len(val_set)))

    pickle_util.save_obj(partitioning_dict, results_path)

    print_partioning_stat(train_set, test_set, val_set, target_col, 'After under sampling')
    return partitioning_dict


def create_partitioning_based_on_time(meta_data_dir_path, meta_data_file_name_prefix, target_col, sample_fraq=1):
    meta_df = get_meta_data(meta_data_dir_path, meta_data_file_name_prefix)
    meta_freq = meta_df.groupby(target_col)[target_col].transform('count')
    meta_df_after_sample_and_sort = meta_df.sample(frac=sample_fraq, weights=meta_freq)
    size = meta_df_after_sample_and_sort.shape[0]
    meta_df_after_sample_and_sort.sort_values(by=['start_t'], inplace=True)
    train_last_idx = int(size / 2)
    val_last_idx = train_last_idx + int(train_last_idx / 2)

    train_set = meta_df_after_sample_and_sort.iloc[0:train_last_idx]
    val_set = meta_df_after_sample_and_sort.iloc[train_last_idx:val_last_idx]
    test_set = meta_df_after_sample_and_sort.iloc[val_last_idx:]

    partitioning_dict = {'train': list(train_set[FILE_COL].values), 'test': list(test_set[FILE_COL]),
                         'val': list(val_set[FILE_COL])}

    return partitioning_dict


def print_partioning_stat(train_set, test_set, val_set, target_col, title):
    print("\n\n########### {} ############".format(title))
    print('train set size {}'.format(len(train_set)))
    print('test set size {}'.format(len(test_set)))
    print('val set size {}'.format(len(val_set)))
    print('\n-----Distributions------')
    print('train set distribution:\n{} \n'.format(train_set[target_col].value_counts()))
    print('test set distribution:\n{} \n'.format(test_set[target_col].value_counts()))
    print('val set distribution:\n{} \n'.format(val_set[target_col].value_counts()))
    print('train set distribution norm:\n{} \n'.format(train_set[target_col].value_counts(normalize=True)))
    print('test set distribution norm:\n{} \n'.format(test_set[target_col].value_counts(normalize=True)))
    print('val set distribution norm:\n{} \n'.format(val_set[target_col].value_counts(normalize=True)))

# partitioning_dict = create_partitioning(META_DATA_DF_PATH, 'sid', 'aug.road_type', 0.2, 0.1)
