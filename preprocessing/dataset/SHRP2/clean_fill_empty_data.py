import pandas as pd
import os

DATA_PATH = '../data_SHRP2/multi_task_timeseries/'
DATA_PATH = '../data_SHRP2/augmented_timeseries_one_roadtype_per_file/'
RELEVANT_COLUMNS_PATH = '../data_SHRP2/relevant_columns.csv'
CLEAN_DATA_PATH = '../data_SHRP2/clean_augmented_timeseries_multi/'
CLEAN_DATA_PATH = '../data_SHRP2/clean_augmented_timeseries_road_type_only_new/'


def fill_fix_value(df, col_name, val):
    res = df[[col_name]].fillna(val, inplace=False)
    return res


def fill_interpolate(df, col_name):
    res = df[[col_name]].interpolate(method='linear', inplace=False, limit_direction='both')
    return res

def fill_empty_values():
    relevant_columns_metadata = pd.read_csv(RELEVANT_COLUMNS_PATH)
    # relevant_columns_metadata = relevant_columns_metadata[0].tolist()
    relevant_cols_names = relevant_columns_metadata['col_name']

    files = os.listdir(DATA_PATH)
    empty_aug_files = list()
    for idx, name in enumerate(files):
        data_df = pd.read_csv(DATA_PATH + name)
        # filter out non relevant columns
        data_df = data_df[relevant_columns_metadata['col_name']]
        orig = data_df
        for curr_col in data_df.columns:
            # if curr_col == 'aug.road_type':
            #     if data_df['aug.road_type'].isnull().all():
            #         empty_aug_files.append(name)
            #         break
            details_row = relevant_columns_metadata.loc[relevant_columns_metadata['col_name'] == curr_col]
            fill_type = details_row['empty_fill_type'].values[0]
            if fill_type == 'fix':
                data_df[[curr_col]] = fill_fix_value(data_df, curr_col, details_row['val'].values[0])
            elif fill_type == 'interpolation':
                data_df[[curr_col]] = fill_interpolate(data_df, curr_col)
                data_df[[curr_col]] = fill_fix_value(data_df, curr_col, details_row['interpolate_rollback_val'].values[0])
            # TODO: better handling, not really padding
            elif fill_type == 'padding':
                freq_road_type = data_df[data_df['aug.road_type'].notnull()]['aug.road_type'].unique()
                common_val = details_row['val'].values[0]
                if len(freq_road_type) > 0:
                    common_val = freq_road_type[0]
                data_df[[curr_col]] = fill_fix_value(data_df, curr_col, common_val)

        data_df.to_csv(CLEAN_DATA_PATH + name, index=False)
    empty_aug_df = pd.DataFrame([empty_aug_files])
    empty_aug_df.to_csv("empty_road_type.csv", index=False)
