import pandas as pd

COLS_TO_USE_PATH = 'relevant_cols.csv'


def fill_interpolate(df, col_name):
    res = df[[col_name]].interpolate(method='linear', inplace=False, limit_direction='both')
    return res

def fill_fix_value(df, col_name, val):
    res = df[[col_name]].fillna(val, inplace=False)
    return res



def  fill_empty_values_by_type_shrp(sessions_dir, session_file_name, results_dir, relevant_cols_path, on_change_only=True):
    cols_df_raw = pd.read_csv(relevant_cols_path)
    cols_df = cols_df_raw.loc[(cols_df_raw['col_type'] == 'signal') & (cols_df_raw['keep'] == 'yes')]
    if on_change_only:
        cols_df = cols_df.loc[cols_df['on_change'] == 'yes']
    cols_to_complete = cols_df[['col_name', 'fill_type_short', 'val_complete']]
    cols_in_results = cols_df_raw.loc[(cols_df_raw['keep'] == 'yes')]['col_name']

    # print("cols to complete are: {}".format(cols_to_complete['col_name']))
    session_df = pd.read_csv(sessions_dir + '/' + session_file_name, usecols=cols_in_results)
    for idx, row in cols_to_complete.iterrows():
        fill_type = row['fill_type_short']
        col_name = row['col_name']
        complete_rollback_val = cols_to_complete.loc[cols_to_complete['col_name'] == col_name]['val_complete'].values[0]
        if fill_type == 'IN':
            session_df[[col_name]] = fill_interpolate(session_df, col_name)
            # if no value in col to interpolate on

            session_df[[col_name]] = fill_fix_value(session_df, col_name, complete_rollback_val)

        elif fill_type == 'PAD':
            session_df[[col_name]] = session_df[col_name].fillna(method='ffill')
            session_df[[col_name]] = session_df[col_name].fillna(method='bfill')
            # if no value in col to pad on
            session_df[[col_name]] = fill_fix_value(session_df, col_name, complete_rollback_val)

            if session_df[col_name].isnull().all() is not session_df[col_name].isnull().any():
                print("check file {}".format(sessions_dir + '/' + session_file_name))


    session_df.to_csv(results_dir + '/' + session_file_name, index=False)
def  fill_empty_values_by_type(sessions_dir, session_file_name, results_dir, relevant_cols_path, on_change_only=True):
    cols_df_raw = pd.read_csv(relevant_cols_path)
    cols_df = cols_df_raw.loc[(cols_df_raw['col_type'] == 'signal') & (cols_df_raw['keep'] == 'yes')]
    if on_change_only:
        cols_df = cols_df.loc[cols_df['on_change'] == 'yes']
    cols_to_complete = cols_df[['col_name', 'fill_type_short', 'val_complete']]
    cols_in_results = cols_df_raw.loc[(cols_df_raw['keep'] == 'yes')]['col_name']

    # print("cols to complete are: {}".format(cols_to_complete['col_name']))
    session_df = pd.read_csv(sessions_dir + '/' + session_file_name, usecols=cols_in_results)
    for idx, row in cols_to_complete.iterrows():
        fill_type = row['fill_type_short']
        col_name = row['col_name']
        complete_rollback_val = cols_to_complete.loc[cols_to_complete['col_name'] == col_name]['val_complete'].values[0]
        if fill_type == 'IN':
            session_df[[col_name]] = fill_interpolate(session_df, col_name)
            # if no value in col to interpolate on
            session_df[[col_name]] = fill_fix_value(session_df, col_name, complete_rollback_val)

        elif fill_type == 'PAD':
            session_df[[col_name]] = session_df[col_name].fillna(method='ffill')
            session_df[[col_name]] = session_df[col_name].fillna(method='bfill')
            # if no value in col to pad on
            session_df[[col_name]] = fill_fix_value(session_df, col_name, complete_rollback_val)

            if session_df[col_name].isnull().all() is not session_df[col_name].isnull().any():
                print("check file {}".format(sessions_dir + '/' + session_file_name))


    session_df.to_csv(results_dir + '/' + session_file_name, index=False)


