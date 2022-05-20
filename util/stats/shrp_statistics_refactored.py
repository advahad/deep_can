import os
import pandas as pd


def calc_sessions_fullness(sessions_path, results_dir, velocity_col_name):
    results_df = pd.DataFrame([])
    sessions_files = os.listdir(sessions_path)
    for i, session_file_name in enumerate(sessions_files):
        session_df = pd.read_csv(sessions_path + session_file_name)
        if session_df[velocity_col_name].isnull().all():
            continue
        # total_missing = session_df.isnull().sum()
        percent_full = pd.DataFrame(100 - (session_df.isnull().sum() * 100 / len(session_df))).T
        sid = session_file_name.split('File_ID_')[1].split('_')[0]
        percent_full.insert(0, 'sid_id', sid)
        percent_full.insert(1, 'session_len', len(session_df))
        results_df = results_df.append(percent_full, ignore_index=True)

    results_df.to_csv(results_dir + 'sessions_fullness.csv', index=False)
    return results_df


# treat session as one instance for checking column missing values
def calc_columns_missing_values(results_dir):
    results_df = pd.read_csv(results_dir + 'sessions_fullness.csv')

    results_df = results_df.loc[results_df['session_len'] / 10 >= 2]
    col_percent_empty = pd.DataFrame((results_df == 0).sum() * 100 / len(results_df)).T
    col_percent_empty = col_percent_empty.T.sort_values(by=0)
    col_percent_empty.reset_index(inplace=True, drop=False)
    col_percent_empty.columns = pd.Series(['col_name', 'percent_empty'])
    col_percent_empty.to_csv(results_dir + 'percent_empty.csv', index=False)
    return col_percent_empty


def join_percent_empty_to_importance(percent_empty_dir_path, signals_importance_path):
    percent_empty = pd.read_csv(percent_empty_dir_path + 'percent_empty.csv')
    percent_empty = percent_empty.drop(['sid_id', 'session_len', 'vtti.timestamp'], axis=1)
    percent_empty = percent_empty.T
    percent_empty['percent_empty'] = percent_empty[0]
    percent_empty.drop(0, inplace=True, axis=1)

    percent_empty.reset_index(inplace=True)
    percent_empty['signal_id'] = percent_empty['index']
    # percent_empty['signal_id'] = pd.to_numeric(percent_empty['signal_id'])
    percent_empty.drop('index', inplace=True, axis=1)
    # percent_empty.rename(columns = {'0':'percent_empty'})
    cols = ['signal_id', 'description', 'importance']
    signals_importance = pd.read_csv(signals_importance_path, usecols=cols)
    print("")
    # in order to be able to merge with signal importance col
    signals_importance['signal_id'] = signals_importance['signal_id'].astype(str)
    merged = pd.merge(left=signals_importance, right=percent_empty, on='signal_id')
    merged = merged.sort_values('percent_empty')
    merged.to_csv(percent_empty_dir_path + 'percent_empty_and_signal_importance.csv', index=False)
