import numpy as np
import pandas as pd

# CAR_ID = "499"
RAW_DATA_PATH = "D:\\clones\\road_type_classifier\\data\\canbus\\canbus_joint_ordered_all.csv"
RAW_DATA_PATH = "../../data/canbus/canbus_joint_ordered_all.csv"
# RESULTS_PATH = "..\\..\\results\\flat\\canbus_joint_flattened_all_new.csv"
RESULTS_PATH_PREFIX = "../../results/flat/sessions/"

CHUNK_SIZE = 40000
# CANBUS_RAW_PATH = "../data/canbus/canbus_joint_ordered.csv"

def flat_canbus(orig_data_path, result_path_prefix):
    # canbus_columns = ['fid', 'eid', 'sid', 'oid', 't', 'vtype', 'f', 'vid', 'signalid']
    all_signals_in_canbus_data_path = '../sql/unique_signals.csv'
    relevant_canbus_columns = ['sid', 'fid', 't', 'vtype', 'f', 'vid', 'signalid']

    csv_stream = pd.read_csv(orig_data_path, sep=',', chunksize=CHUNK_SIZE, low_memory=True, encoding="ISO-8859-1",
                             error_bad_lines=False, usecols=relevant_canbus_columns, parse_dates=['t'])

    all_signals = ['11', '4483', '4229', '1158', '1159', '1160', '4487', '646',
                   '280', '281', '282', '843', '283', '156', '287',
                   '36', '37', '4133', '43', '175', '47', '3377',
                   '192', '962', '452', '326', '337', '101', '2804']
    all_signals_df = pd.read_csv(all_signals_in_canbus_data_path, index_col=[0],  dtype={'signalid': object})
    all_signals = list(all_signals_df['signalid'])
    merge_joint_colums = ['vid', 'sid', 't', 'fid', 'vtype']

    header = merge_joint_colums + all_signals
    header_df = pd.DataFrame([], columns=header)
    # TODO: try not to write separately
    # header_df.to_csv(result_path, encoding='utf-8', mode='a', header=True, index=False)
    sids_dict = {}
    for i, chunk in enumerate(csv_stream):
        # if i < 15:
        #     continue
        unique = chunk['sid'].unique()
        print((str(i)))
        chunk = chunk.set_index('t')
        # chunk = chunk[relevant_canbus_columns].set_index('t')
        # chunk = chunk.loc[chunk['sid'].isin(list(joint_sids.values))]
        chunk = chunk.loc['2010-01-01':'2019-07-01']
        if chunk.empty:
            print("empty")
            continue
        print("not empty")

        # session_df = pd.DataFrame([], header = header)
        current_sid = 0
        series = chunk.groupby(['sid', pd.Grouper(freq='100ms')], axis=0)
        for key, session_time_interval_rows in series:
            # print(len(session_time_interval_rows))
            if len(session_time_interval_rows) > 3:
                print(session_time_interval_rows)
            sid_key = key[0]
            # if session_time_interval_rows.shape[0] > 1:
            #     print(session_time_interval_rows)
            if sid_key in sids_dict:
                session_df = sids_dict[sid_key]
            else: # never saw this sid before
                if len(sids_dict) > 0: # check if not empty, if empty its the first session readed
                    for dict_key, df in sids_dict.items():
                        # save the last session in file and continue
                        df.to_csv(RESULTS_PATH_PREFIX + str(dict_key) + '.csv', index=False)
                    sids_dict = {}
                session_df = pd.DataFrame([], columns=header)
                sids_dict[sid_key] = session_df



            # print(res2.get_group(key), "\n\n")
            # all_signals_rows = series.get_group(key)
            empty = np.empty(shape=(1, len(header)))
            empty[:] = np.nan
            flatten_signals_row = pd.DataFrame(empty, columns=header)
            flatten_signals_row['vid'] = session_time_interval_rows.iloc[0]['vid']
            flatten_signals_row['sid'] = session_time_interval_rows.iloc[0]['sid']
            flatten_signals_row['t'] = session_time_interval_rows.index[0]
            flatten_signals_row['fid'] = session_time_interval_rows.iloc[0]['fid']
            flatten_signals_row['vtype'] = session_time_interval_rows.iloc[0]['vtype']
            # populate signals
            for index, signal_row in session_time_interval_rows.iterrows():
                signal_id = int(signal_row['signalid'])
                signal_value = signal_row['f']
                flatten_signals_row[str(signal_id)] = signal_value



            # joint_data_from_row = pd.DataFrame(all_signals_rows.iloc[0].drop(['signalid', 'f'])).T
            # # joint_data_from_row['t'] = all_signals_rows.index[0]
            # joint_data_from_row.insert(2, 't', all_signals_rows.index[0])
            # joint_data_from_row['tmp_key'] = 1
            # flatten_signals_row['tmp_key'] = 1
            # flatten_row_to_append = pd.merge(joint_data_from_row, flatten_signals_row, on='tmp_key')
            # flatten_row_to_append = flatten_row_to_append.drop('tmp_key', axis=1)
            session_df = session_df.append(flatten_signals_row, ignore_index=True, sort=False)
            if session_df.shape[1] > 34:
                print("wrong shape for sid: {}".format(sid_key))
            sids_dict[sid_key] = session_df
    # write last session
    if len(sids_dict) > 0:  # check if not empty, if empty its the first session readed
        for dict_key, df in sids_dict.items():
            # save the last session in file and continue
            # df.to_csv(RESULTS_PATH_PREFIX + str(dict_key) + '.csv', index=False)
            print('save df here')
    # session_df.to_csv(RESULTS_PATH_PREFIX + str(sids_dict) + '.csv', index=False)


flat_canbus(RAW_DATA_PATH, RESULTS_PATH_PREFIX)