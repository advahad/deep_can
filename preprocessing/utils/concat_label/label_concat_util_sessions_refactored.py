import pandas as pd
import os


def concat_label_to_can_bus(relevant_sessions_path_full, session_file_name, augmented_gps_data_path, result_path,
                            signals_path, sub_sessions_stat):
    gps_files = os.listdir(augmented_gps_data_path)

    cols_df = pd.read_csv(signals_path)
    all_signals = list(cols_df.loc[(cols_df['col_type'] == 'signal')]['col_name'])

    session_df = pd.read_csv(relevant_sessions_path_full + session_file_name, parse_dates=['t'])

    # find matching gps file
    gps_session_file_name = session_file_name.split('_')[0]
    if not '.' in gps_session_file_name:
        gps_session_file_name = gps_session_file_name + '.csv'

    if gps_session_file_name in gps_files:

        gps_road_type_df = pd.read_csv(augmented_gps_data_path + gps_session_file_name)

        try:
            # chunk['t_time'] = pd.to_datetime(chunk['t'], format=CANBUS_DATE_FORMAT)
            session_df['day'] = session_df['t'].apply(lambda x: str(x.date()))
            session_df['hour'] = session_df['t'].apply(lambda x: x.hour)
            session_df['minute'] = session_df['t'].apply(lambda x: x.minute)
            session_df['second'] = session_df['t'].apply(lambda x: x.second)

            merged = pd.merge(session_df, gps_road_type_df, 'inner', left_on=['sid', 'day', 'hour', 'minute', 'second'],
                              right_on=['sid', 'day', 'hour', 'minute', 'second'])


        except:
            print("skipped file " + str(session_file_name))
            return

        # merged['velocity_diff_val'] = merged['f'] - merged['v']

        if merged.empty:
            print("no merge")
        else:
            merged = merged[['vid', 'sid', 'fid_x', 't_x'] + all_signals + ['road_type']]
            all_nan = merged['road_type'].isnull().all()
            try:
                all_NA = (merged['road_type'] == 'NA').all()
            except:
                all_NA = False
            some_nan = merged['road_type'].isnull().any()
            try:
                some_NA = (merged['road_type'] == 'NA').any()
            except:
                some_NA = False

            if all_nan:
                sub_sessions_stat.road_type_nan_sessions.append(gps_session_file_name)
            elif some_nan:
                sub_sessions_stat.partial_road_type_nan_sessions.append(gps_session_file_name)

            if all_NA:
                sub_sessions_stat.road_type_NA_sessions.append(gps_session_file_name)
            elif some_NA:
                sub_sessions_stat.partial_road_type_NA_sessions.append(gps_session_file_name)

            if all_nan or all_NA:
                print('not saving {}'.format(session_file_name))
            else:
                merged.to_csv(result_path + session_file_name, encoding='utf-8', index=False)
