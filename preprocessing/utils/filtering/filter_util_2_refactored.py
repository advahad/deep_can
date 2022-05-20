from datetime import timedelta

import pandas as pd


# create relevant sessions
def create_relevant_sessions(sessions_dir_path, session_file_name, relevant_sessions_path, timedelta_sec):
    session_df = pd.read_csv(sessions_dir_path + session_file_name, parse_dates=['t'])

    # separate to sub-sessions if needed
    deltas = session_df['t'].diff()[1:]
    gaps = deltas[deltas > timedelta(seconds=timedelta_sec)]
    if gaps.empty:
        velocity_missing = session_df['11'].isnull().all()
        if not velocity_missing:
            session_df.to_csv(relevant_sessions_path + session_file_name, index=False)
    else:
        indices = list(gaps.index) + [list(session_df.index)[-1] + 1]
        start_idx = 0
        for i, end_idx in enumerate(indices):
            sub_session_df = session_df.iloc[start_idx:end_idx]
            start_idx = end_idx

            velocity_missing = sub_session_df['11'].isnull().all()

            if not velocity_missing:
                session_id = session_file_name.split('.')[0]
                sub_session_df.to_csv(relevant_sessions_path + session_id + '_' + str(i) + '.csv',
                                      index=False)




