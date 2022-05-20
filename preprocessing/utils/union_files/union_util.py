import os
import pandas as pd
LABELED_SESSIONS_PATH = '../../results/labeled/relevant_sessions_full_labeled/'
RESULT_FILE_PATH = '../../results/labeled/union_labeld_data.csv'
labeled_sessions = os.listdir(LABELED_SESSIONS_PATH)



def merge_all_sessions_to_one_file():
    merged_df = pd.DataFrame([])
    for file_name in labeled_sessions:
        session_df = pd.read_csv(LABELED_SESSIONS_PATH + file_name)

        session_df.insert(2, 'sub_sid', file_name.split('.')[0])
        merged_df = merged_df.append(session_df)
        # print(merged_df.shape)
        # print(merged_df.columns)
    merged_df.to_csv(RESULT_FILE_PATH, index=False)

merge_all_sessions_to_one_file()