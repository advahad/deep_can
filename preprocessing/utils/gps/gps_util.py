import pandas as pd


# ROAD_TYPE_PATH = '../results/road_type/road_type_joint.csv'
# SEPARATE_ROAD_TYPE_PATH = '../results/road_type/road_type_joint_separate.csv'
#
# ALL_GPS_PATH = '../data/gps/all_gps.csv'
# SEPARATE_ALL_GPS_PATH = '../results/all_gps_separate.csv'


def _separate_dates(orig_file_path, des_file_path):
    df = pd.read_csv(orig_file_path, parse_dates=['t'])
    df['day'] = df['t'].apply(lambda x: x.date())
    df['hour'] = df['t'].apply(lambda x: x.hour)
    df['minute'] = df['t'].apply(lambda x: x.minute)
    df['second'] = df['t'].apply(lambda x: x.second)
    df.to_csv(des_file_path, index=False)


# load gps data
def _concat_gps_data_and_separate_dates(should_separate_dates):
    data_6 = pd.read_csv('..\\data\\gps\\6.csv', parse_dates=['t'])
    data_7 = pd.read_csv('..\\data\\gps\\7.csv', parse_dates=['t'])
    data_8 = pd.read_csv('..\\data\\gps\\8.csv', parse_dates=['t'])
    data_10 = pd.read_csv('..\\data\\gps\\10.csv', parse_dates=['t'])
    all_gps_data = pd.concat([data_6, data_7, data_8, data_10], ignore_index=True)
    if should_separate_dates:
        all_gps_data['day'] = all_gps_data['t'].apply(lambda x: x.date())
        all_gps_data['hour'] = all_gps_data['t'].apply(lambda x: x.hour)
        all_gps_data['minute'] = all_gps_data['t'].apply(lambda x: x.minute)
        all_gps_data['second'] = all_gps_data['t'].apply(lambda x: x.second)

    return all_gps_data


def _save_gps_as_one_file(file_name, should_separate_dates):
    all_gps_data = _concat_gps_data_and_separate_dates(should_separate_dates)
    all_gps_data.to_csv("..\\data\\gps\\" + file_name, encoding='utf-8', index=False)


def concat_separate_and_save_gps(result_path):
    _save_gps_as_one_file(result_path, should_separate_dates=True)


def save_all_unique_sid():
    all_gps_data = pd.read_csv('../../data/gps/all_gps.csv')
    all_gps_sid = all_gps_data['sid'].unique()
    pd.DataFrame(all_gps_sid, columns=['sid']).to_csv('../../data/gps/all_sid_in_gps.csv', index=False)

# save_all_unique_sid()
