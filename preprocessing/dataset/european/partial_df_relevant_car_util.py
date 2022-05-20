import pandas as pd

CAR_ID = "499"

def create_partial_gps_df(car_id):
    all_gps_data = pd.read_csv("..\\data\\gps\\all_gps.csv", encoding='utf-8')
    car_sid_values = pd.read_csv("..\\data\\distinct_sid_car_" + car_id + ".csv", encoding='utf-8')

    car_gps_data = all_gps_data.loc[all_gps_data['sid'].isin(car_sid_values.values)]
    joint_sids = pd.DataFrame(car_gps_data['sid'].unique(), columns=['sid'])
    # car_gps_data.to_csv("..\\data\\gps\\car_" + car_id + "_gps_data.csv", encoding='utf-8', index=False)
    joint_sids.to_csv("..\\data\\joint_sid_gps_canbus_car_" + car_id + ".csv",  encoding='utf-8', index=False)




create_partial_gps_df(CAR_ID)