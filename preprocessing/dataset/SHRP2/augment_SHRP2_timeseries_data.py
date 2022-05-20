import math
import os
import pandas as pd
from data_augmentation import augmentation_util

BASELINES_PATH = '../data_SHRP2/timeseries/'
AUGMENTED_BASELINE_PATH = '../data_SHRP2/augmented_timeseries/'
NO_GPS_BASELINE_PATH = '../data_SHRP2/no_gps_timeseries/'


def augment_data():
    files = os.listdir(BASELINES_PATH)
    for idx, name in enumerate(files):
        current_file_df = pd.read_csv(BASELINES_PATH + name)

        # in case there is no gps values in dataframe
        if current_file_df['vtti.latitude'].isnull().all() or current_file_df['vtti.longitude'].isnull().all():
            current_file_df.to_csv(NO_GPS_BASELINE_PATH + name, index=False)
            break

        print("Augmenting file " + name + " at index " + str(idx))
        for i, row in current_file_df.iterrows():
            road_type = max_speed = lanes = 'NA'
            try_count = lanes = radius_roads = math.nan
            lat = row['vtti.latitude']
            long = row['vtti.longitude']
            # if gps data is not empty, get road type
            current_file_df.at[i, 'aug.road_type'] = ''
            current_file_df.at[i, 'aug.try_count'] = ''
            current_file_df.at[i, 'aug.max_speed'] = ''
            current_file_df.at[i, 'aug.lanes'] = ''
            current_file_df.at[i, 'aug.radius_roads'] = ''

            if pd.isnull(lat) or pd.isnull(long):
                continue
            # get augmentation from api
            try:
                igrd_res = augmentation_util.iterative_get_road_details(float(lat), float(long))
                response_lan = len(igrd_res)
                road_type = igrd_res[0]
                try_count = igrd_res[1]
                if response_lan > 4:
                    max_speed = igrd_res[2]
                    lanes = igrd_res[3]
                    radius_roads = igrd_res[4]
                elif response_lan > 3:
                    max_speed = igrd_res[2]
                    lanes = igrd_res[3]
                elif response_lan > 2:
                    max_speed = igrd_res[2]

            except Exception as e:
                print(e)
            try:
                radius_roads = augmentation_util.road_total_surroundings_ways(float(lat), float(long), 300)
            except Exception as e:
                print(e)

            current_file_df.at[i, 'aug.road_type'] = '' if road_type == 'NA' else road_type
            current_file_df.at[i, 'aug.try_count'] = '' if try_count == 'NA' else try_count
            current_file_df.at[i, 'aug.max_speed'] = '' if max_speed == 'NA' else max_speed
            current_file_df.at[i, 'aug.lanes'] = '' if lanes == 'NA' else lanes
            current_file_df.at[i, 'aug.radius_roads'] = '' if radius_roads == 'NA' else radius_roads

        current_file_df.to_csv(AUGMENTED_BASELINE_PATH + name, index=False)

        print("Done with file " + name)
