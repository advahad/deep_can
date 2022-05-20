import math
import random as rnd

import numpy as np
import pandas as pd
import requests

ALL_GPS_SEPARATE_PATH = '../data/gps/all_gps_data_seperate_time.csv'
AUGMENTED_GPS_SEPARATE_DIR = '../results/road_type/sessions/'


def road_request(lat, long, radius):
    # overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_url = "https://overpass.kumi.systems/api/interpreter"
    overpass_query = """
            [out:json];
            (
            way(around:%s,%s,%s)[highway~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"];
            >;);out;
        """
    query = overpass_query % (str(radius), str(lat), str(long))
    response = requests.get(overpass_url,
                            params={'data': query})
    data = response.json()
    return data.get('elements')


def get_road_details(road_req):
    road_type_dictionary = {}
    for x in road_req:
        if x['type'] == 'way':
            key = str(x['tags']['highway'])
            if road_type_dictionary.__contains__(key):
                road_type_dictionary[key] += 1
            else:
                road_type_dictionary[key] = 1
    return road_type_dictionary


def get_road_maxSpeed(road_req):
    maxspeed = 'NA'
    for x in road_req:
        if x['type'] == 'way':
            try:
                maxspeed = str(x['tags']['maxspeed'])
            except KeyError:
                maxspeed = 'NA'
    return maxspeed


def dictionary_majority(dict):
    if dict:
        return max(dict, key=dict.get)
    else:
        return ''


def iterative_get_road_details(it_lat, it_long):
    it_rad = 5
    try_count = 1
    response = road_request(it_lat, it_long, it_rad)
    result = dictionary_majority(get_road_details(response))
    # print('majority dict took ' + str((t2 - t1) * 1000) + ' ms')
    while result == '' and it_rad < 15:
        try_count += 1
        it_rad += 5
        response = road_request(it_lat, it_long, it_rad)
        result = dictionary_majority(get_road_details(response))
    # print('request count :' + str(try_count))
    max_speed = get_road_maxSpeed(response)
    # print('json max speed took ' + str((t2 - t1) * 1000) + ' ms')
    lane_nmber = get_roads_lane_number(response)
    # print('json road lanes took ' + str((t2 - t1) * 1000) + ' ms')
    return result, try_count, max_speed, lane_nmber


def road_total_surroundings_ways(lat, long, rad):
    new_req = road_request(lat, long, rad)
    road_dict = get_road_details(new_req)
    sum = 0
    for key in road_dict.keys():
        sum += int(road_dict[key])
    return sum


def get_random(min, max, amount):
    random_number_list = []
    i = 0
    while i < amount:
        rnd_to_add = rnd.randint(min, max + 1)
        if not random_number_list.__contains__(rnd_to_add):
            random_number_list.append(rnd_to_add)
            i += 1
    return random_number_list


def get_roads_lane_number(road_req):
    lane_num = 'NA'
    for x in road_req:
        if x['type'] == 'way':
            try:
                lane_num = str(x['tags']['lanes'])
            except KeyError:
                lane_num = 'NA'
    return lane_num


def augment_gps_data(gps_data_path, result_path):
    orig_data_cols = ['sid', 'fid', 'oid', 'lon', 'lat', 'alt', 't', 'v', 'day',
                      'hour', 'minute', 'second']
    raw_gps_df = pd.read_csv(gps_data_path, usecols=orig_data_cols)
    sids_to_augment_df = pd.read_csv('../util/sid_to_augment.csv')
    sids_to_augment = list(sids_to_augment_df['sid'])

    raw_gps_df = raw_gps_df.loc[raw_gps_df['sid'].isin(sids_to_augment)]

    aug_columns = ['road_type', 'road_max_speed', 'lanes']

    all_cols = orig_data_cols + aug_columns

    sid_groups = raw_gps_df.groupby('sid')
    num_of_sids = len(sid_groups)
    num_of_sids_in_worker = math.ceil(num_of_sids / 4)
    cut_1 = num_of_sids_in_worker
    cut_2 = num_of_sids_in_worker*2
    cut_3 = num_of_sids_in_worker*3

    print("total number of sids to augment: {}".format(num_of_sids))
    sid_idx = 0
    print("sid intervals: {} to {}".format(cut_1, cut_2))
    for sid_group_name, sid_group_df in sid_groups:
        if cut_1 <= sid_idx < cut_2:
            print("start augmenting idx : {}, sid: {}, len: {}".format(sid_idx, sid_group_name, len(sid_group_df)))
            sid_group_augmented_df = pd.DataFrame([])
            augmented_data_dict = {}
            for i, row in sid_group_df.iterrows():
                # t1_start = time.time()
                try:
                    # req_t_start = time.time()
                    igrd_res = iterative_get_road_details(float(row['lat']), float(row['lon']))
                    # req_t_end = time.time()
                    # print('http request took ' + str(req_t_end - req_t_start) + ' sec')

                    road_type = igrd_res[0]
                except:
                    road_type = 'NA'
                try:
                    max_speed = igrd_res[2]
                except:
                    max_speed = 'NA'
                try:
                    lanes = igrd_res[3]
                except:
                    lanes = 'NA'

                empty = np.empty(shape=(1, len(all_cols)))
                empty[:] = np.nan
                res_row = pd.DataFrame(empty, columns=all_cols)

                for col_name in orig_data_cols:
                    res_row[col_name] = row[col_name]

                res_row['road_type'] = road_type
                res_row['road_max_speed'] = max_speed
                # res_row['300M_radius'] = radius_roads
                res_row['lanes'] = lanes
                augmented_data_dict[i] = res_row


            sid_group_augmented_df = sid_group_augmented_df.append(list(augmented_data_dict.values()))

            sid_group_augmented_df.to_csv(result_path + str(sid_group_name) + '.csv', index=False)
            print("done augmenting sid: {}".format(sid_group_name))

        sid_idx += 1


augment_gps_data(ALL_GPS_SEPARATE_PATH, AUGMENTED_GPS_SEPARATE_DIR)
