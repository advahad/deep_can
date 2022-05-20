import random as rnd
import time

import pandas as pd
import requests
from geopy.geocoders import Here
from geopy.geocoders import Nominatim


def road_request(lat, long, radius):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
            [out:json];
            (
              way(around:%s,%s,%s)
              [highway~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"];
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
    t1 = time.time()
    rr = road_request(it_lat, it_long, it_rad)
    t2 = time.time()
    # print('http request took ' + str((t2-t1)*1000) + ' ms')
    t1 = time.time()
    result = dictionary_majority(get_road_details(rr))
    t2 = time.time()
    # print('majority dict took ' + str((t2 - t1) * 1000) + ' ms')
    while result == '' and it_rad < 15:
        try_count += 1
        it_rad += 5
        rr = road_request(it_lat, it_long, it_rad)
        result = dictionary_majority(get_road_details(rr))
    # print('request count :' + str(try_count))
    t1 = time.time()
    ms = get_road_maxSpeed(rr)
    t2 = time.time()
    # print('json max speed took ' + str((t2 - t1) * 1000) + ' ms')
    t1 = time.time()
    rl = get_roads_lane_number(rr)
    t2 = time.time()
    # print('json road lanes took ' + str((t2 - t1) * 1000) + ' ms')
    sw = extract_surroundings_ways(rr)
    return result, try_count, ms, rl, sw

def extract_surroundings_ways(req):
    road_dict = get_road_details(req)
    sum = 0
    for key in road_dict.keys():
        sum += int(road_dict[key])
    return sum

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


# fearure is one of : ['road_max_speed', 'road_type']
def visualized_output(path_agu, path_gps, to_write_path, feature):
    agumantion_df = pd.read_csv(path_agu, dtype=str, index_col='i')
    if feature == 'road_max_speed':
        indexes = agumantion_df.loc[agumantion_df['road_max_speed'] == 'NA']
    elif feature == 'road_type':
        indexes = agumantion_df.loc[pd.isna(agumantion_df['road_type'])]
    index_list = list(indexes['index'])
    gps_df = pd.read_csv(path_gps, dtype=str)
    to_excel_df = pd.DataFrame(columns=['lat', 'lon', 'shape', 'color'])
    for i, idx in enumerate(agumantion_df['index']):
        gps_row = gps_df.loc[int(idx)]
        s = pd.Series([gps_row['lat'], gps_row['lon'], 'circle2', 'red' if index_list.__contains__(idx) else 'green'],
                      index=['lat', 'lon', 'shape', 'color'])
        to_excel_df = to_excel_df.append(s, ignore_index=True)

    to_excel_df.to_csv(to_write_path)


def find_city_nominatim(lat, lon):
    geolocator = Nominatim(timeout=10, user_agent='bgu_finder')
    location = geolocator.reverse(str(lat) + "," + str(lon))
    try:
        ans = location.raw['address']
        if 'city' in ans.keys():
            return ans['city']
        elif 'town' in ans.keys():
            return ans['town']
        elif 'village' in ans.keys():
            return ans['village']
        else:
            return 'NOT FOUND'
    except:
        return 'NOT FOUND'


def find_city_Here(lat, lon):
    time.sleep(1)
    geolocator = Here(app_id='0gMBtS2Wvh6ZeUE5oTkR', app_code='8_DflCJWm7e_xuf0sFb0dw', timeout=10,
                      user_agent='bgu_finder')
    location = geolocator.reverse(str(lat) + "," + str(lon))
    try:
        ans = location.raw['Location']['Address']
        return ans['City']
    except:
        return 'NOT FOUND'


def color_func(str1, str2):
    if str1 == 'NOT FOUND' and str2 == 'NOT FOUND':
        return 'blue'
    elif str1 != 'NOT FOUND' and str2 == 'NOT FOUND':
        return 'green'
    elif str1 == 'NOT FOUND' and str2 != 'NOT FOUND':
        return 'red'
    elif str1 != 'NOT FOUND' and str2 != 'NOT FOUND':
        return 'yellow'


def get_vid():
    df_gps = pd.read_csv(
        "C:\\Users\\odedblu\\Desktop\\drivers\\part-00008-f2190756-9ceb-4366-ad34-5b22de531e15-c000.csv")
    df_canbus = pd.read_csv("C:\\Users\\odedblu\\Desktop\\drivers\\dfsavename.csv", chunksize=100000)
    keys = set(df_gps['sid'])
    session_driver_dict = {}
    i = 0
    for chunk in df_canbus:
        chunk_keys = set(chunk['sid'])
        intersection = keys.intersection(chunk_keys)
        print('chunk:' + str(i) + ', intersection size:' + str(len(intersection)) + ',done:' + "%.2f" % (
                    (i / 3142) * 100) + '%')
        if len(intersection) != 0:
            for sesstion in intersection:
                session_driver_dict[str(sesstion)] = chunk.loc[chunk['sid'] == sesstion].iloc[0]['vid']
        if len(keys) == len(session_driver_dict):
            break
        i += 1
    df = pd.DataFrame.from_dict(session_driver_dict, orient='index', columns=['vid'])
    df.to_csv("C:\\Users\\odedblu\\Desktop\\drivers\\dict_2str.csv")


def vid_per_city():
    df = pd.read_csv("C:\\Users\\odedblu\\Desktop\\points_to_cities_5000_00008.csv")
    cities = set(df['Here'])
    dict = {}
    for city in cities:
        dict[city] = len(set(df.loc[df['Here'] == city]['vid']))
    ans = pd.DataFrame.from_dict(dict, orient='index')
    ans.to_csv("C:\\Users\\odedblu\\Desktop\\cities_to_vid.csv")