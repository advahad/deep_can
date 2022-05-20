import pandas as pd
import os
import shutil


SRC_PATH = '../data_SHRP2/augmented_timeseries/'
DEST_PATH = '../data_SHRP2/no_gps_timeseries/'

files = os.listdir(SRC_PATH)

for idx, file_name in enumerate(files):
    src = SRC_PATH + file_name

    df = pd.read_csv(src)
    if df['aug.road_type'].isnull().all():
        dest = DEST_PATH + file_name
        shutil.move(src, dest)
