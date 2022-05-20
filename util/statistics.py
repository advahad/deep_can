import pandas as pd
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import matplotlib.pyplot as plt



TARGET = 'aug.road_type'
seq_len = 30
step = 30
data_path = '..\\data_SHRP2\\windows\\all_targets\\labeled_sequence_win_' + str(step) + '_seq_' + str(seq_len) \
            + '_new_multi_no_clean.csv'
# data_path = '..\\data_SHRP2\\windows\\aug.road_type\\labeled_sequence_win_10_seq_30.csv'




def disply_hist(data, col):
    print(col)
    ax = data.hist(column=col, bins=25, grid=False, figsize=(12, 8), color='#86bf91', zorder=2, rwidth=0.9)
    ax = ax[0]
    for x in ax:

        # Despine
        x.spines['right'].set_visible(False)
        x.spines['top'].set_visible(False)
        x.spines['left'].set_visible(False)

        # Switch off ticks
        x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off",
                      labelleft="on")

        # Draw horizontal axis lines
        vals = x.get_yticks()
        for tick in vals:
            x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

        # Remove title
        # x.set_title()

        # Set x-axis label
        x.set_xlabel("", labelpad=20, weight='bold', size=12)

        # Set y-axis label
        x.set_ylabel("", labelpad=20, weight='bold', size=12)

        # Format y-axis label
        x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

def disply_hist_1(data, coll):
    # An "interface" to matplotlib.axes.Axes.hist() method
    import matplotlib.pyplot as plt
    d = data[col]
    n, bins, patches = plt.hist(x=d.dropna().values, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(col)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

data = pd.read_csv(data_path)


fig, ax = plt.subplots(1,1,figsize=(26,8))
tmp = pd.DataFrame(data.groupby(['file_id', 'aug.road_type'])['series_num'].count().reset_index())
tmp = tmp[0:30]
m = tmp.pivot(index='aug.road_type', columns='file_id', values='series_num')
s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")
s.set_title('Number of road_type category per file_id', size=16)
plt.show()

# print(data.columns)
# print(data.describe())

import matplotlib.pyplot as plt
# pd.options.display.mpl_style = 'default'
# data.boxplot()



columns = data.columns

part_1 = ['vtti.accel_x',
       'vtti.accel_y', 'vtti.accel_z', 'vtti.cruise_state',
       'vtti.engine_rpm_instant', 'vtti.gyro_x', 'vtti.gyro_y', 'vtti.gyro_z',
       'vtti.headlight', 'vtti.light_level', 'vtti.pedal_brake_state',
       'vtti.pedal_gas_position', 'vtti.prndl', 'vtti.seatbelt_driver']
part_2 = ['vtti.speed_network', 'vtti.steering_wheel_position',
       'vtti.temperature_interior', 'vtti.traction_control_state',
       'vtti.turn_signal', 'vtti.wiper']

all_cols =part_1 + part_2

# remove_cols = ['series_id', 'file_id', 'vtti.timestamp']
# TARGETS_COLS = ['aug.road_type', 'aug.surfaceCondition', 'aug.lighting', 'aug.relationToJunction', 'aug.weather',
#                 'aug.locality']
# data = data.drop(remove_cols + TARGETS_COLS)

# disply_hist(data[part_1])
# disply_hist(data[part_2])

for col in all_cols:
    disply_hist_1(data, col)
describe = data.describe()

describe.to_csv('./describe_no_clean.csv')