from SimRaVisuals import SimRaVisuals
import os
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np

if __name__ == '__main__':
    path_dataset = '..\\dataset\\Berlin\\Rides'
    path_pickle = '..\\pickle\\Berlin\\Rides'
    bbox = {'lon': [12.95, 13.8],
            'lat': [52.35, 52.7]}
    name = 'Berlin'

    if not os.path.isdir(path_pickle):
        os.makedirs(path_pickle)

    srv = SimRaVisuals(path_dataset, path_pickle)

    # read data
    srv.read_data()
    rides_description = srv.get_rides_description()
    print('{} datapoints in {} rides'.format(rides_description['data points'], rides_description['number rides']))
    print('bounding box of data:', rides_description['bounding box'])
    srv.filter_rides(bbox)
    rides_description = srv.get_rides_description()
    print('{} datapoints in {} rides'.format(rides_description['data points'], rides_description['number rides']))

    # download graph and find nearest edges
    srv.setup_graph_bbox(bbox)
    srv.find_nearest_edges()

    # analyze rides
    srv.analyze_ride_freq(clean=True, kind='quantile', offset=0.01)

    # plot heat map
    fig, ax, bins = srv.plot_heatmap(save_file=name, show_plot=False, dpi=300,
                                     bbox=bbox['lat'][::-1] + bbox['lon'][::-1], bins=None)

    # adjust bins to cyclist per hour and without an upper limit
    bins /= 24
    bins[-1] = np.inf

    for hour in range(0, 24):
        srv_cpy = deepcopy(srv)

        # filter hourly
        srv_cpy.filter_hourly(hour)

        # analyze rides
        srv_cpy.analyze_ride_freq(clean=True, kind='quantile', offset=0.01)

        # plot heat map
        heatmap_file_name = name.replace(', ', '_') + '_hour_' + str(hour)
        fig, ax, _ = srv_cpy.plot_heatmap(save_file=heatmap_file_name, show_plot=False, dpi=100,
                                           bbox=bbox['lat'][::-1] + bbox['lon'][::-1], bins=bins)
