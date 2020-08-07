from SimRaVisuals import SimRaVisuals
import os
import pickle

if __name__ == '__main__':
    path_dataset = '..\\dataset\\Berlin\\Rides'
    path_pickle = '..\\pickle\\Berlin\\Rides'
    districts = {'Mitte, Berlin, Germany': 2,
                 'Pankow, Berlin, Germany': 1,
                 'Charlottenburg-Wilmersdorf, Berlin, Germany': 1,
                 'Friedrichshain-Kreuzberg, Berlin, Germany': 1,
                 'Lichtenberg, Berlin, Germany': 1,
                 'Marzahn-Hellersdorf, Berlin, Germany': 1,
                 'Neukölln, Berlin, Germany': 2,
                 'Reinickendorf, Berlin, Germany': 1,
                 'Spandau, Berlin, Germany': 2,
                 'Steglitz-Zehlendorf, Berlin, Germany': 1,
                 'Tempelhof-Schöneberg, Berlin, Germany': 1,
                 'Treptow-Köpenick, Berlin, Germany': 1
                 }

    for district, which_result in districts.items():

        if not os.path.isdir(path_pickle):
            os.makedirs(path_pickle)

        srv = SimRaVisuals(path_dataset, path_pickle)

        # read data
        srv.read_data()
        rides_description = srv.get_rides_description()
        print('{} datapoints in {} rides'.format(rides_description['data points'], rides_description['number rides']))
        print('bounding box of data:', rides_description['bounding box'])

        # download graph and find nearest edges
        srv.setup_graph_place(district, which_result=which_result)
        srv.find_nearest_edges()

        # analyze rides
        srv.analyze_ride_freq(clean=True)

        # plot heat map
        heatmap_file_name = district.replace(', ', '_')
        heatmap_pickle_path = 'images\\pickle'
        heatmap_folium_path = 'images\\web'
        if not os.path.isdir(heatmap_pickle_path):
            os.makedirs(heatmap_pickle_path)
        if not os.path.isdir(heatmap_folium_path):
            os.makedirs(heatmap_folium_path)

        # fig, ax = srv.plot_heatmap(save_file=heatmap_file_name)
        # pickle.dump((fig, ax), open(heatmap_pickle_path + os.sep + heatmap_file_name + '.pkl', 'bw'))

        srv.create_folium_webpage(heatmap_folium_path + os.sep + heatmap_file_name + '.html')
