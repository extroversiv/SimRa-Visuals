import zipfile
import os
import pandas as pd
# import geopandas as gp
import pickle
import osmnx
import numpy as np
import folium
import matplotlib.colors as clr
import networkx as nx

class SimRaVisuals:
    """This class collects the data from the SimRa project for visualization.
    The data is contained in zip archives of files with individual ride data."""
    def __init__(self, path_dataset, path_pickle):
        self.path_dataset = path_dataset    # dataset path name
        self.path_pickle = path_pickle      # path for pickle files
        self.rides = pd.DataFrame()         # main data frame with the information about the rides to be stored
        self.graph = 0
        self.filtered_bbox = []

    def read_data(self):
        """Read all data from zip files in provided path. Previously processed data is read from pickle file."""
        print('+'*15, 'read data', '+'*15)
        zip_list = [name for name in os.listdir(self.path_dataset) if name.endswith('.zip')]
        print('Found these zip files in {}: {}'.format(self.path_dataset, zip_list))
        self.rides = []  # read list of data frames, then concatenate (faster)
        for zip_name in zip_list:
            pickle_path = self.path_pickle + os.sep + zip_name.replace('.zip', '.pkl')
            zip_path = self.path_dataset + os.sep + zip_name
            if os.path.isfile(pickle_path):
                # Read previously processed data and append to list of data frames
                print('Reading pickle file', os.path.basename(pickle_path))
                self.rides.append(pickle.load(open(pickle_path, 'br')))
            else:
                # Read rides from zip file, save pickle and append to list of data frames
                df = self._read_data_zip(zip_path)
                pickle.dump(df, open(pickle_path, 'bw'))
                self.rides.append(df)
        self.rides = pd.concat(self.rides, ignore_index=True, sort=False)
        print('-'*42)

    def get_rides_description(self):
        data_points = self.rides['lat'].apply(len).sum()
        number_rides = self.rides.groupby('zip path')['file name'].nunique().sum()
        bbox = {'lon': [np.min(self.rides['lon'].apply(np.min)), np.max(self.rides['lon'].apply(np.max))],
                'lat': [np.min(self.rides['lat'].apply(np.min)), np.max(self.rides['lat'].apply(np.max))]}
        return {'data points': data_points, 'number rides': number_rides, 'bounding box': bbox}

    def filter_rides(self, bbox):
        pass
        # todo: neu machen
        self.bbox = bbox
        print('filtering data to bounding box', bbox)
        mask = self.rides['lon'].apply(lambda x: x.between(*bbox['lon']).all())
        self.rides = self.rides.loc[mask,:]
        mask = self.rides['lat'].apply(lambda x: x.between(*bbox['lat']).all())
        self.rides = self.rides.loc[mask, :]

    def setup_graph_bbox(self, bbox):
        print('+'*15, 'setup graph from bounding box', '+'*15)
        graph_name = 'graph_' + '_'.join([str(v) for l in bbox.values() for v in l]).replace('.', 'p') + '.pkl'
        pickle_path = self.path_pickle + os.sep + graph_name
        if os.path.isfile(pickle_path):
            # Read previously downloaded graph
            print('Reading pickle file', os.path.basename(pickle_path))
            self.graph = pickle.load(open(pickle_path, 'rb'))
        else:
            print('setting up graph for bounding box', bbox, ': downloading', end=' ... ')
            self.graph = osmnx.graph_from_bbox(south=bbox['lat'][0], north=bbox['lat'][1],
                                               west=bbox['lon'][0], east=bbox['lon'][1],
                                               network_type='all')      # could use bike but this might be to restrictive
            print('converting to undirected graph', end=' ... ')
            self.graph = osmnx.get_undirected(self.graph)               # forget the direction, this might be to restrictive
            # print('projecting to UTM', end=' ... ')
            # self.graph = osmnx.project_graph(self.graph)                # convert to x,y projection for faster euclidean calculations
            print('saving to', pickle_path)
            pickle.dump(self.graph, open(pickle_path, 'bw'))
        print('-'*42)

    def setup_graph_place(self, place, which_result=1):
        print('+'*15, 'setup graph from place', '+'*15)
        graph_name = 'graph_' + place + '.pkl'
        pickle_path = self.path_pickle + os.sep + graph_name
        if os.path.isfile(pickle_path):
            # Read previously downloaded graph
            print('Reading pickle file', os.path.basename(pickle_path))
            self.graph = pickle.load(open(pickle_path, 'rb'))
        else:
            print('setting up graph for place', place, ': downloading', end=' ... ')
            self.graph = osmnx.graph_from_place(query=place, which_result=which_result, network_type='all')      # could use bike but this might be to restrictive
            print('converting to undirected graph', end=' ... ')
            self.graph = osmnx.get_undirected(self.graph)               # forget the direction, this might be to restrictive
            # print('projecting to UTM', end=' ... ')
            # self.graph = osmnx.project_graph(self.graph)                # convert to x,y projection for faster euclidean calculations
            print('saving to', pickle_path)
            pickle.dump(self.graph, open(pickle_path, 'bw'))
        print('-'*42)

    def find_nearest_edges(self):
        # todo: mal ausprobieren ob das schneller ist: osmnx.get_nearest_edges(self.graph, X=1, Y=2, method='kdtree', dist=1)
        print('+'*15, 'find edges', '+'*15)
        # calculate edges if not done yet in each grp and save pickle
        for zip_path, df in self.rides.groupby(['zip path']):
            print(zip_path)
            if np.any([edges.isnull().any() for edges in df['edges']]):
                index = [idx for idx, edge in df['edges'].iteritems() if edge.isnull().any()]
                edges = [self._find_nearest_edges_ride(coords['lat'], coords['lon'])
                         for idx, coords in df.loc[index, ['lat', 'lon']].iterrows()]   # läßt sich leider schlecht parallelisieren (tried swifter, Dask, joblib)
                self.rides.loc[index, 'edges'] = edges
                # save pickle
                pickle_path = self.path_pickle + os.sep + os.path.basename(zip_path).replace('.zip', '.pkl')
                pickle.dump(self.rides.loc[index, :], open(pickle_path, 'wb'))
        print('-'*42)

    def filter_hourly(self, hour):
        print('+' * 15, 'filter', str(hour)+'.', 'hour', '+' * 15)
        # filter by the hour
        for idx, ride in self.rides.iterrows():
            # keep the entries in this hour only
            if not ride['time'].empty:
                mask = ride['time'].dt.hour == hour
                for col in ['time', 'lat', 'lon', 'edges']:
                    ride[col] = ride[col][mask]
            else:
                for col in ['time', 'lat', 'lon', 'edges']:
                    ride[col] = pd.Series([])

        # delete empty rows
        mask = self.rides['edges'].apply(lambda ride: not ride.empty)
        self.rides = self.rides.loc[mask, :]
        print('-' * 42)

    def analyze_ride_freq(self, clean=False, kind=None, offset=0):
        print('+' * 15, 'analyze frequency of rides on streets', '+' * 15)

        edges_list = [edges_one_ride.unique() for edges_one_ride in self.rides['edges']]

        if clean:
            # clean from little detours into sidearms due to gps-jitter and fill the gaps
            edges_list_clean = []
            for edges_one_ride in edges_list:
                # take edges only if not duplicated and not just 'short one ways'
                if len(edges_one_ride) > 0:
                    # ignore entries with just one entry (sidearms)
                    edges_clean = [edges_one_ride[0]]
                    for idx, edg in enumerate(edges_one_ride[1:-1], start=1):
                        last_edge = edges_clean[-1]
                        next_edge = edges_one_ride[idx+1]
                        new_edge = True
                        for e in last_edge:
                            if e in next_edge:
                                new_edge = False
                        if new_edge:
                            edges_clean.append(edg)
                    # look for missing links
                    # for idx, edg in enumerate(edges_clean, start=1):
                    #     nx.shortest_path()
                    # append to cleaned edges
                    edges_list_clean.append(edges_clean)
            edges_list = edges_list_clean

        edges_list = [edge for edges in edges_list for edge in edges]
        edges_list = pd.value_counts(edges_list, normalize=False)

        # clean from low freuency parts
        if clean and kind=='quantile':
            # delete offset quantile from rides
            edges_list = edges_list[edges_list > edges_list.quantile(offset)]
        elif clean and kind=='linear':
            # delete offset linear from rides
            edges_list = edges_list[edges_list > offset]


        # count rides and store as edge attribute
        nx.set_edge_attributes(self.graph, 0, 'rides')

        for edge, count in edges_list.items():
            if self.graph.has_edge(*edge):
                # check for multiple edges connecting the same nodes -> take shortest one
                e = np.argmin([self.graph[edge[0]][edge[1]][e]['length'] for e in self.graph[edge[0]][edge[1]]])
                try:
                    self.graph[edge[0]][edge[1]][e]['rides'] = count
                except:
                    print('a problem occured with edge', edge, 'number', e)

        # relative number of rides in promille

        print('-' * 42)

    def plot_heatmap(self, save_file=None, show_plot=True, bbox=None, margin=0, bins=None, dpi=600):
        print('+' * 15, 'plot heatmap', '+' * 15)
        # drop edges with no rides
        nodes_list = [[u, v] for u, v, data in self.graph.edges(keys=False, data=True) if data['rides'] > 0]
        nodes_list = np.unique(nodes_list)

        if len(nodes_list) > 0:
            subgraph = self.graph.subgraph(nodes_list)

            save_figure = False
            if save_file is not None:
                save_figure = True
                print('saving figure')
            edge_colors, bins = self._get_edge_colors_by_attr(subgraph, attr='rides', cmap='hot', num_bins=50, bins=bins)  # changed this function in osmnx
            fig, ax = osmnx.plot_graph(subgraph, edge_color=edge_colors, node_size=0, bgcolor='k',
                                       save=save_figure, filename=save_file, dpi=dpi, show=show_plot,
                                       bbox=bbox, margin=margin)

            # ec = self._get_edge_colors_by_attr(subgraph, attr='rides', cmap='Reds', num_bins=20)  # changed this function in osmnx
            # fig, ax = osmnx.plot_graph(subgraph, edge_color=ec, node_size=0, bgcolor='w',
            #                            save=save_figure, filename=save_file, dpi=600)
        else:
            print('nothing to show')
            fig = 0
            ax = 0
        print('-' * 42)
        return fig, ax, bins

    def create_folium_webpage(self, file_name):
        print('+' * 15, 'create folium webpage', '+' * 15)
        # drop edges with no rides
        nodes_list = [[u, v] for u, v, data in self.graph.edges(keys=False, data=True) if data['rides'] > 0]
        nodes_list = np.unique(nodes_list)
        subgraph = self.graph.subgraph(nodes_list)

        print('saving folium webpage to', file_name)
        ec, _ = self._get_edge_colors_by_attr(subgraph, attr='rides', cmap='hot', num_bins=50)
        ec = [clr.to_hex(c[0:3]) for c in ec]    # convert to hex representation for html/folium
        folium_map = self._plot_graph_folium(subgraph, tiles='cartodbdark_matter',
                                             edge_color=ec, edge_width=3, edge_opacity=0.3)
        folium_map.save(file_name)
        print('-' * 42)

    def _read_data_zip(self, zip_path):
        """Read files in zip file as individual rides and return data frame"""
        print('Reading archive files from ', os.path.basename(zip_path), end=' ... ')
        zip_archive = zipfile.ZipFile(zip_path, 'r')
        zip_files = zip_archive.namelist()
        rides_df = []
        for file_name in zip_files:
            zip_data_raw = zip_archive.read(file_name)  # read binary
            zip_data_raw = zip_data_raw.decode('utf-8').splitlines()  # convert to list of lines in file
            # first lines is incidents data, separated by a line of hashes to the ride data
            # ignore incidents data for now
            # read ride data
            separator_line = next(
                i for i in range(len(zip_data_raw)) if zip_data_raw[i].startswith('==='))  # first separator
            # ridesDataVersion = zip_data_raw[separator_line + 1]
            rides_data_raw = zip_data_raw[separator_line + 2:]
            rides_data = [row.split(',') for row in rides_data_raw]
            df = pd.DataFrame(rides_data[1:], columns=rides_data[0])
            # drop rows without lat,lon for now (todo: sollen wir die Informationen behalten?)
            df = df[(df[['lat', 'lon']] != '').any(axis=1)]
            if not df.empty:  # klappt das?
                df[['lat', 'lon']] = df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce', axis=1)
                df['timeStamp'] = df['timeStamp'].apply(pd.to_datetime, unit='ms', errors='coerce')
                df = df.reset_index(drop=True)
                rides_df.append({'zip path': zip_path,
                                 'file name': file_name,
                                 'time': df['timeStamp'],
                                 'lon': df['lon'],
                                 'lat': df['lat'],
                                 'edges': pd.Series((np.nan, np.nan))})

        # concat to one dataframe and convert to numeric
        print('creating dataframe', end=' ... ')
        rides_df = pd.DataFrame(rides_df)
        #numeric_cols = ['lat', 'lon', 'X', 'Y', 'Z', 'acc', 'a', 'b', 'c']
        #rides_df[numeric_cols] = rides_df[numeric_cols].apply(pd.to_numeric, errors='coerce', axis=1)

        # convert to geopandas data frame and change coordinate system to projected x,y
        #rides_df = gp.GeoDataFrame(rides_df, geometry=gp.points_from_xy(rides_df['lon'], rides_df['lat']), crs='+init=epsg:4326')
        #rides_df = osmnx.project_gdf(rides_df)

        print('done')
        return rides_df

    def _find_nearest_edges_ride(self, lats, lons):
        edge_prev = -2
        edge_list = []
        for lat, lon in zip(lats, lons):
            if edge_prev == -2:
                edge_prev = -1
                edge = osmnx.get_nearest_edge(self.graph, (lat, lon))[1:]
            else:
                if edge != edge_prev:
                    # get nodes of edge and its neighbors to induce relevant subgraph
                    nodes_list = [[node, *list(self.graph.neighbors(node))] for node in edge]
                    # flatten list
                    nodes_list = set([node for sublist in nodes_list for node in sublist])
                    # find point on subgraph of neighbors of last point#
                    subgraph = self.graph.subgraph(nodes_list)
                edge_prev = edge
                edge = osmnx.get_nearest_edge(subgraph, (lat, lon))[1:]
            edge_list.append(edge)
        return pd.Series(edge_list)

    def _get_edge_colors_by_attr(self, G, attr, num_bins=5, cmap='viridis', start=0, stop=1, na_color='none', bins=None):
        # todo: send to osmnx with option qcut oder cut
        """
        Get a list of edge colors by binning some continuous-variable attribute into
        quantiles.

        Parameters
        ----------
        G : networkx multidigraph
        attr : string
            the name of the continuous-variable attribute
        num_bins : int
            how many quantiles
        cmap : string
            name of a colormap
        start : float
            where to start in the colorspace
        stop : float
            where to end in the colorspace
        na_color : string
            what color to assign nodes with null attribute values

        Returns
        -------
        list
        """
        if bins is None:
            bin_labels = range(num_bins)
            colors = osmnx.get_colors(num_bins, cmap, start, stop)
        else:
            num_bins = bins
            bin_labels = range(len(num_bins)-1)
            colors = osmnx.get_colors(len(num_bins)-1, cmap, start, stop)
        attr_values = pd.Series([data[attr] for u, v, key, data in G.edges(keys=True, data=True)])
        cats, bins = pd.cut(x=attr_values, bins=num_bins, labels=bin_labels, retbins=True)
        edge_colors = [colors[int(cat)] if pd.notnull(cat) else na_color for cat in cats]
        return edge_colors, bins

    def _plot_graph_folium(self, G, graph_map=None, popup_attribute=None,
                           tiles='cartodbpositron', zoom=1, fit_bounds=True,
                           edge_color='#333333', edge_width=5, edge_opacity=1):
        """
        Plot a graph on an interactive folium web map.

        Note that anything larger than a small city can take a long time to plot and
        create a large web map file that is very slow to load as JavaScript.

        Parameters
        ----------
        G : networkx multidigraph
        graph_map : folium.folium.Map
            if not None, plot the graph on this preexisting folium map object
        popup_attribute : string
            edge attribute to display in a pop-up when an edge is clicked
        tiles : string
            name of a folium tileset
        zoom : int
            initial zoom level for the map
        fit_bounds : bool
            if True, fit the map to the boundaries of the route's edges
        edge_color : string
            color of the edge lines
        edge_width : numeric
            width of the edge lines
        edge_opacity : numeric
            opacity of the edge lines

        Returns
        -------
        graph_map : folium.folium.Map
        """

        # check if we were able to import folium successfully
        if not folium:
            raise ImportError('The folium package must be installed to use this optional feature.')

        # create gdf of the graph edges
        gdf_edges = osmnx.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

        # get graph centroid
        x, y = gdf_edges.unary_union.centroid.xy
        graph_centroid = (y[0], x[0])

        # create the folium web map if one wasn't passed-in
        if graph_map is None:
            graph_map = folium.Map(location=graph_centroid, zoom_start=zoom, tiles=tiles)

        # add each graph edge to the map
        for index, row in gdf_edges.iterrows():
            pl = osmnx.make_folium_polyline(edge=row, edge_color=edge_color[index], edge_width=edge_width,
                                      edge_opacity=edge_opacity, popup_attribute=popup_attribute)
            pl.add_to(graph_map)

        # if fit_bounds is True, fit the map to the bounds of the route by passing
        # list of lat-lng points as [southwest, northeast]
        if fit_bounds:
            tb = gdf_edges.total_bounds
            bounds = [(tb[1], tb[0]), (tb[3], tb[2])]
            graph_map.fit_bounds(bounds)

        return graph_map