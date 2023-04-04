# imports
import os
os.environ['USE_PYGEOS'] = '0'
from datetime import datetime, timezone
import math

import geopandas as gpd
import pandas as pd
import numpy as np

from geopy.distance import geodesic as gd
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt

import tilemapbase as tmb
from matplotlib import pyplot as plt
from matplotlib import lines


def load_csv_data(path: str) -> pd.DataFrame:
    """
    loads a csv from garmin fit file as a dataframe

    Args:
        path (str): the path to the csv file in format 'file.csv'.

    Returns:
        pd.DataFrame: pandas dataframe of the csv data.
    """

    # define root directory
    root_path = os.path.dirname(__file__)
    
    # define location of the full path using supplied string
    file_path = os.path.join(root_path, 'data', path)
    
    # load in file as dataframe
    df = pd.read_csv(file_path)
    
    # remove rows with no time associated 
    # - not present in current dataset but useful when handling garmin data
    df = remove_null_datetimes(df)
    
    # add in an column of unix timestamp in addition to the datetime
    df = add_unix(df)
    
    return df

def remove_null_datetimes(df:pd.DataFrame) -> pd.DataFrame:
    """
    removes any null datetimes in loaded df

    Args:
        df (pd.DataFrame): pandas dataframe of the csv data.

    Returns:
        pd.DataFrame: pandas dataframe of the csv data without null datetimes
    """
    
    # create copy
    dat = df.copy()
    
    # remove the null datetimes
    dat = dat[~dat.datetime.isnull()]
    
    return dat

def add_unix(df:pd.DataFrame) -> pd.DataFrame:
    """
    adds a unix timestamp column to loaded dataframe

    Args:
        df (pd.DataFrame): pandas dataframe of the csv data.

    Returns:
        pd.DataFrame: pandas dataframe of the csv data with a new timestamp column
    """
    # set unix datetime, tz aware
    unix = datetime(1970,1,1, tzinfo = timezone.utc)
    
    # create a copy of the df
    dat = df.copy()
    
    # format the datetime
    dat.datetime = pd.to_datetime(dat.datetime)
    
    # create unix timestamp
    dat['timestamp_unix'] =  dat.loc[:,'datetime'].apply(lambda x:int((x - unix).total_seconds()))
    
    return dat

def id_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    """
    identifies duplicated timestamps in dataframe

    Args:
        df (pd.DataFrame): pandas dataframe of the csv data.

    Returns:
        pd.DataFrame: rows where df is duplicated based on time
    """
    # create copy
    dat= df.copy()
    
    # create time difference
    dat['delta_seconds'] = dat.loc[:,'timestamp_unix'].diff()
    
    # id where a duplicate ocurred
    duplicates = dat[dat.delta_seconds == 0]
        
    return duplicates

def id_discons(df:pd.DataFrame) -> pd.DataFrame:
    """
    identifies discontinuties in dataframe based on timestamp

    Args:
        df (pd.DataFrame): pandas dataframe of the csv data.

    Returns:
        pd.DataFrame: rows where df has discontinuty based on time
    """
    
    # create copy
    dat = df.copy()
    
    # create time difference
    dat['delta_seconds'] = dat.loc[:,'timestamp_unix'].diff()
    
    # id where a discon ocurred
    discontinuities = dat[dat.delta_seconds > 1]
    
    return discontinuities

def id_gps_gaps(df:pd.DataFrame) -> pd.DataFrame:
    """
    checks if there are any gaps in lat/lon data by looking for nulls

    Args:
        df (pd.DataFrame): pandas dataframe of the csv data.

    Returns:
        pd.DataFrame: rows where latitude is null
    """
    
    dat = df.copy()
    
    gaps = dat[dat.latitude.isnull()]
    
    return gaps
    

def plot_one_gps(df: pd.DataFrame, 
                 figsize: tuple, 
                 color: str):
    """
    plots one dataframe's gps data

    Args:
        df (pd.DataFrame): pandas dataframe of the csv data.
        figsize (tuple): the size of the figure in inches (width, height)
        color (str): colour of the markers

    Returns:
        _type_: a matplotlib figure and axis pairing
    """
    
    # define figure and axis
    fig, ax = plt.subplots(1, 1, 
                           figsize = figsize)
    
    # load in df to geodf
    geodat = load_geodf(df = df, 
                        epsg_from = 4326, 
                        epsg_to = 3857)
    
    # get tiles and plot as background map
    plotter = setup_tmb(geodat)
    plotter.plot(ax)
    
    # plot the markers on top of line
    geodat.plot(ax = ax,
                markersize = 0.25,
                color = color,
                alpha = 0.25)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax

def plot_compare_gps(df1:pd.DataFrame, df2: pd.DataFrame, 
                     figsize:tuple, 
                     colors:list, labels:list,
                     start:bool, special_marker:str):
    """
    plots two dataframes gps data at either the start or end of the race

    Args:
        df1 (pd.DataFrame):  pandas dataframe of the csv data.
        df2 (pd.DataFrame):  pandas dataframe of the csv data. 
        figsize (tuple): the figure size in inches (width, height)
        colors (list): the colours to plot for respective athlete
        labels (list): names of the athletes for legend
        start (bool): at the start or end of race
        special_marker (str): type of marker to denote first/last marker

    Returns:
        _type_: a matplotlib figure and axis pairing
    """
    
    # define figure and axis
    fig, ax = plt.subplots(1, 1, 
                           figsize = figsize)
    # load in df to geodf
    geodat1 = load_geodf(df = df1, 
                         epsg_from = 4326, 
                         epsg_to = 3857)
    geodat2 =  load_geodf(df = df2, 
                          epsg_from = 4326, 
                          epsg_to = 3857)
    
    # get tiles and plot as background map
    plotter = setup_tmb(geodat1)
    plotter.plot(ax)
    
    # plot the markers
    geodat1.plot(ax = ax,
                 markersize = 0.25,
                 color = colors[0],
                 alpha = 0.33)
    geodat2.plot(ax = ax,
                 markersize = 0.25,
                 color = colors[1],
                 alpha = 0.33)
    
    # if it's the starting location
    if start == True:
        # mark the starting location with a sideways triangle
        geodat1.iloc[:1,:].plot(ax = ax,
                               marker = special_marker,
                               markersize = 100,
                               color = colors[0],
                               alpha = 1)
        geodat2.iloc[:1,:].plot(ax = ax,
                               marker = special_marker,
                               markersize = 100,
                               color = colors[1],
                               alpha = 1)
    # if it's the end location
    elif start == False:
        # mark the starting location with a sideways triangle
        geodat1.iloc[-1:,:].plot(ax = ax,
                               marker = special_marker,
                               markersize = 100,
                               color = colors[0],
                               alpha = 1)
        geodat2.iloc[-1:,:].plot(ax = ax,
                               marker = special_marker,
                               markersize = 100,
                               color = colors[1],
                               alpha = 1)        
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # add legend
    ax.legend(handles = [lines.Line2D([], [], color = colors[0]),
                         lines.Line2D([], [], color = colors[1])],
              labels = [labels[0],
                       labels[1]],
              loc = "lower left",
              fontsize = 8)    
        
    return fig, ax

def load_geodf(df: pd.DataFrame, epsg_from: int, epsg_to: int) -> gpd.GeoDataFrame:
    """
    loads in a geodataframe and then changes the epsg format

    Args:
        df (pd.DataFrame): pandas dataframe from one athlete
        epsg_from (int): what the epsg was in orignial format
        epsg_to (int): what the epsg is being set to

    Returns:
        gpd.GeoDataFrame: a geopandas dataframe with a geometry column
    """
    
    # turn df1 into geodataframe
    geodf =  gpd.GeoDataFrame(df.copy(),
                                geometry = gpd.points_from_xy(df.longitude, 
                                                              df.latitude))
    
    # change coordinate system to work with tilemapbase
    geodf = geodf.set_crs(epsg=epsg_from)
    geodf = geodf.to_crs(epsg=epsg_to)

    return geodf

def setup_tmb(geodf: gpd.GeoDataFrame) -> tmb.Plotter:
    """
    creates a tile from openstreetmaps to use as a map base

    Args:
        geodf (gpd.GeoDataFrame): geodataframe of an athletes garmin file

    Returns:
        tmb.Plotter: an openstreetmap tile
    """
    
    # boiler plate for tilemapbase of the area
    tmb.init(create=True)
    extent = tmb.extent_from_frame(geodf, buffer = 20)
    extent1 = extent.to_aspect(1, False)
    tile = tmb.tiles.build_OSM()
    plotter = tmb.Plotter(extent1, 
                          tile,
                          height = 400)
    
    return plotter

def distance_from_gps(df:pd.DataFrame) -> list:
    """
    computes a distance travelled between gps points

    Args:
        df (pd.DataFrame): dataframe of garmin data with at least columns of latitude and longitude

    Returns:
        list: list of the distances in meters
    """
    # create copy of df
    dat = df.copy()
    # compute the distance travelled in each step
    distance = compute_distance(dat.latitude.values.tolist(), dat.longitude.values.tolist())
            
    return distance

def compute_distance(latitude: list, longitude: list) -> list:
    """
    function to return the distance travelled based on gps coordinates for time step
        - uses geopy geodesic calculation in meters

    Args:
        latitude (list): in degrees
        longitude (list): in degrees

    Returns:
        list: distance per time step in meters
    """

    # preset initial for sake of first loop
    lat1 = 0
    lon1 = 0
    
    # loop through the lat/lons
    for i, (lat2, lon2) in enumerate(zip(latitude, longitude)):
        # special first loop
        if i == 0:
            # preset the first distance to 0m
            distance = [0]
        # normally
        else:
            
            # starting coords
            start = (lat1, lon1)
            end = (lat2, lon2)
            
            # use ellipse model to get distance in meters
            distance = distance + [gd(start, end).meters]
        
        # set the coords for next loop starting position
        lat1 = lat2
        lon1 = lon2
    
    return distance


def start_end_markers(df:pd.DataFrame, marker_columns:list, distance_start:int, distance_end:int, speed_threshold:float) -> list:
    """
    find the start and the end of a race based on searching for speeds above and below, before and after distance_start and distance_end, respectively.

    Args:
        df (pd.DataFrame): df of garmin file
        marker_columns (list): the columns needed from df
        distance_start (int): where to stop looking in data for speeds above threshold
        distance_end (int): where to start looking in data for speeds below threshold
        speed_threshold (float): the speed threshold to identify the start and end of a race

    Returns:
        list: markers as a nested list with each value being one of the marker_columns
    """
    # copy df
    dat = df.copy()
    # create slice indicies
    start_indexer = (dat.loc[:,'speed'] > speed_threshold) & (dat.loc[:,'distance_travelled_raw'] < distance_start)
    end_indexer = (dat.loc[:,'speed'] < speed_threshold) & (dat.loc[:,'distance_travelled_raw'] > distance_end)
    # apply slice incices and get first instance for both
    start_marker = dat.loc[start_indexer,marker_columns].iloc[0]
    end_marker = dat.loc[end_indexer, marker_columns].iloc[0]
    # markers are these values
    markers = [start_marker.values.tolist(),
               end_marker.values.tolist()]
        
    return markers

def smooth_repeat_gps(df:pd.DataFrame) -> pd.DataFrame:
    """
    function to smooth when a gps signal freezes and overcorrects in consequtive samples

    Args:
        df (pd.DataFrame): pandas dataframe of garmin file

    Returns:
        pd.DataFrame: smoothed df
    """
    
    dat = df.copy()
    
    # find all the data with 0
    zero_distance = dat[(dat.loc[:,'distance_raw'] == 0)]
    
    # drop the start if its zero
    if zero_distance.index.values[0] == 0:
        zero_distance = zero_distance.drop(index = 0)
    
    # find mean and stdev of sample
    stdev = dat.distance_raw.std()
    mean = dat.distance_raw.mean()
    # set the upper limit to the mean + 3.5x stdev
    upper_lim = mean + (3.5*stdev)
    
    # loop through the zero distances
    for i, index in enumerate(zero_distance.index.values):
        # the subsequent distance
        next_distance = dat.loc[index+1, 'distance_raw']
        # if the subsequent distance is more than the upper limit
        if next_distance > upper_lim:
            # average the distance over 2 
            mean_distance = next_distance / 2
            # linear interpolation
            dat.loc[index:index+1, 'distance_raw'] = mean_distance
    
    return dat
    
def crop_by_markers(df:pd.DataFrame, markers:list, distance_measure:str) -> pd.DataFrame:
    """
    crops a df of race based on passed markers of start/end
        - uses a distance measure to refine search in case of overlapping lat/lon

    Args:
        df (pd.DataFrame): garmin df
        markers (list): list of nested markers with lat/lon/distance
        distance_measure (str): how to narrow the search for minimum distance to marker

    Returns:
        pd.DataFrame: cropped dataframe
    """
    # create copy of df and preset a column of segment to 0
    dat = df.copy()
    dat['segment'] = 0
    
    # loop through markers
    for i, (lat, lon, dist) in enumerate(markers):
        # slice out a chunk of data around the distance
        sample = dat.loc[(dat.loc[:, distance_measure] > (dist - 1000)) & (dat.loc[:, distance_measure] < (dist + 1000)), :].copy()
        # create a euclidian distance from marker 
        euclid = (((sample.loc[:,'latitude'] - lat) ** 2) + ((sample.loc[:,'longitude'] - lon) ** 2))
        euclid = euclid.apply(math.sqrt)
        # find the minimum distance
        break_point = euclid.idxmin()
        # on initial loop
        if i == 0: 
            last_point = 0
        # slice out based on the break point and previous break point
        dat.loc[last_point:break_point, 'segment'] = i
        # set for next loop
        last_point = break_point
    # when extra data - remove it
    if len(dat.loc[:,'segment'] == 0) > 1:
        dat = dat.loc[~(dat.loc[:,'segment'] == 0), :].reset_index(drop = True)
        dat.loc[:, distance_measure] = dat.loc[:,distance_measure] - dat.loc[0, distance_measure]
    
    return dat

def distance_fft(df:pd.DataFrame):
    """
    get FFT params for frequency spectrum

    Args:
        df (pd.DataFrame): df from race

    Returns:
        _type_: fft params for viewing spectrum
    """
    
    dat = df.copy()

    N = len(dat.distance_raw)
    
    yf = rfft(dat.distance_raw.values)
    xf = rfftfreq(N)
    
    return yf, xf

def butterworth_lowpass_filtfilt(df:pd.DataFrame, order:int, hz:float) -> list:
    """
    apply a butterworth lowpass filter to dataset based on order and cutoff freq

    Args:
        df (pd.DataFrame): df from race
        order (int): the order of the filter
        hz (float): the cutoff freq of filter

    Returns:
        list: list of the filtered signal
    """
    
    dat = df.copy()
    
    sos = butter(N = order, 
                 Wn = hz, 
                 btype = 'lowpass',
                 output = 'sos',
                 fs = 1)
    
    distance_flit = sosfiltfilt(sos, dat.distance_raw.values)
    
    return distance_flit
    
    
def compare_time(df:pd.DataFrame, compare_against:str, interp_distance:int, distance_column:str, speed_column:str) -> pd.DataFrame:
    """
    compares time of all athletes in df against one athlete at set distances based on the provided distance and speed columns

    Args:
        df (pd.DataFrame): df of race concatenated with multiple athlete
        compare_against (str): the name of the athlete to compare against
        interp_distance (int): the distance interval to interpolate at
        distance_column (str): the column to use for measuring distance
        speed_column (str): the column to use when needing speed to adjust time

    Returns:
        pd.DataFrame: df with a Time_Gap column...negative values behind comparing athlete; positive in front
    """
    # create copy
    dat = df.copy()
    # interpolate time based on position
    delta_dat = interp_time(dat, interp_distance, distance_column, speed_column)
    # compare against a predetermined athlete at the set positions
    delta_dat = compare_time_to_athlete(delta_dat, compare_against)
    # reset the index to make distance a column again
    delta_dat = delta_dat.reset_index()
    
    return delta_dat

def interp_time(df:pd.DataFrame, interp_distance:int, distance_column:str, speed_column:str) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): df of race concatenated with multiple athlete
        interp_distance (int): the distance interval to interpolate at
        distance_column (str): the column to use for measuring distance
        speed_column (str): the column to use when needing speed to adjust time

    Returns:
        pd.DataFrame: df with time at set distances
    """
    # unique athletes
    athletes = df.athlete.unique().tolist()
    # the minimum max distance travelled by any athlete
    min_distance = min(df.groupby('athlete')[distance_column].max())
    # where to interpolate
    interp_points = np.arange(0, min_distance, interp_distance)
    # preset dataframe
    dat_interp = pd.DataFrame()
    # loop through the athletes
    for athlete in athletes:
        # set a sliced df of only current athlete
        curr_df = df.loc[df.athlete == athlete, :].copy()
        # loop through the interpolation points
        for i, interp_point in enumerate(interp_points):
            # find the closest point based on distance
            indx = abs(curr_df.loc[:, distance_column] - interp_point).idxmin()
            # get the distance of this point
            indx_dist = curr_df.loc[indx, distance_column]
            # if its greater than interp point, use previous
            if indx_dist > interp_point:
                indx_dist = curr_df.loc[indx - 1, distance_column]
            # get the speed and time
            indx_speed = curr_df.loc[indx,speed_column]
            indx_time = curr_df.loc[indx, 'time_sec']
            # how far to travel to make it to interp point
            dist_to_travel = interp_point - indx_dist
            # if not moving, indx time is new time
            if indx_speed == 0:
                new_time = 0 + indx_time
            # otherwise, figure out how long it would take to travel based on current speed
            else:
                new_time = indx_time + (dist_to_travel/indx_speed)
             
            # create a new df based on this
            new_frame = pd.DataFrame(data = [[athlete, interp_point, new_time]], 
                                     columns = ['Athlete', 'Distance', 'Time'])
            # concat these frames
            dat_interp = pd.concat([dat_interp, new_frame],
                                    axis = 0,
                                    sort = False).reset_index(drop = True)
            
    return dat_interp

def compare_time_to_athlete(df:pd.DataFrame, compare_against:str) -> pd.DataFrame:
    """
    compares athletes times at set interpolated distances

    Args:
        df (pd.DataFrame): 
        compare_against (str): 

    Returns:
        pd.DataFrame: 
    """
    
    dat = df.copy()
    # set the index to distance
    dat = dat.set_index('Distance')
    # athletes 
    athletes = dat.Athlete.unique().tolist()
    # create time to compare against
    time_compare = dat.loc[dat.Athlete == compare_against, 'Time']
    # loop through athletes
    for athlete in athletes:
        # compare each athlete at all distances
        dat.loc[dat.Athlete == athlete, 'Time_Gap'] = dat.loc[dat.Athlete == athlete, 'Time'] - time_compare
    # flip so that negative means losing time
    dat.Time_Gap = dat.Time_Gap * -1
    
    return dat

def compare_position(df:pd.DataFrame, compare_against:str, interp_distance:int, distance_column:str, speed_column:str) -> pd.DataFrame:
    """
    Same idea as compare_time but using difference in position based on distance travelled

    Args:
        df (pd.DataFrame): df of race concatenated with multiple athlete
        compare_against (str): the name of the athlete to compare against
        interp_distance (int): the distance interval to interpolate at
        distance_column (str): the column to use for measuring distance
        speed_column (str): the column to use when needing speed to adjust time

    Returns:
        pd.DataFrame: df with a Position_Gap column
    """
    
    dat = df.copy()
    
    delta_dat = interp_position(dat, interp_distance, distance_column, speed_column)
    
    delta_dat = compare_position_to_athlete(delta_dat, compare_against)
        
    delta_dat = delta_dat.reset_index()
    
    return delta_dat

def interp_position(df:pd.DataFrame, interp_distance:int, distance_column:str, speed_column:str) -> pd.DataFrame:
    """
    slightly different than interp_time as you need to get both lat and lon out

    Args:
        df (pd.DataFrame): df of race concatenated with multiple athlete
        interp_distance (int): the distance interval to interpolate at
        distance_column (str): the column to use for measuring distance
        speed_column (str): the column to use when needing speed to adjust time

    Returns:
        pd.DataFrame: df with distance, lat and lon 
    """
    
    dat = df.copy()
    
    athletes = dat.athlete.unique().tolist()
    
    min_distance = min(dat.groupby('athlete')[distance_column].max())
    
    interp_points = np.arange(0, min_distance, interp_distance)
    
    dat_interp = pd.DataFrame()
    
    for athlete in athletes:
        
        curr_df = dat.loc[dat.athlete == athlete, :]

        for i, interp_point in enumerate(interp_points):

            indx = abs(curr_df.loc[:, distance_column] - interp_point).idxmin()
            
            indx_dist = curr_df.loc[indx, distance_column]
            
            if indx_dist > interp_point:
                indx = indx - 1
            
            indx_dist = curr_df.loc[indx, distance_column]            
            dist_to_travel = interp_point - indx_dist
            indx_speed = curr_df.loc[indx, speed_column]
            
            indx_lat = curr_df.loc[indx, 'latitude']
            indx_lon = curr_df.loc[indx, 'longitude']
            
            next_indx_lat = curr_df.loc[indx+1, 'latitude']
            next_indx_lon = curr_df.loc[indx+1, 'longitude']
            
            if indx_speed == 0:
                lat = indx_lat
                lon = indx_lon
                
            else:
                time_to_dist = dist_to_travel/indx_speed
                # assumes proportion of time and linear route to next time step
                lat = indx_lat + ((next_indx_lat - indx_lat) * time_to_dist)
                lon = indx_lon + ((next_indx_lon - indx_lon) * time_to_dist)
                
            
            new_frame = pd.DataFrame(data = [[athlete, interp_point, lat, lon]], 
                                     columns = ['Athlete', 'Distance', 'Latitude', 'Longitude'])
            
            dat_interp = pd.concat([dat_interp, new_frame],
                                    axis = 0,
                                    sort = False).reset_index(drop = True)
            
    return dat_interp
                
def compare_position_to_athlete(df:pd.DataFrame, compare_against:str) -> pd.DataFrame:
    """
    very similar to compare_time, but using difference in position between two gps points instead of time

    Args:
        df (pd.DataFrame): 
        compare_against (str): 

    Returns:
        pd.DataFrame: 
    """

    dat = df.copy()
    
    dat = dat.set_index('Distance')
    
    athletes = dat.Athlete.unique().tolist()

    lat_compare = dat.loc[dat.Athlete == compare_against, 'Latitude']
    
    lon_compare = dat.loc[dat.Athlete == compare_against, 'Longitude']
    
    for athlete in athletes:
        
        lat1 = lat_compare
        lon1 = lon_compare
        
        lat2 = dat.loc[dat.Athlete == athlete, 'Latitude']
        lon2 = dat.loc[dat.Athlete == athlete, 'Longitude']
        
        index = dat[dat.Athlete == athlete].index.values
        
        for start_lat, start_lon, end_lat, end_lon, distance in zip(lat1, lon1, lat2, lon2, index):
            
            start = (start_lat, start_lon)
            end = (end_lat, end_lon)
        
            dat.loc[(dat.index == distance) & (dat.Athlete == athlete), 'Distance_Gap'] = gd(start, end).meters

    return dat

def map_distance_travelled(df1:pd.DataFrame, df2:pd.DataFrame) -> list:
    """
    maps the distance from one athlete to another based on their gps position

    Args:
        df1 (pd.DataFrame): race df from athlete with map distance
        df2 (pd.DataFrame): race df from athlete to get mapped distance

    Returns:
        list: the distance for df2
    """
    # create copies and new columns
    dat1 = df1.copy()
    dat1['latlon'] = list(zip(dat1.latitude, dat1.longitude))
    dat2 = df2.copy()
    # preset output
    distance_travelled = []
    
    # associate distance travelled according to distance from lat,lat from df1
    for i, (lat2, lon2, dist2) in enumerate(zip(dat2.latitude, dat2.longitude, dat2.distance_travelled_filt)):

        dat1_sample = dat1.loc[(dat1.loc[:,'distance_travelled_filt'] > (dist2 - 200)) & (dat1.loc[:,'distance_travelled_filt'] < (dist2 + 200)), :].copy()

        dat1_sample['dist_from_gps'] = dat1_sample.latlon.apply(lambda x: gd((lat2,lon2), (x[0], x[1])).meters)
        
        current_distance = dat1_sample.loc[dat1_sample['dist_from_gps'].idxmin(),'distance_travelled_filt'].copy()
        
        # if it ever goes backwards, just say it's in the same spot
        if i == 0: 
            pass
        # if its less than or the same as previous then add 0.1 
        elif current_distance <= distance_travelled[i-1]:
            current_distance = distance_travelled[i-1] + 0.1
        
        # add to list
        distance_travelled = distance_travelled + [current_distance]

    return distance_travelled

def get_markers_from_distances(reference_df:pd.DataFrame, distances:list, columns:list) -> list:
    """
    returns list of markers [lat, lon, dist] to break up segments in a df

    Args:
        reference_df (pd.DataFrame): who's gps do you want to extract
        distances (list): the distances in the race where markers should be placed
        columns (list): columns to extract for markers

    Returns:
        list: nested list of markers with [lat, lon, distance]
    """
    
    dat = reference_df.copy()
    
    markers = []
    
    end_dist_num = len(distances)
    
    for i, dist in enumerate(distances):
        # first loop, just take the first value as start
        if i == 0:
            curr_marker = dat.loc[dat.index.values[0], columns].values.tolist()
        # take the closest point to requested distance
        elif i != 0 and i < (end_dist_num - 1):
            curr_marker = dat.loc[(dat.distance_travelled_mapped - dist).abs().idxmin(), columns].values.tolist()
        # if last, take the last index
        elif i == (end_dist_num - 1):
            curr_marker = dat.loc[dat.index.values[-1], columns].values.tolist()
            
        markers = markers + [curr_marker]
        
    
    return markers
        
def avg_sensors_by_segment(df:pd.DataFrame, sensors:list) -> dict:
    """
    groupby segment and get mean and percent change from start of requested sensors

    Args:
        df (pd.DataFrame): df of race with segment column
        sensors (list): the sensor columns to average

    Returns:
        dict: of dfs with keys = 'segment_means' & 'segment_change'
    """
    # create a copy of incoming df
    dat = df.copy()
    # average requested sensors by segment
    segment_means = dat.groupby('segment')[sensors].mean()
    # convert into % change
    segment_change = ((segment_means / segment_means.loc[segment_means.index.values[0], :]) * 100) - 100
    # melt df into long form
    segment_change = pd.melt(segment_change.reset_index(),
                             id_vars = 'segment',
                             value_vars = sensors,
                             var_name = 'sensor',
                             value_name = 'percent_change')
    # format df's into a dictionary    
    output_dict = {'segment_means':segment_means,
                   'segment_change':segment_change}
    
    return output_dict

def pace_average_over_distance(df:pd.DataFrame, distance_interval:int, distance_measure:str, round_digits:int = 2) -> pd.DataFrame:
    """
    creates an average pace over set distances during the race
    Args:
        df (pd.DataFrame): race df
        distance_interval (int): how far to average over (in meters)
        distance_measure (str): the distance column to use
        round_digits (int, optional): how many digits to round the time to. Defaults to 2.

    Returns:
        pd.DataFrame: df with time per km in seconds for distance intervals
    """


    # create copy of frame
    dat = df.copy()
    
    max_distance_remainder = dat.loc[:,distance_measure].max() % distance_interval
    
    if max_distance_remainder > 0:
        max_distance = dat.loc[:,distance_measure].max() + (distance_interval - max_distance_remainder)
    else: 
        max_distance = dat.loc[:,distance_measure].max()
        
    # create array of distance to avg over
    distances = np.arange(0, max_distance + 1, distance_interval)
    # get the length of this array
    num_bins = len(distances) - 1
    # turn distances into binned tuples
    distance_bins = [(x,y) for i, (x, y) in enumerate(zip(distances, distances[1:])) if i < num_bins]
    
    # preset output list
    data_out = []
    
    # loop through the distance bins
    for i, (start, end) in enumerate(distance_bins):
        
        # format slice
        sample_index = (dat.loc[:,distance_measure] >= start) & (dat.loc[:,distance_measure] < end)
        
        # get the slice based on current bin
        sample_dat = dat.loc[sample_index, ['time_sec', distance_measure]].copy()
        
        # get total time and distance travelled in this bin
        sample_time = sample_dat.time_sec.values[-1] - sample_dat.time_sec.values[0]
        sample_distance = sample_dat.loc[sample_dat.index.values[-1], distance_measure] - sample_dat.loc[sample_dat.index.values[0], distance_measure]
        
        # convert to pace in s/km
        seconds_per_km = sample_time / (sample_distance / 1000)
        
        # format output
        out_bin = [str(int(start/1000)) + '-' + str(int(end/1000)) + 'km',
                   start, end,
                   seconds_per_km]
        
        data_out = data_out + [out_bin]
    
    df_out = pd.DataFrame(data = data_out,
                          columns = ['distance_bin', 'distance_start', 'distance_end', 'pace_s_km'])
    
    df_out.pace_s_km = df_out.pace_s_km.round(round_digits)
    
    return df_out