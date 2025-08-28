
"""
Urban Heat Island Effect Exploration in Toronto
Created: Summer 2025
Author: Noah Vaillant

This script loads, cleans, and analyzes mobile (bike/truck), stationary
weather station and topographical and spatial datasets to study the UHI effect. 
There are some plotting functions mostly aimed at data analysis .
UHIresults.py has more visualizations and results based functionality.

"""

#=============================================================================
# Packages 
#=============================================================================

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from math import cos, sin, asin, sqrt
import geopandas as gpd
import shapely as sp
from shapely.ops import nearest_points
import scipy.stats as stats
import seaborn as sns
import pyproj
import contextily as ctx

#=============================================================================
# Helper Functions
#=============================================================================

# From stack overflow
# https://stackoverflow.com/questions/4913349
# /haversine-formula-in-python-bearing-and-distance-between-two-gps-points

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1 *= np.pi / 180.0
    lat1 *= np.pi / 180.0
    lon2 *= np.pi / 180.0
    lat2 *= np.pi / 180.0
    

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def lonlat_to_mercator(lon, lat):
    k = 6378137
    x = lon * (np.pi / 180) * k
    y = np.log(np.tan((np.pi / 4) + (lat * (np.pi / 180) / 2))) * k
    return x, y


def haversine_series(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)

    Applicable to pandas Series

    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    return 6371 * 2 * np.arcsin(np.sqrt(a))

# =============================================================================
# General Format
# =============================================================================

# All datasets with temperature information have an index of UTC time,
# a 'tod' in fraction hours, a temperature in celcius, latitute and longitude,
# year, month and day


# =============================================================================
# Set Path
# =============================================================================

#Path to be changed by user
MAIN_DIR = '/Users/noahvaillant/Desktop/CGCS/Heat_Island/'

#subfolder containing station data
STATION_DIR = MAIN_DIR + 'stations'

#subfolder containing geodata
GEODATA_DIR = MAIN_DIR + 'GeoData'


# =============================================================================
# Loading Data
# =============================================================================


#Mobile Cycling and Truck Data
def load_mobile_data(date, calibrated=True, truck=False):
    """
    Loads weather data from bike and truck mobile weather stations, one route at a time.
    Inconsistencies in the formating of the data pre-2019 created the need for some messy if statements.
    Data is named differently if calibrated or uncalibrated. Calibration does not effect anything weather related.

    Parameters
    ----------
    date : 'YYYY-MM-DD'
    calibrated : Optional, The default is True.
    truck : Optional, deafult False. 

    Returns
    -------
    DataFrame object with gps time as the index. Includes columns for time of day (fraction hours),
    temperature, latitude, longitude, month, date and year. 

    """
   
    if date in ['2017-06-28','2017-07-04', '2017-07-07', '2017-07-14','2017-07-19'] :

        if truck:
             raw = pd.read_csv(STATION_DIR + '/Truck_Data/sync_data_'+date+'.csv',
                              usecols=['gps_time','lat','lon','temp'], parse_dates=[0], index_col=0)
        else:
            raw = pd.read_csv(STATION_DIR + '/Bike_Data/'+ 
                              date[:4]+'_bikedata_uncalibrated/datasource_'+date+'.csv',
                              usecols=[0,1,2,4], parse_dates=[0], index_col=0)
        
        
        raw.index.rename('gps_time')

        #Format mixed as some datapoints have mircoseconds and some do not
        raw.index = pd.to_datetime(raw.index, format='mixed')
        
        #Creating a time of day in fraction hours
        tod = raw.index.array.hour + raw.index.array.minute/60 + raw.index.array.second/3600
        raw.insert(0,'tod',tod)
        
        #There was spaces in the first bike csvs
        raw = raw.rename(columns={' lat': 'lat', ' lon': 'lon', ' temp': 'temp'})
    

        return raw
            

    if int(date[:4]) < 2019:
        
        
        if truck:
             raw = pd.read_csv(STATION_DIR + '/Truck_Data/sync_data_'+date+'.csv',
                              usecols=['gps_time','lat','lon','temp'], parse_dates=[0], index_col=0)
        
       
        
        elif calibrated:
            
            raw = pd.read_csv(STATION_DIR + '/Bike_Data/'+ 
                              date[:4]+'_bikedata_calibrated/sync_data_'+date+'_cal.csv',
                              usecols=['gps_time', 'temp','lat', 'lon'], parse_dates=[0],
                              index_col=0)
            
      
        else:
            raw = pd.read_csv(STATION_DIR + '/Bike_Data/'+ 
                              date[:4]+'_bikedata_uncalibrated/sync_data_'+date+'.csv',
                              usecols=['gps_time','temp','lat', 'lon'], parse_dates=[0],
                              index_col=0)
        
        
        raw.index.rename('gps_time')
        
        #Format mixed as some datapoints have mircoseconds and some do not
        raw.index = pd.to_datetime(raw.index, format='mixed')
        
        #Creating a time of day in fraction hours
        tod = raw.index.array.hour + raw.index.array.minute/60 + raw.index.array.second/3600
        raw.insert(0,'tod',tod)
        

        return raw
    
    #Consistent after 2019
    
    # Data is differently named if calibrated or uncalibrated. Calibration does not
    # effect anything we are trying to look at

    if truck:
        if calibrated:
             raw = pd.read_csv(STATION_DIR + '/Truck_Data/sync_data_calibrated_'+date+'.csv',
                              usecols=['gps_time','lat','lon','temp'], parse_dates=[0], index_col=0)
        else:
            raw = pd.read_csv(STATION_DIR + '/Truck_Data/sync_data_'+date+'.csv',
                              usecols=['gps_time','lat','lon','temp'], parse_dates=[0], index_col=0)
    elif calibrated:
        
        
        raw = pd.read_csv(STATION_DIR + '/Bike_Data/'+ 
                          date[:4]+'_bikedata_calibrated/sync_data_calibrated_'+date+'.csv',
                          usecols=['gps_time', 'temp','lat', 'lon'], parse_dates=[0],
                          index_col=0)
          
    else:
        if int(date[:4]) == 2024:
            raw = pd.read_csv(STATION_DIR + '/Bike_Data/'+ 
                              date[:4]+'_bikedata_uncalibrated/sync_data_UoT_Licor_'+date+'.csv',
                              usecols=['gps_time','temp','lat', 'lon'], parse_dates=[0],
                              index_col=0)
        else:
            raw = pd.read_csv(STATION_DIR + '/Bike_Data/'+ 
                          date[:4]+'_bikedata_uncalibrated/sync_data_'+date+'.csv',
                          usecols=['gps_time','temp','lat', 'lon'], parse_dates=[0],
                          index_col=0)
    
    
    
    raw.index.rename('gps_time')
    
    # Create a time of day in fraction hours
    tod =  24*(raw.index.array.day - raw.index[0].day )+ raw.index.array.hour + raw.index.array.minute/60 + raw.index.array.second/3600
    raw.insert(0,'tod',tod)
    

    return raw


# Toronto Atmospheric Observatory Weather Data
def load_TAO():
    
    """
    Loads weather data from the Toronto Atmospheric Observatory (TAO).
    Does so over the entire time period at once.
    Formatting is a little different in 2017

    Parameters
    ----------

    None

    Returns
    -------

    DataFrame object with gps time as the index. Includes columns for time of day (fraction hours),
    temperature, latitude, longitude, month, date and year. 

    """
    

    years=['17','18','19', '20', '21', '22', '23', '24']
    months=['Mar', 'Apr', 'May', 'Jun', 'Jul',
          'Aug', 'Sep', 'Oct', 'Nov']
    
    first = True
    df = pd.DataFrame({})
    
   
    for year in years:
        
            
        for month in months:

            if year == '17':
                raw = pd.read_csv(STATION_DIR + '/TAO_Data/20'
                                  + year +'/' +month+year+'log.txt', 
                            names=['Date', 'tod', 'temp'], usecols=[0,1,2], skiprows=0
                           )
            elif year == '18' and month == 'Nov':
                raw = pd.read_csv(STATION_DIR + '/TAO_Data/20'
                                  + year +'/' +month+year+'log.txt', 
                            names=['Date', 'tod', 'temp'], usecols=[1,2,3], skiprows=1)
            else:
                raw = pd.read_csv(STATION_DIR + '/TAO_Data/20'
                              + year +'/' +month+year+'log.txt', 
                        names=['Date', 'tod', 'temp'], usecols=[0,1,2], skiprows=1
                       )
            
            
            
            if year == '17':
                y = pd.to_datetime(raw['Date'], format='%d/%m/%y')
                d = pd.to_datetime(raw['tod'], format='%H:%M') - pd.to_datetime('1900-01-01')
                
                c = y+d
                raw = raw[['tod','temp']]
                raw.index = c
                raw['lat'] = 43.66
                raw['lon'] = -79.4
                
                
                
                d = d.array
                fractionhours = d / np.timedelta64(1, 'h')
                raw['tod'] = fractionhours

            else:
                y = pd.to_datetime(raw['Date'], format='%d-%m-%y')
                d = pd.to_datetime(raw['tod'], format='%H:%M')- pd.to_datetime('1900-01-01')
                # - pd.to_datetime('2025-05-26')
                c = y+d
                raw = raw[['tod','temp']]
                raw.index = c
                raw['lat'] = 43.66
                raw['lon'] = -79.4
                
                d = d.array
                fractionhours = d / np.timedelta64(1, 'h')
                raw['tod'] = fractionhours
                
            if first:
                df = raw
                first = False
            else:
                df = pd.concat([df,raw])
            
           
    
    df['year'] = df.index.year
    df['year'] = df.year.astype(str)
    df['month'] = df.index.month.astype(str)
    df['month'] = df.month.astype(str)
    df['route'] = df.index.array.day.astype(str)


    # Some tweaking to make the mobile and TAO route column match formats
    df['route'] = df.route.apply(lambda x:  '0'+x if len(x)==1 else x)

    df['route'] = df.month.apply(lambda x:  '0'+x if len(x)==1 else x)

    df['route'] = df.index.year.astype(str) + '-' + df.month + '-' + df.route 



    return df

TAOdata = load_TAO()

#Enivroment and Climate Change Canada Weather data


#Eccc has a strange id system for it's weather stations.
stationtoid = {'Oshawa': '6155875', 
               'Port Weller': '6136699', 
               'Airport': '6158731',
               'Center Island': '6158359', 
               'Toronto City':'6158355' }


def load_ecccdata(station):
    """
    Loads in weather data from the Eccc's Historical Weather Data archive. 

    Parameters
    ----------
    station: station name, must be in stationtoid keys

    Returns
    -------
    DataFrame object with gps time as the index. Includes columns for time of day (fraction hours),
    temperature, latitude, longitude, month, date, year and temperture anomaly.

    """

    months=['03', '04', '05', '06', '07', '08', '09', '10', '11']
    years=['2017','2018','2019', '2020', '2021', '2022', '2023','2024']
    
    first = True
    
    df = pd.DataFrame({})
    
    for year in years:
            
        for month in months:
            raw = pd.read_csv(STATION_DIR + '/eccc_Data/'
                       + station + '/en_climate_hourly_ON_' + stationtoid[station] +'_' + month +
                       '-' + year +'_P1H.csv',
                       usecols=['Temp (°C)','Date/Time (UTC)' , 'Longitude (x)',
                                'Latitude (y)'], index_col='Date/Time (UTC)',
                       parse_dates=['Date/Time (UTC)']
                       )
           
            tod = raw.index.array.hour + raw.index.array.minute/60 + raw.index.array.second/3600
            raw.insert(0,'tod',tod)
            
            if first:
                df = raw
                first = False
            else:
                df = pd.concat([df,raw])
        
    df = df.rename(columns={'Temp (°C)': 'temp','Longitude (x)': 'lon', 
                            'Latitude (y)': 'lat'})
    
    df['year'] = df.index.year
    df['year'] = df.year.astype(str)
    df['month'] = df.index.month
    df['month'] = df.month.astype(str)
    df['day'] = df.index.array.day
    
    df['route'] = pd.to_datetime(df[["year", "month", "day"]])
    df['route']  = df['route'].astype(str)

    df = df.drop(columns='day')


    # anom column represents temperture anomaly from the TAO station
    df['anom'] = None
    
    TAOtemps = TAOdata['temp'].copy()

    df = df[~df.index.duplicated(keep='first')]
    TAOtemps = TAOtemps[~TAOtemps.index.duplicated(keep='first')]

    TAOtemps = TAOtemps.reindex(index=df.index, method='nearest',tolerance=pd.Timedelta('30m'))

    df['anom'] = df['temp'] - TAOtemps

    return df


ecccdata = {'Oshawa': load_ecccdata('Oshawa'), 
           'Airport': load_ecccdata('Airport'),
           'Center Island': load_ecccdata('Center Island'),
           'Toronto City': load_ecccdata('Toronto City')}

# Dictionary for plotting functions later
eccccolors = {'Oshawa':'red',
             'Airport': 'navy',
             'Toronto City': 'orange',
             'Center Island': 'green'}


print('done loading')

# =============================================================================
# Dates 
# =============================================================================

bikedates = [
    '2017-06-28','2017-07-04', '2017-07-07', '2017-07-14','2017-07-19','2017-08-10',
    '2017-08-15','2017-08-18','2018-07-13', '2018-07-18', '2018-07-20', '2018-07-24', 
    '2018-07-27', '2018-08-23','2018-09-05', '2018-09-13', '2018-10-12', '2018-10-16', 
    '2018-10-24', '2018-10-26', '2019-05-16', '2019-05-17', '2019-05-21', '2019-05-24', 
    '2019-05-27', '2019-05-31', '2019-06-06', '2019-06-07', '2019-06-11', '2019-06-12', 
    '2019-06-18', '2019-06-19', '2019-06-27', '2019-07-02','2019-07-03', '2019-07-08', 
    '2019-07-09', '2019-07-15','2019-07-18', '2019-07-22', '2019-07-23', '2019-07-24', 
    '2019-07-29', '2019-08-09', '2019-08-14', '2019-08-16', '2019-08-22', '2020-07-24', 
    '2020-07-28', '2020-07-31','2020-08-07', '2020-08-08','2020-08-31', '2020-09-11', 
    '2020-09-24', '2020-10-07', '2020-10-15', '2020-10-22', '2020-10-29', '2020-11-21',
    '2021-03-30', '2021-05-06', '2021-05-14', '2021-05-19', '2021-05-21', '2021-05-25', 
    '2021-05-27', '2021-06-01', '2021-06-02', '2021-06-10', '2021-06-16', '2021-06-22', 
    '2021-06-24', '2021-07-06', '2021-07-15', '2021-07-22', '2021-08-04', '2021-08-06', 
    '2021-08-11', '2021-08-27', '2021-08-31', '2021-09-02', '2021-10-20', '2021-10-23',
    '2022-04-27', '2022-04-29', '2022-05-10', '2022-06-10', '2022-06-14', '2022-06-17',
    '2022-08-10', '2023-05-09', '2023-05-17', '2023-05-30', '2023-06-06', '2023-06-21',
    '2023-07-06', '2023-07-12', '2023-07-19', '2023-09-19', '2024-05-30', '2024-05-31',
    '2024-06-12', '2024-07-05', '2024-07-26', '2024-08-07'
             ]

truckdates = [
    '2018-08-16','2018-08-17', '2018-11-20','2018-11-22','2018-11-29','2018-12-04',
    '2018-08-28','2018-08-30', '2018-09-27','2018-10-03','2018-10-06','2018-10-13',
    '2018-12-07','2019-04-04','2019-04-15', '2019-04-25', '2019-06-05', '2019-06-06',
    '2019-06-21', '2022-06-23', '2022-07-04','2022-07-22', '2022-07-23', '2022-09-21',
    '2022-10-14', '2022-10-19', '2022-11-02', '2022-11-23', '2022-11-25','2022-12-02', 
    '2022-12-06', '2022-12-29', '2023-01-06','2023-01-31', '2023-02-08', '2023-02-16', 
    '2023-03-16', '2023-03-20','2023-03-22','2023-04-27', '2023-05-02','2023-05-17', 
    '2023-05-18', '2023-05-25', '2023-05-31', '2023-07-14', '2023-07-26', '2023-07-26',
    '2023-07-26', '2023-08-09', '2023-08-21', '2023-08-28', '2023-08-31','2023-09-01', 
    '2023-09-25','2023-10-12', '2023-11-29'
              ]

# =============================================================================
# Data wrangling/interpolating/filtering
# =============================================================================



def bike_datasets():
    """
    Loops through every route of bike data.
    Creates a full DataFrame of bike data and an interpolation of both eccc and TAO data over the same times.
    Compiles the different datasets so they can be used in a consistent way.

    Parameters
    ----------
    None.

    Returns
    -------
    Three DataFrames
    """


    eccclists = {key: [] for key in ecccdata.keys()}
    TAOlist = []
    bikelist =[]

    TAOorig = {}




    for day in bikedates:

            year = day[:4]
            month = day[5:7]

            #check calibration
            cal = True
            if year in ['2017','2019', '2023', '2024']:
                cal = False

         
            #Load in route
            bikedf = load_mobile_data(day, cal)
            bikedf.loc[:,'route'] = day
            bikedf.loc[:,'month'] = month
            bikedf.loc[:,'year'] = year

            #Only summer months (kinda)
            if not bikedf.loc[:,'month'].array[0] in ['04','05','06','07','08','09']:
                continue

            

            # To reduce the spatial redundancy that happens when the bike is still
            # for a long time, this takes the average of all points within 20 meters of eachother
            # taken within 10 mininutes of eachother into a new point.

            # This also finds jumps of over 45 minutes where the bike is inactive and removes 15m.
            # I added this as there are some days where the bike goes inside and so the bike will
            # need to acclimatize to the outside before the temperture data is valid again.
            
            # Removes the first ten minutes or however long the bike is within is within 100m
            # of MP.    

            b = bikedf.copy()
            i = 0
            newpoints = []
            acclimatized = False
     
            while i < len(b.index) - 1 :
                
                # acclimitization from start
                if (not acclimatized):
                    if ((b.index.array[i] - bikedf.index.array[0]) < pd.Timedelta('10 min')) or (
                        haversine(b.iloc[i, 2], b.iloc[i, 1], -79.4, 43.66 ) < 0.1):

                        firsttime = max(b[haversine_series(b['lon'], b['lat'], -79.4, 43.66)>0.1].index.min(),
                                        b[b.index - b.index.array[0]>pd.Timedelta('10 min')].index.min())
                        
                        
                        acclimatized = True
                        b.drop(b[b.index<firsttime].index, inplace=True)

                # catching jumps
                if (b.index.array[i+1] - b.index.array[i]) > pd.Timedelta('45 min'):
                    b.drop(b[(b.index >= b.index.array[i+1]) & (b.index < b.index.array[i+1] + pd.Timedelta('15min'))].index, inplace=True)

                
                # Catching redudant points
                X,Y = b.iloc[i, 2], b.iloc[i, 1]
                
                redundant = (haversine_series(X,Y, b['lon'], b['lat']) < 0.02) & (b.index < b.index[i] + pd.to_timedelta(10, unit='m'))

                df = b[redundant]
                b.drop(b[redundant].index, inplace=True)

                df = df.loc[:, ['tod', 'lat', 'lon', 'temp']]
                t = pd.Series(df.index).median()
                med = df.median()
                

                point = pd.DataFrame({'tod': med.loc['tod'],
                                                'lat':med.loc['lat'],
                                                'lon':med.loc['lon'],
                                                    'temp': med.loc['temp'],
                                                    'route': day,
                                                    'year': year,
                                                    'month': month
                                                    }, index=[t])
                newpoints.append(point)
                
                
            bikedf = pd.concat(newpoints)

            

            
            # creating bounds to get the weather stations just at routes
            # the extra two hours wont matter after the interpolation
            starttime = (bikedf.index - pd.to_timedelta(2, unit='h')).min()
            endtime = (bikedf.index + pd.to_timedelta(2, unit='h')).max()

            TAOdf = TAOdata[(TAOdata.index>=starttime)&(TAOdata.index<=endtime)].copy()

            # TAO
            TAOdf.loc[:,'route'] = day
            TAOdf.loc[:,'month'] = month
            TAOdf.loc[:,'year'] = year

            # check that there is actually data
            if  len(TAOdf.index) == 0:
                print(day)
                continue
        

            #orig datasets are the original data before interpolation
            TAOorig = TAOdf.copy()

            # eccc
            ecccdf = {}
            for key in ecccdata.keys():
    
                
                ecccdf[key] = ecccdata[key][(ecccdata[key].index > TAOdf.index[0])&(ecccdata[key].index < TAOdf.index[-1])].copy()
                
                ecccdf[key].loc[:,'route'] = day
                ecccdf[key].loc[:,'month'] = month
                ecccdf[key].loc[:,'year'] = year
                    
                #  check that there is actually data
                if  len(ecccdf[key].index) == 0:
                    print(day)
                    continue

            #orig datasets are the original data before interpolation
            ecccorig = {}
            for key in ecccdata.keys():
                ecccorig[key] = ecccdf[key].copy()
            
            
            #Interpolating 
            # create total seconds for use in scipy interp
            
            bdt = bikedf.index - bikedf.index[0]
            bikedf.insert(1, 'seconds', bdt.total_seconds())
            
            # create total seconds for use in scipy interp
            tdt = TAOdf.index - bikedf.index[0]

            TAOdf.insert(1, 'seconds', tdt.total_seconds())
            
            # interpolate function
            taotemp = spi.interp1d(
                            TAOdf['seconds'],
                            TAOdf['temp'], 
                            kind='linear' )

            # spit out new df with route's times
            TAOdf =  pd.DataFrame(
                                {'tod': bikedf['tod'],
                                'lat': 43.66,
                                'lon': -79.4,
                                'seconds':  bikedf['seconds'],
                                    'route': day,
                                    'year': day[0:4],
                                    'month': month,
                                'temp': taotemp(bikedf['seconds']),
                                    'original': False },
                                index=bikedf.index)
            
                



            for key in ecccdata.keys():

                # create total seconds for use in scipy interp
                edt = ecccdf[key].index - bikedf.index[0]

                ecccdf[key].insert(1, 'seconds', edt.total_seconds())
                
                
                
                # interpolate function
                eccctemp = spi.interp1d(ecccdf[key]['seconds'], ecccdf[key]['temp'], 
                                kind ='linear')
                

                # spit out new df with route's times
                ecccdf[key] = pd.DataFrame(
                                    {'tod': bikedf['tod'],
                                    'lat': ecccdata[key].iloc[0, 2] ,
                                    'lon': ecccdata[key].iloc[0, 1],
                                    'seconds':  bikedf['seconds'],
                                        'route': day,
                                        'year': day[0:4],
                                        'month': month,
                                    'temp': eccctemp(bikedf['seconds']),
                                        'original': False},
                                    index=bikedf.index)
                
                
                
                    
            
            

            bikedf = bikedf[~bikedf.index.duplicated(keep='first')]
            bikedf.sort_index(inplace=True)
            
            TAOdf = TAOdf[~TAOdf.index.duplicated(keep='first')]
            TAOdf.sort_index(inplace=True)


            # Mark data points as original rather than interpolated
            for i in TAOorig.index:

                orig = TAOdf.index.get_indexer([i], method='nearest')
                orig = TAOdf.index[orig[0]]
                if (orig - i)< pd.Timedelta(seconds=30):
                    TAOdf.loc[TAOdf.index==orig, 'original'] = True
            
            for key in ecccdata.keys():
                ecccdf[key][~ecccdf[key].index.duplicated(keep='first')]
                ecccdf[key].sort_index(inplace=True)

                for i in ecccorig[key].index:
                    ecccdf[key].loc[abs(ecccdf[key].index - i) <pd.Timedelta(seconds=10), 'original'] = True
                
                ecccdf[key]['anom'] = ecccdf[key]['temp'] - TAOdf['temp']
                
                eccclists[key].append(ecccdf[key])

            # add to full dfs
            
            
            TAOlist.append(TAOdf)
            bikelist.append(bikedf)

    return {key: pd.concat(eccclists[key]) for key in ecccdata.keys()}, pd.concat(bikelist),pd.concat(TAOlist)
   
ecccdatafull, fullbikedata, TAOdatafull = bike_datasets()


del ecccdata


# Anom is anomaly temperature, calculated with TAO as the background.
# This is used to account for seasonal and diurnal changes in temperture.
# All spatial comparsions of heat use anomaly.
fullbikedata['anom'] =  fullbikedata['temp'] - TAOdatafull['temp']



# Creating a similar dataset for truck data, 
def truck_dataset():
    """
    Loops through every route of turck data.
    Creates a full DataFrame of bike data.
    Interpolates TAO over the data for anomaly temperture.
    Only creates one Dataset, unlike bikedata.

    Parameters
    ----------
    None.

    Returns
    -------
    One DataFrame
    """

    trucklist = []
    for day in truckdates:

        year = day[:4]
        month = day[5:7]

        # calibration check
        cal = True
        if year in ['2017', '2018', '2019', '2020']:
            cal = False
        tdf = load_mobile_data(day, calibrated=cal, truck = True)

        tdf.loc[:, 'route'] = day
        tdf.loc[:, 'month'] = month

        # Limits to summer months
        if not tdf.loc[:,'month'].array[0] in ['05','06','07','08','09']:
            continue

        tdf.loc[:, 'year'] = year


        # Limits to the Toronto area
        tdf = tdf[(tdf['lat'] > 43.5)& (tdf['lat'] < 44.0) &
                (tdf['lon'] > -80.0) & (tdf['lon'] < -79.0)].copy()
        tdf = tdf[~tdf['temp'].isna()]

        if len(tdf) == 0:
            continue
            
        
        # Filtered in the same way as teh bike data
        

        t = tdf.copy()
        i = 0
        acclimatized = False
        newpoints = []

        while i < len(t.index) - 1 :

            if (not acclimatized):
                    if ((t.index.array[i] - tdf.index.array[0]) < pd.Timedelta('10 min')) or (
                        haversine(t.iloc[i, 2], t.iloc[i, 1], -79.4, 43.66 ) < 0.1) or (
                            haversine(t.iloc[i, 2], t.iloc[i, 1], -79.4, 43.66 ) < 0.1):

                        firsttime = max(t[haversine_series(t['lon'], t['lat'], -79.4, 43.66)>0.1].index.min(),
                                        t[t.index - t.index.array[0]>pd.Timedelta('10 min')].index.min())
                        
                        
                        acclimatized = True
                        t.drop(t[t.index<firsttime].index, inplace=True)


            if (t.index.array[i+1] - t.index.array[i]) > pd.Timedelta('45 min'):
                t.drop(t[(t.index >= t.index.array[i+1]) & (t.index < t.index.array[i+1] + pd.Timedelta('15min'))].index, inplace=True)



            X,Y = t.iloc[i, 2], t.iloc[i, 1]
            
            redundant = (haversine_series(X,Y, t['lon'], t['lat']) < 0.02) & (t.index < t.index[i] + pd.to_timedelta(10, unit='m'))

            df = t[redundant]
            t.drop(t[redundant].index, inplace=True)

            df = df.loc[:, ['tod', 'lat', 'lon', 'temp']]
            times = pd.Series(df.index).median()
            med = df.median()
            

            point = pd.DataFrame({'tod': med.loc['tod'],
                                            'lat':med.loc['lat'],
                                            'lon':med.loc['lon'],
                                                'temp': med.loc['temp'],
                                                'route': day,
                                                'year': year,
                                                'month': month
                                                }, index=[times])
            newpoints.append(point)
            
            
            
        tdf = pd.concat(newpoints)

        starttime = (tdf.index - pd.to_timedelta(4, unit='h')).min()

        endtime = (tdf.index + pd.to_timedelta(4, unit='h')).max()

        Tdf = TAOdata[((TAOdata.index > starttime) & (TAOdata.index < endtime))]

        if  len(Tdf.index) == 0:
                # print(day)
                continue
            
        TAOorig = Tdf.copy()

        #Interpolating 
        # create total seconds for use in scipy interp
        
        secs = tdf.index - tdf.index[0]
        tdf.insert(1, 'seconds', secs.total_seconds())
        
    
        # create total seconds for use in scipy interp
        Tsecs = Tdf.index - tdf.index[0]

        Tdf.insert(1, 'seconds', Tsecs.total_seconds())
        
        # interpolate function
        taotemp = spi.interp1d(
                        Tdf['seconds'],
                            Tdf['temp'], 
                            kind='linear' )
        
        tdf.loc[:,'anom'] = tdf.temp - taotemp(tdf['seconds'])

        trucklist.append(tdf)

    

    return pd.concat(trucklist)


fulltruckdata = truck_dataset()

del TAOdata
print('Done bikedates loop')

# =============================================================================
# daily
# =============================================================================

#Loading Eccc's station which take daily data
station = '615S001'
def load_daily(stationid, station):


    dfs=[]
    for year in ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']:
        dfs.append(pd.read_csv(STATION_DIR + '/eccc_Data/'
                       + station + '/en_climate_daily_ON_' + stationid +'_'+ year +'_P1D.csv',
                       usecols=['Max Temp (°C)','Date/Time' , 'Longitude (x)',
                                'Latitude (y)'], index_col='Date/Time'
                       ))
    

    df=pd.concat(dfs)
    df = df[df.index.isin([*bikedates, *truckdates])]
    df = df[df.index.isin(TAOdatafull.route)]
    df = df.rename(columns={'Max Temp (°C)': 'temp',
                       'Longitude (x)': 'lon',
                       'Latitude (y)': 'lat'})
    


    # Anomaly taken using max temp from eccc and TAO
    # The maxes should be during the day, so this should be consistent
    # with the time limations on the other data
    maxes = TAOdatafull.loc[TAOdatafull['original'], ['temp','tod', 'route']].groupby('route').max()
    

    df['anom'] = df.temp - maxes.temp
    df['tod'] = maxes.tod

    # Some stuff that makes sure everything works consistently togther.
    # Routes and original are kind of meaningless here
    df['original'] = True
    df['month'] = [x[5:7] for x in df.index]
    df['year'] = [x[:4] for x in df.index]
    df['route'] = df.index

    return df
    
ecccdaily ={}
ecccdaily['NorthYork'] = load_daily('615S001', 'NorthYork')

 
 
# =============================================================================
# Geo Data
# =============================================================================



# Data from stats canada
# https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21

tract_shapes = gpd.read_file (GEODATA_DIR + '/lct_000a21a_e/lct_000a21a_e.shp', )


#Limits to the city of Toronto
tract_shapes = tract_shapes[tract_shapes['PRUID'] == '35']
tract_shapes = tract_shapes[tract_shapes['CTUID'].apply(lambda x: x[0:3] == '535')]


# Data from the Canaidian Census analyser / CHASS
density = pd.read_csv(GEODATA_DIR + '/nyrRpxgPj51_data.csv')

# Matching the ids from stats canada dataset to the CHASS dataset
tract_shapes.index = pd.to_numeric(tract_shapes['CTNAME'])
tract_shapes = tract_shapes[~tract_shapes['CTNAME'].duplicated(keep='first')]

# Adding geometries to density and population data
density = density.loc[density['COL3'].isin(tract_shapes.index)]
density = density.set_index('COL3', drop=True)

density = gpd.GeoDataFrame(density, geometry=tract_shapes.loc[density.index, 'geometry'])

density = density.to_crs('EPSG:4326')
density = density.rename(columns={'COL4': 'area',
                           'COL5': 'pop',
                           'COL6': 'density' 
                           })


# Load in GeoData from Toronto Open Data Portal

# Topographic Datasets

#Waterbodies
water = gpd.read_file("/Users/noahvaillant/Desktop/CGCS/Heat_Island/GeoData/Waterbodies & Rivers/TOPO_Waterbody_WGS84.shp")
water = water.set_crs('EPSG:4326') 

lake = sp.union_all(water[(water.WATERBODY_ == 'Lake Ontario')|(water.WATERBODY_ == 'Toronto Harbour')].geometry)

watertotal = sp.union_all(water.geometry)

#Treed area
treedarea = gpd.read_file(GEODATA_DIR + '/Treed Area WGS84/TOPO_TREED_AREA_WGS84.shp')

treedareatotal= sp.union_all(treedarea.geometry)

#Permeable Surface
perm = gpd.read_file(GEODATA_DIR + '/perm.csv', columns=['field_2'], rows=slice(1,13752))

perm = gpd.GeoDataFrame(perm, geometry=sp.from_wkt(perm.field_2), crs='EPSG:4326')

permtotal = sp.union_all([*perm.geometry])


# Parks
# Toronto Open Data Portal
parks = gpd.read_file("/Users/noahvaillant/Desktop/CGCS/Heat_Island/GeoData/parks-wgs84/CITY_GREEN_SPACE_WGS84.shp")
parks = parks.to_crs('EPSG:4326') 

parktotal = sp.union_all(parks.geometry)


# Elevation
# Toronto open Data Portal
tin = gpd.read_file(GEODATA_DIR + '/triangular-irregular-network/2023_TIN.shp')

tin = tin.to_crs('EPSG:4326')


# Massive land cover dataset
# Toronto open Data Portal
# Made from lyr format into raster into gdf into geojson

landcover = gpd.read_file(GEODATA_DIR + '/landcover.geojson')

landcover.rename({'land_cover':'int', 'land_cover_label':'label'},inplace=True, axis=1)

landcover['int'] = landcover.int.astype(int)


usedict = {
    i: (sp.union_all(landcover[landcover['int'] == i].geometry),landcover.loc[landcover['int'] == i, 'label'].values[0])
    for i in np.arange(1, 9)
}


# This all creates a 8 row DataFrame with columns of label and multipolygon.
usearray = np.array(list(usedict.values()))

usage = gpd.GeoDataFrame({'label': usearray[:,1],
                          'geometry':usearray[:,0]})

usage.loc[0, 'label'] = 'treecanopy'
usage.loc[1, 'label'] = 'grass/shrub'
usage.loc[2, 'label'] = 'bare earth'
usage.loc[3, 'label'] = 'watercover'
usage.loc[4, 'label'] = 'building'
usage.loc[5, 'label'] = 'road'
usage.loc[6, 'label'] = 'otherpaved'
usage.loc[7, 'label'] = 'agricultural'


print('landcover end')


# =============================================================================
# Limiting survey area
# =============================================================================

# between the lake and 401; between the humber river and scarborough

# Manual list of tracts to cut
cutlist = [5350247.01,5350244.01,5350244.02, 5350250.02,5350246.0,5350250.05,5350247.02,5350248.05,5350248.04,5350247.01,5350248.03,5350249.03, 5350249.02,5350249.01,5350250.04,5350245,5350248.02,5350250.01,5350249.05,5350249.04,5350300.01]

pop = density[density['COL0'].astype(float)<5350288]
pop = pop[~pop['COL0'].isin(cutlist)]


def is_east_of_river(neigh_centroid, river_geom):
    nearest = nearest_points(neigh_centroid, river_geom)[1]
    return neigh_centroid.x > nearest.x

river = water[water['WATERBODY_']=='Humber River'].geometry

pop["is_east"] = pop.geometry.apply(lambda pt: is_east_of_river(pt.centroid, river))
pop = pop[pop["is_east"]]
downtown = sp.union_all(pop.geometry)


# =============================================================================
# Averaging Functions
# =============================================================================


def gridaverage(years=['2017','2018','2019','2020','2021','2022','2023', '2024'], months=['05','5','06','6','07','7','08','8','09','9'],
                yticks=0.005, xticks=2*0.005, latmin=43.6, latmax=43.7,lonmin= -79.5,lonmax= -79.275, truck=False,
                stations=[], bar=2):
    
    """
    Grid-averaged anomaly temperatures over a spatial domain.

    Bins mobile anomaly temperature data, and optionally station datasets. 
    
    Averages each sqaure returned in 

    Parameters
    ----------
    years : list of str, optional
        List of years to include (default is all 2017–2024).
    months : list of str, optional
        List of months to include (as strings, e.g., '07' or '7').
        Default includes May–September.
    yticks : float, optional
        Latitude spacing of grid cells in degrees.
    xticks : float, optional
        Longitude spacing of grid cells in degrees.
    latmin, latmax : float, optional
        Minimum and maximum latitude bounds for the grid.
    lonmin, lonmax : float, optional
        Minimum and maximum longitude bounds for the grid.
    truck : bool, optional
        Whether to include truck data in the grid averaging (default False).
        If True, only truck data between 14:00–21:00 local time are included.
    stations : list of pandas.DataFrame, optional
        List of station datasets to include. Each dataset must contain
        columns `lat`, `lon`, `temp`, `anom`, `year`, and `month`.
    bar : int, optional
        Minimum number of days required in a grid cell for the value
        to be considered valid. Cells with fewer days are set to NaN.

    Returns
    -------
    ndarray of shape (n_lat, n_lon):
        grid : Average anomaly temperature per gridsquare.
        count : Number of individual data points in each gridsqaure.
        daycount : Number of distinct days (not just days, more like days from each source) contributing to each grid cell.

    Pandas series:
        lat : Latitude values corresponding to the grid rows.
        lon : Longitude values corresponding to the grid columns.

    ndarray of shape (n_points, 2):
        points : Coordinates of valid grid points (lon, lat).

    ndarray of shape (n_lat*n_lon, 2):
        allpoints: Coordinates of all grid points, valid or not.
   

    """
    
    
    if type(stations) is not list:
        stations = [stations]

    lat = np.arange(latmin, latmax + yticks, yticks)
    lon = np.arange(lonmin, lonmax + xticks, xticks)
    
    
    grid = np.zeros((len(lat), len(lon)))
    count = np.zeros((len(lat), len(lon)))
    daycount = np.zeros((len(lat), len(lon)))

    lat = pd.Series(lat, index=np.arange(0, len(lat), 1))
 
    lon = pd.Series(lon, index=np.arange(0, len(lon), 1))
    
    
    b = fullbikedata[fullbikedata.year.isin(years) & fullbikedata.month.isin(months)]
    t = fulltruckdata[fulltruckdata.year.isin(years) & fulltruckdata.month.isin(months)]
    


    for i in lat.index:
            if i + 1 < len(lat):
                for j in lon.index:
                    if j + 1 < len(lon): 
                          
                        
                        grid[i][j] += sum(list(b[(b['lat']>=lat[i]) &
                            (b['lat']<=lat[i+1]) &
                            (b['lon']>=lon[j])&
                            (b['lon']<=lon[j+1])]['anom'].array)) 
                
                
                        count[i][j] += len(list(b[(b['lat']>=lat[i]) &
                            (b['lat']<=lat[i+1]) &
                            (b['lon']>=lon[j])&
                            (b['lon']<=lon[j+1])]['anom'].array))
                

                        daycount[i][j] += len(set(b[(b['lat']>=lat[i]) &
                            (b['lat']<=lat[i+1]) &
                            (b['lon']>=lon[j])&
                            (b['lon']<=lon[j+1])].route)) 
                        
                        if truck:
                            
                            mask = (t['lat']>=lat[i]) & (t['lat']<=lat[i+1]) & (t['lon']>=lon[j])& (t['lon']<=lon[j+1]) & (t['tod']>14) &(t['tod']<21) &(t.year.isin(years))&(t.month.isin(months))
                            
                            if len(t[mask])>0:
                            
                                grid[i][j] += sum(list(t.loc[mask,'anom'].array))

                                count[i][j] += len(list(t.loc[mask,'anom'].array))

                                daycount[i][j] += len(set(t.loc[mask,'route'].array))
                        
                        


                        for s in stations:
                            

                            mask = (s['lat'] >= lat[i]) & (s['lat'] <= lat[i+1]) &(s['lon'] >= lon[j]) & (s['lon'] <= lon[j+1]) & (s.month.isin(months)) & (s.year.isin(years)) &(s.tod<21) & (s.tod>14) & s['original']

                            temps_s = s.loc[mask]


                            if len(temps_s)>0:


                                grid[i][j] += np.sum(temps_s.anom)

                                count[i][j] += len(list(temps_s.temp.array))

                                

                                daycount[i][j] += len(set(temps_s.route))
                            
    points = np.array([[0,0]])
    allpoints = points
    
    for i in lat.index:
        for j in lon.index:

            if daycount[i][j] >  bar:

                grid[i][j] = grid[i][j]/count[i][j]
                points = np.append(points, [[lon[j], lat[i]]], axis=0)

            elif count[i][j] == 0:
                grid[i][j] = np.nan
                count[i][j] = np.nan
                daycount[i][j] = np.nan

            else:
                grid[i][j] = np.nan
            allpoints = np.append(allpoints, [[lon[j], lat[i]]], axis=0)
    
    # print(len(points), len(grid.copy().flatten()[~np.isnan(grid.copy().flatten())]))
    fig, ax = plt.subplots(figsize=(8,5))
    imshow= ax.imshow(grid,origin='lower')
    plt.colorbar(imshow, ax=ax,)
    ax.set_aspect('equal')
    ax.set_xticks(lon.index[::4])
    ax.set_yticks(lat.index[::2])
    ax.set_xticklabels(round(lon,2)[::4]) 
    ax.set_yticklabels(round(lat,2)[::2]) 
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    return grid, count, lat, lon, points[1:,], allpoints[1:,], daycount


def grid_to_gdf(grid, count, lat, lon, points, allpoints, days):
    
    """
    Take the output from grid average and creates a GeoDataFrame with the squares as polygons.

    Parameters
    ----------
    The output from grid_average

    Returns
    -------
    gdf : A GeoDataFrame with polygon geometries representing each grid cell. Columns for temp (anomaly temperture), day count and count

    """
    
    xticks = lon[1] - lon[0]
    yticks = lat[1] - lat[0]
    
    polygons = []
    for i in range(len(lon) -1 ):
        for j in range(len(lat) -1 ):
            p = sp.Polygon([(lon[i], lat[j]),
                            (lon[i+1], lat[j]),
                            (lon[i+1], lat[j+1]),
                            (lon[i], lat[j+1])])
            polygons.append(p)

    
        
   


    values = grid[:-1, :-1].T.flatten()
    d = days[:-1, :-1].T.flatten()
    c = count[:-1, :-1].T.flatten()

    
    gdf = gpd.GeoDataFrame({'geometry': polygons, 'temp': values, 'day count': d, 'count': c }, crs='EPSG:4326')

    return gdf


# ================================================================  
# Grid Creation
# ================================================================


# Stations to use anomaly data from
stations = [ecccdatafull['Toronto City'], ecccdatafull['Center Island'], ecccdatafull['Airport'], ecccdaily['NorthYork']]


g, count, lat, lon, points, allpoints, days = gridaverage(years=['2017','2018','2019','2020','2021','2022','2023', '2024'], xticks=0.005, yticks=0.005 , truck=True, lonmax=-79.24,lonmin=-79.59, latmax=43.78,latmin=43.59, stations=stations, bar=2)

gridmap = grid_to_gdf(g, count, lat, lon, points, allpoints, days )

gridmap.set_crs('EPSG:4326', inplace=True)


#Limiting survey area
gridmap = gridmap[gridmap.intersects(downtown)]

print('Grid Created')

# ================================================================  
# Grid Metrics 
# ================================================================

# this is the longest part of the code ( in terms of runtime )


gridmap['lon'] = gridmap.geometry.centroid.x
gridmap['lat'] = gridmap.geometry.centroid.y

gridmap['water'] = sp.area(sp.intersection(gridmap.geometry, watertotal))
gridmap['land'] = sp.area(gridmap.geometry) - gridmap['water']


# Removes squares completely over water
gridmap = gridmap[gridmap['land']>0.000001]



gridmap['shoredistance'] = sp.distance(gridmap.geometry, lake) # Distance in lat lon coord

# Good estimation for distance in km. Could be made more accurate.
gridmap['shoredistance'] = haversine_series(gridmap['lon'], gridmap['lat'],
                                     gridmap['lon'] + gridmap['shoredistance']/np.sqrt(2),
                                     gridmap['lat'] + gridmap['shoredistance']/np.sqrt(2))



#Greenspace
gridmap['greenspace'] = sp.area(sp.intersection(gridmap.geometry,  parktotal))

gridmap['greenspace'] = gridmap['greenspace']/gridmap['land']



#Treedarea

gridmap['treedarea'] = sp.area(sp.intersection(gridmap.geometry,  treedareatotal))

gridmap['treedarea'] = gridmap['treedarea']/gridmap['land']



#Permeable/Impermeable Surface

gridmap['perm'] = sp.area(sp.intersection(gridmap.geometry,  permtotal))

gridmap['perm'] = gridmap['perm']/gridmap['land']

gridmap['imperm'] = 1 - gridmap['perm']


#Elevation 

gridmap['elevation'] = [tin.loc[tin.geometry.intersects(x), 'Avg_Elev'].mean() for x in gridmap.geometry]

gridmap['elevation'].bfill()


#density 

gridmap['density'] = [density.loc[density.geometry.intersects(i), 'density'].mean() for i in gridmap.geometry]


# Land use

for i in usage.label.unique():
    if i == 'watercover':
         
        gridmap[i] = sp.area(sp.intersection(gridmap.geometry,  usage[usage.label==i].geometry.values[0]))

        gridmap[i] = gridmap[i]/sp.area(gridmap.iloc[0].geometry)
    

    gridmap[i] = sp.area(sp.intersection(gridmap.geometry,  usage[usage.label==i].geometry.values[0]))

    gridmap[i] = gridmap[i]/gridmap['land']


#Saves the measured tempertures before the regression kriging
gridmap['basetemp'] = gridmap['temp']


# ================================================================
# Regression
# ================================================================'

from sklearn.linear_model import LinearRegression
from pykrige.ok import OrdinaryKriging


#piecewise by within and beyond 1km of shore

# Saving base (measured) temps

gridmap['temp'] = gridmap['basetemp']


# Want to only factor in well-studied points
gridmap.loc[(gridmap['day count']<3)|(gridmap['day count'].isna()), 'temp'] = None



# Regression Variables

mlr_variables =['imperm', 'building', 'elevation', 'grass/shrub', 
                'bare earth', 'otherpaved', 'agricultural', 'road',
                'watercover','treecanopy','shoredistance']


# Regression prediction, Residual based on regression error
gridmap['prediction'] = None
gridmap['residual'] =None


#Under 1km
model_within = LinearRegression()


#Creating a regression model
x = gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']<1),
                mlr_variables]


x_gaps = gridmap.loc[gridmap['temp'].isna()&(gridmap['shoredistance']<1),
                 mlr_variables]


y =gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']<1),
               'temp']

model_within.fit(x,y)

print(f"Coefficients: {list(zip(mlr_variables,model_within.coef_))}")
print(f"Intercept: {model_within.intercept_}")
print(f"R-squared: {model_within.score(x, y)}")

# regression prediction
gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']<1), 
            'prediction'] = model_within.predict(x)

# resiudal

gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']<1), 
            'residual'] = y - gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']<1),'prediction']

#kriging

OK = OrdinaryKriging(
    gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']<1), 'lon'], 
    gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']<1), 'lat'], 
    gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']<1), 'residual'],
    variogram_model='linear'
)

#Predicting a spatial residual bassed on the krig
pred_residual, _ = OK.execute('grid',
                              gridmap.loc[gridmap['temp'].isna()&(gridmap['shoredistance']<1), 'lon'],
                              gridmap.loc[gridmap['temp'].isna()&(gridmap['shoredistance']<1), 'lat'])



#final prediction
gridmap.loc[gridmap['temp'].isna()&(gridmap['shoredistance']<1), 'temp'] = model_within.predict(x_gaps) + pred_residual.data[:,0]


# over 1km
model_beyond = LinearRegression()

#Creating a regression model
x = gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']>=1),
                mlr_variables]

x_gaps = gridmap.loc[gridmap['temp'].isna()&(gridmap['shoredistance']>=1),
                mlr_variables]


y =gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']>=1),
                'temp']

model_beyond.fit(x,y)

print(f"Coefficients: {list(zip(mlr_variables, model_beyond.coef_))}")
print(f"Intercept: {model_beyond.intercept_}")
print(f"R-squared: {model_beyond.score(x, y)}")


# regression prediction
gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']>=1), 
            'prediction'] = model_beyond.predict(x)

# resiudal
gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']>=1), 
            'residual'] = y - gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']>=1), 
                                            'prediction']


#kriging

OK = OrdinaryKriging(
    gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']>=1), 'lon'], 
    gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']>=1), 'lat'], 
    gridmap.loc[~gridmap['temp'].isna()&(gridmap['shoredistance']>=1), 'residual'],
    variogram_model='linear'
)

#Predicting a spatial residual bassed on the krig
pred_residual, _ = OK.execute('grid',
                              gridmap.loc[gridmap['temp'].isna()&(gridmap['shoredistance']>=1), 'lon'],
                              gridmap.loc[gridmap['temp'].isna()&(gridmap['shoredistance']>=1), 'lat'])


#final prediction
gridmap.loc[gridmap['temp'].isna()&(gridmap['shoredistance']>=1), 'temp'] = model_beyond.predict(x_gaps) + pred_residual.data[:,0]



# ================================================================
# By Census Tract
# ================================================================

print('bytract start')
# Averaging by census tract

bytract = pop.copy()

bytract['temp'] = None
bytract['perm'] = None
bytract['treedarea'] = None

bytract['land'] = [sp.area(bytract.loc[x,'geometry'])-(sp.area(sp.intersection(bytract.loc[x,'geometry'], watertotal))) for x in bytract.index]


for i in bytract.index:
    tract = bytract.loc[i, 'geometry']
    landarea = bytract.loc[i,'land']
    inter = gridmap[gridmap.geometry.intersects(tract)].copy()
    if (len(inter)>0) and (sp.area(sp.union_all(inter.geometry)) >= landarea/2):
        bytract.loc[i, ['temp', 'count', 'shoredistance', 'days', 'lat', 'lon', 'elevation']] = inter['temp'].mean(), inter.loc[~inter['count'].isna(), 'count'].sum(), inter['shoredistance'].mean(), inter['day count'].max(), inter['lat'].mean(),inter['lon'].mean(), inter['elevation'].mean()


bytract['perm'] = [sp.area(sp.intersection(bytract.loc[x,'geometry'], permtotal))/bytract.loc[x,'land'] for x in bytract.index]

bytract['treedarea'] = [sp.area(sp.intersection(bytract.loc[x,'geometry'], treedareatotal))/bytract.loc[x,'land'] for x in bytract.index]


bytract = bytract[~(bytract['temp'].isna())]
bytract = bytract[~(bytract['perm'].isna())]
bytract = bytract[~(bytract['treedarea'].isna())]
bytract = bytract[~(bytract['shoredistance'].isna())]


bytract['temp'] = bytract['temp'].astype('float64')
bytract['perm'] = bytract['perm'].astype('float64')

bytract['treedarea'] = bytract['treedarea'].astype('float64')
bytract['shoredistance'] = bytract['shoredistance'].astype('float64')

bytract['imperm'] =  1 - bytract['perm']

print('bytract usage')

for i in usage.label.unique():

    if i == 'watercover':
        bytract[i] = None
        bytract[i] = [sp.area(sp.intersection(bytract.loc[x,'geometry'], usage[usage.label==i].geometry.values[0]))/bytract.area for x in bytract.index]
    
    bytract[i] = None
    bytract[i] = [sp.area(sp.intersection(bytract.loc[x,'geometry'], usage[usage.label==i].geometry.values[0]))/bytract.loc[x,'land'] for x in bytract.index]


print('bytract done')


bytract['density'] = 1000*bytract['pop']/sp.area(bytract.geometry.to_crs('EPSG:3978'))


# ================================================================
# Creating geodataframe representing bike routes as line. 
# For plotting and visualizations
# ================================================================


georoutes = gpd.GeoDataFrame(columns=['route','geometry'])
indy = 0
gr = []

for i in set(fullbikedata['route'].array):
    if len(fullbikedata[fullbikedata['route'] == i]) > 1:

        gr.append(gpd.GeoDataFrame({'route': i, 'type': 'bike'},index=[indy], 
                        geometry=[sp.LineString(fullbikedata[fullbikedata['route'] == i][['lon', 'lat']].values)]))
    indy+=1

for i in set(fulltruckdata['route'].array):
    if len(fulltruckdata[fulltruckdata['route'] == i]) > 0:
       gr.append(gpd.GeoDataFrame({'route': i, 'type': 'truck'}, index=[indy],
                         geometry=[sp.LineString(fulltruckdata[fulltruckdata['route'] == i][['lon', 'lat']].values)]))
    indy+=1


georoutes = pd.concat(gr, ignore_index=True)

georoutes[georoutes['route']=='2021-09-02'] = sp.LineString(georoutes[georoutes['route']=='2021-09-02'].geometry.values[0].coords[20:800])






