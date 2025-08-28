"""
Urban Heat Island Effect Exploration in Toronto
Created: Summer 2025
Author: Noah Vaillant

This script visualizes and produces results and correlations for UHI
analysis. All data in UHI data.

"""




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

from UHIdata import fullbikedata, fulltruckdata, TAOdatafull, ecccdatafull, gridmap, bytract, georoutes




#=============================================================================
# Plotting Functions
# =============================================================================

# Using bike data 
def plot_route(station=None, day=None, month=None, 
               year=['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'], 
               truck=False, bike=True):
    """
    Plots by route with options for station, day, month and type of mobile data.
    
    Parameters
    ----------
    station: station df or None, optional
        Station's name. Data to compare to mobile temperature. The default is None.
        Possible names are 'TAO' and anything in eccc data's keys.
        Eccc and Truck are currently not compatible.
    day : list[str] or str, optional
        Days of which to plot the routes for. The default is None.
    month : list[str] or str, optional
        Months of which to plot the routes for. The default is None.
    year : list[str] or str, optional
        Years of which to plot routes for. The default is 2017-2023.

    

    Returns
    -------
    None.

    """

    stationlabel = station

    if station not in ['TAO', *ecccdatafull.keys(), None]:
        print('error: Invalid Station')
        return None

    elif station in ecccdatafull.keys():
        station = ecccdatafull[station]
        if truck:
            print('error: ECCC data not compatible with truck data')
            return None

    elif station == 'TAO':
        station = TAOdatafull

    else:
        station = ''
    
    

    if (not 'bike') and (not 'truck'):
        print('error: neither bike or truck')
        return None
    
    if type(day) is str:
        day = [day]
    if type(month) is str:
        month = [month]
    if type(year) is str:
        year = [year]


    fig, ax = plt.subplots(figsize=(10,10))
    
    if day is not None:
        if len(station) == 0: 
            
            
            
            if bike:
                norm = mpl.colors.Normalize(
                fullbikedata[fullbikedata['route'].isin(day)]['temp'].min(), 
                 fullbikedata[fullbikedata['route'].isin(day)]['temp'].max()
                 )
                ax.scatter(fullbikedata[fullbikedata['route'].isin(day)]['lon'],
                        fullbikedata[fullbikedata['route'].isin(day)]['lat'],
                        c=fullbikedata[fullbikedata['route'].isin(day)]['temp'],
                        s=0.2
                        )
            else:
                norm = mpl.colors.Normalize(
                fulltruckdata[fulltruckdata['route'].isin(day)]['temp'].min(), 
                 fulltruckdata[fulltruckdata['route'].isin(day)]['temp'].max()
                 )
            
            if truck:
                ax.scatter(fulltruckdata[fulltruckdata['route'].isin(day)]['lon'],
                            fulltruckdata[fulltruckdata['route'].isin(day)]['lat'],
                            c=fulltruckdata[fulltruckdata['route'].isin(day)]['temp'],
                            s=0.2)
        else:
            
            if bike:
                norm = mpl.colors.Normalize(
                fullbikedata[fullbikedata['route'].isin(day)]['anom'].min(), 
                 fullbikedata[fullbikedata['route'].isin(day)]['anom'].max()
                 )
                
                ax.scatter(fullbikedata[fullbikedata['route'].isin(day)]['lon'],
                            fullbikedata[fullbikedata['route'].isin(day)]['lat'],
                            c=fullbikedata[fullbikedata['route'].isin(day)]['temp']-
                            station[station['route'].isin(day)]['temp'],
                            s=0.2, norm=norm
                            )
            else:
                norm = mpl.colors.Normalize(
                fulltruckdata[fulltruckdata['route'].isin(day)]['anom'].min(), 
                 fulltruckdata[fulltruckdata['route'].isin(day)]['anom'].max()
                 )
            if truck :
                
                ax.scatter(fulltruckdata[fulltruckdata['route'].isin(day)]['lon'],
                            fulltruckdata[fulltruckdata['route'].isin(day)]['lat'],
                            c=fulltruckdata[fulltruckdata['route'].isin(day)]['anom'],
                            s=0.2, norm=norm
                            ) 

    
    elif month is not None:

        inmonth = fullbikedata[fullbikedata['month'].isin(month)].copy()
        inyear  = inmonth[inmonth['year'].isin(year)].copy()
        

        if len(station) == 0:
            
            if bike: 
                norm = mpl.colors.Normalize(
                inyear['temp'].min(), 
                inyear['temp'].max()
                 )
                
                ax.scatter(inyear['lon'],
                       inyear['lat'],
                        c=inyear['temp'],
                        s=0.2
                        )
            else:
                norm = mpl.colors.Normalize(
                fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['temp'].min(), 
                fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['temp'].max()
                 )

            if truck:
                ax.scatter(fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['lon'],
                            fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['lat'],
                            c=fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['temp'],
                            s=0.2
                            )
            
            
        else:
            if bike:
                norm = mpl.colors.Normalize(
                inyear['anom'].min(), 
                inyear['anom'].max()
                 )
                ax.scatter(inyear['lon'],
                        inyear['lat'],
                            c=inyear['temp']-
                            station[(station['month'].isin(month)) & (station['year'].isin(year))]['anom'],
                            s=0.2, norm=norm
                            )
       
            else:
                norm = mpl.colors.Normalize(
                fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['anom'].min(), 
                fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['anom'].max()
                 )
                
            if truck:
                norm = mpl.colors.Normalize(vmin=-5, vmax=5)
                ax.scatter(fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['lon'],
                            fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['lat'],
                            c=fulltruckdata[(fulltruckdata['route'].isin(month))&(fulltruckdata['year'].isin(year))]['anom'],
                            s=0.2, norm=norm
                            )
        
      
       

                
    else:


        inyear  = fullbikedata[fullbikedata['year'].isin(year)]

        if len(station) == 0:
            if bike:
                norm = mpl.colors.Normalize(
                inyear['temp'].min(), 
                inyear['temp'].max()
                 )
                ax.scatter(inyear['lon'],
                       inyear['lat'],
                        c=inyear['temp'],
                        s=0.2
                        )
            else:
                norm = mpl.colors.Normalize(
                fulltruckdata[(fulltruckdata['year'].isin(year))]['temp'].min(), 
                fulltruckdata[(fulltruckdata['year'].isin(year))]['temp'].max()
                 )

            if truck:
                tinyear = fulltruckdata[fulltruckdata['year'].isin(year)]
                ax.scatter(tinyear['lon'],
                            tinyear['lat'],
                            c=tinyear['temp'],
                            s=0.2, norm=norm
                            )
        else:
            if bike:
                norm = mpl.colors.Normalize(
                inyear['anom'].min(), 
                inyear['anom'].max()
                 )
                norm = mpl.colors.Normalize(vmin=-5, vmax=5)
        
                ax.scatter(inyear['lon'],
                        inyear['lat'],
                            c=inyear['temp']-
                            station[station['year'].isin(year)]['temp'],
                            s=0.2, norm=norm
                            )
            else:
                norm = mpl.colors.Normalize(
                fulltruckdata[(fulltruckdata['year'].isin(year))]['anom'].min(), 
                fulltruckdata[(fulltruckdata['year'].isin(year))]['anom'].max()
                 )

            if truck:

                tinyear = fulltruckdata[fulltruckdata['year'].isin(year)]
                norm = mpl.colors.Normalize(vmin=-5, vmax=5)
                ax.scatter(tinyear['lon'],
                            tinyear['lat'],
                            c=tinyear['anom'],
                            s=0.2, norm=norm
                            )
    if len(station)>0:
        plt.colorbar(ax.collections[0], ax=ax, 
                     label='Anomaly Temperture (Background:' + stationlabel +') (C)', orientation='vertical', 
                     shrink=1)
    else:
        plt.colorbar(ax.collections[0], ax=ax, 
                     label='Temperture (C)', orientation='vertical', 
                     shrink=1)

                
    plt.grid()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()



def plot_timeseries(day=None, station=None):
    """
    

    Parameters
    ----------
    day : list[str] or str, optional
        Days of which to plot the routes for. The default is None.
    month : list[str] or str, optional
        Months of which to plot the routes for. The default is None.
    year : list[str] or str, optional
        Years of which to plot routes for. The default is 2017-2023.
    station: station df or None, optional
        Station's name. Data to compare to mobile temperature. The default is None.
        Possible names are 'TAO' and anything in eccc data's keys.
        Eccc and Truck are currently not compatible.
    
    
    Returns
    -------
    None.

    """
    

    if (day is not None) and (type(day) is not list):
        day = [day]
    
    elif day is None:
        print('day is none')
        return
    


    if (station is not None) and (type(station) is not list):
        station = [station]

    if type(station[0]) is str:
        for i in range(len(station)):
            if station[i] not in ['TAO', *ecccdatafull.keys(), None]:
                print('error: Invalid Station')
                return None

            elif station[i] in ecccdatafull.keys():
                station[i] = ecccdatafull[s]
                

            elif station[i] == 'TAO':
                station[i] = TAOdatafull

            else:
                print('invalid station')
                return
        


    


    if station is not None:
        fig, [timeseries, comparison]  = plt.subplots(ncols=1, nrows=2, height_ratios=(3,2), figsize=(12,4))
        fig.subplots_adjust(hspace=0.3)
        timeseries.set_xlabel('Time of Day (h)')
        comparison.set_xlabel('Time of Day (h)')
        timeseries.set_ylabel('Temperature (°C)')
        comparison.set_ylabel('Anomaly (°C)')
        timeseries.grid()
        comparison.grid()
        plt.tight_layout()
    else:
        
        fig, timeseries = plt.subplots(ncols=1, nrows=1)
        timeseries.set_xlabel('Time of Day (h)')
        timeseries.set_ylabel('Temperature (C)')
        timeseries.grid()
        

    colors =[]
    for d in day:

        line, = timeseries.plot(fullbikedata[fullbikedata['route'] == d]['tod'],
                                fullbikedata[fullbikedata['route'] == d]['temp'],
                                c='black', label='Bike Data')
        timeseries.set_xlim(fullbikedata[fullbikedata['route'] == d]['tod'].min()- 0.25,fullbikedata[fullbikedata['route'] == d]['tod'].max() + 0.25)
        comparison.set_xlim(timeseries.get_xlim())
        
    if station is not None:
        for s in station:
            line, = timeseries.plot(fullbikedata[fullbikedata['route'] == day[0]]['tod'], s[s['route'] == day[0]]['temp'], ls='--')
            c = line.get_color()
            

            colors.append(c)

            timeseries.scatter(s[(s['route'] == day[0]) & s['original']]['tod'], s[(s['route'] == day[0]) & s['original']]['temp'], s=12, color=c, label='TAO Data')
            

            comparison.plot(fullbikedata[(s['route'] == day[0]) & s['original']]['tod'], fullbikedata[(s['route'] == day[0]) & s['original']]['anom'],'o--', markersize=4, color='orange', label='Bike Temp - TAO Temp')


        
        for d in day[1:]:
            for i in range(len(station)):
                line = timeseries.plot(station[i][station[i]['route'] == d]['tod'], station[i][station[i]['route'] == d]['temp'], c=colors[i], ls='--')

                comparison.scatter(fullbikedata[(station[i]['route'] == d) & station[i]['original']]['tod'], fullbikedata[(station[i]['route'] == d) & station[i]['original']]['temp']- station[i][(station[i]['route'] == d) & station[i]['original']]['temp'], c=colors[i], s=2)

                
                comparison.scatter(fullbikedata[(station[i]['route'] == d)]['tod'], fullbikedata[(station[i]['route'] == d)]['anom'],
                    c=colors[i], s=2, label='Anomaly')
                
    timeseries.legend()
    comparison.legend()
    plt.tight_layout()



def long_timeseries(station, start='2017', end='2025', scale=1, 
                 months =['05','06','07','08'] ):
    """
    Plot long-term anomaly time series between bike data and a reference station.

    Parameters
    ----------
    station : str representing station
        Reference station data with 'temp' and datetime index.
    start : str, optional
        Start year (default '2017').
    end : str, optional
        End year (default '2025').
    scale : int or float, optional
        Bin size in years for averaging. Default is 1.
    months : str or list of str, optional
        Months to include (default ['5','6','7','8']).

    Returns
    -------
    None
        Displays a scatter plot of anomalies vs. time.
    """
    

    if station not in ['TAO', *ecccdatafull.keys(), None]:
        print('error: Invalid Station')
        return None

    elif station in ecccdatafull.keys():
        name = station
        station = ecccdatafull[station]
        

    elif station == 'TAO':
        name = station
        station = TAOdatafull

  

    
    if type(months) == str:
        months = [months]
    s=(start)
    e=(end)
    scale = str(int(12*scale))
    
    x = np.arange(np.datetime64(s, 'Y'), 
                  np.datetime64(e, 'Y'), 
                  np.timedelta64(scale, 'M'))
    
  
    y=np.zeros(len(x))

    inmonth = fullbikedata['month'].isin(months)
    if type(station) == list:
        for i in station:
            long_timeseries(station, start=start, end=end, scale=scale)
        
    else:

        for i in range(0, len(x)-1):
            y[i] = np.average(fullbikedata[inmonth & (fullbikedata.index>x[i])&
                                       (fullbikedata.index<x[i+1])]['temp'].array 
                                     -station[(station['month'].isin(months)) & (station.index>x[i])&
                                      (station.index<x[i+1])]['temp'].array)

    x= x[:-1]
    y = y[:-1]
    
    plt.scatter(x, y, label=name)
    plt.ylabel('Anomaly Temperture (C)')
    plt.xlabel('year')
    plt.show()




# Using a gridaverage grid

def latlonslice(grid, latitude, longitude, lat=None, lon=None,):

    """
    Plots a latitude / longitude slice of average anomaly temperture.
    Uses grid, latitude, longitude from gridaverage 

    Parameters
    ----------

    grid: ndarray of shape (n_lat, n_lon): Average anomaly temperature per gridsquare.
    latitude: pandas Series length = n_lat: latitudes matching to rows of grid
    longitude: pandas Series length = n_lon: latitudes matching to columns of grid
    lat, lon : float : input only one, lat or lon where slice is to be taken.

    Returns
    -------
    None. Plots.
   

    """

    if lat is lon is None:
        return "Can't leave both blank"
    if lat is not None and lon is not None:
        return "Can't have both"
    
    xticks = lon.array[1] - lon.array[0]
    yticks = lat.array[1] - lat.array[0]
    if lat is not None:
        if type(lat) is list:
            for l in lat:
                 i = latitude[abs(latitude - l) < yticks/2].index.array
            
                 plt.plot(longitude.array,  grid[i].flatten(), '--o', label=l)
        else:

            i = latitude[abs(latitude - lat) < yticks/2].index.array
            

            
            plt.plot(longitude.array,  grid[i].flatten(), '--o', label=lat)
            
    if lon is not None:
        if type(lon) is list:
            for l in lon:
                 i = latitude[abs(longitude - l) < xticks/2].index.array
            
                 plt.plot(latitude.array,  grid[:,i].flatten(), '--o', label=l)
        else:
            
            i = latitude[abs(latitude - lon) < xticks/2].index.array
            

            
            plt.plot(latitude.array,  grid[:,i].flatten(), '--o', label=lon)




# Using the gridmap or bytract gdfs

def corrplot(df: pd.DataFrame, metric: str, binsize=None, kind='per land area'):
    
    """

    Plot correlation between a geographic metric and average anomaly temperature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a temperature column ('temp') and the chosen metric.
    metric : str
        Column name of the metric to compare with temperature.
    binsize : float, optional
        Bin width for grouped averages and error bars. If None (default), no binning is shown.
    kind : {'per land area', 'km'}, optional
        Controls axis labeling convention. Default is 'per land area'.

    Returns
    -------
    None
        Displays a scatter plot with regression line, correlation coefficient, and optional binned averages.
    """

    df= df.copy()


    data = df[metric].copy()
    temps = df.temp.copy()


    if df is gridmap:

        sns.lmplot(data=df, y='temp', x=metric, line_kws={'color': 'red', 'lw': 1.5}, scatter_kws={'s': 1, 'color': 'black', 'alpha': 0.3}, height=3, aspect=1)


    else:
        sns.lmplot(data=df, y='temp', x=metric, line_kws={'color': 'red', 'lw': 1.5}, scatter_kws={'s': 1, 'color': 'black', 'alpha': 0.8}, height=5, aspect=1)
    # print(stats.pearsonr(data.array, temps.array))

    r_value = stats.pearsonr(data, temps)[0]
    plt.text(0.05, 0.95, f'$r = {r_value:.2f}$', transform=plt.gca().transAxes)

    if binsize is not None:
        ff=  ff = df.loc[:,[metric, 'temp']]
        ff = ff.groupby(pd.cut(df[metric], np.arange(0, df[metric].max()+binsize, binsize))).agg(['mean', 'std'])

        

        plt.errorbar(ff[metric]['mean'], ff['temp']['mean'], yerr=np.sqrt(ff['temp']['std']**2), fmt='o', elinewidth=1, markersize=4, label='1 km bins', alpha=0.8)


    plt.legend(loc='lower right', fontsize=10)
    if kind=='km':
        if metric == 'shoredistance':
            plt.xlabel('Distance from Shoreline (km)')
        else:
            plt.xlabel('Average Elevation')
        plt.ylabel('Average Anomaly (°C)')

    
    elif kind=='per land area':

        if metric=='treedarea':
            plt.xlabel('Treed Area / Land Area')
        
        if metric=='imperm':
            plt.xlabel('Impermeable Area / Land Area')

        if metric=='density':
            plt.xlabel('Person per km^2')

        if metric=='treecount':
            plt.xlabel('Tree per km^2')
        
        if metric =='treecanopy':
            plt.xlabel('Tree Covered Area / Land Area')

        else:
            plt.xlabel(f'{metric} / Land Area')

        plt.ylabel('Average Anomaly (°C)')
    # plt.show()
    plt.show()


def plot_maps(df=bytract,metric='temp',routes=False, eccc=False, tao=False, countoverzero=False):

    """
    Plot spatial maps of temperature anomalies or geographic metrics across tracts or grids.

    Parameters
    ----------
    df : GeoDataFrame, optional
        Spatial dataset to plot. Default is `bytract`.
    metric : str, optional
        Variable to visualize. Options include:
        - 'temp' (default): average anomaly (°C).
        - 'count' : number of observations.
        - 'elevation' : elevation in meters.
        - 'imperm' : impermeable surface fraction.
        - 'treedarea' : tree cover fraction.
    routes : bool, optional
        If True, overlay bike/truck routes. Default is False.
    eccc : bool, optional
        If True, mark ECCC station locations. Default is False.
    tao : bool, optional
        If True, mark TAO station location. Default is False.
    countoverzero : bool, optional
        If True, only plot cells/tracts with at least 2 days of data. Default is False.

    Returns
    -------
    None
        Displays a choropleth map with optional overlays and basemap.
    """



    mpl.rcParams.update({'font.size': 16})

    coordtransformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


   
    tractproj = bytract.to_crs('EPSG:3857')
    gridproj = gridmap.to_crs('EPSG:3857')
    routesproj = georoutes.set_crs('EPSG:4326').to_crs('EPSG:3857')


    geotaodata =gpd.GeoDataFrame(TAOdatafull.iloc[0:5], geometry= gpd.points_from_xy(TAOdatafull['lon'].iloc[0:5],TAOdatafull['lat'].iloc[0:5]))
    taopoint = geotaodata.set_crs('EPSG:4326').to_crs('EPSG:3857').iloc[0:2]
    geoecccdata = {}
    
    geoecccdata['Toronto City'] = gpd.GeoDataFrame(ecccdatafull['Toronto City'].iloc[0:5], 
                               geometry= gpd.points_from_xy( ecccdatafull['Toronto City']['lon'].iloc[0:5],
                                                  ecccdatafull['Toronto City']['lat'].iloc[0:5]))
    geoecccdata['Toronto City'] = gpd.GeoDataFrame(ecccdatafull['Toronto City'].iloc[0:5], 
                               geometry= gpd.points_from_xy( ecccdatafull['Toronto City']['lon'].iloc[0:5],
                                                  ecccdatafull['Toronto City']['lat'].iloc[0:5]))
    
    geoecccdata['Center Island'] = gpd.GeoDataFrame(ecccdatafull['Center Island'].iloc[0:5], 
                               geometry= gpd.points_from_xy( ecccdatafull['Center Island']['lon'].iloc[0:5],
                                                  ecccdatafull['Center Island']['lat'].iloc[0:5]))

    ecccityproj = geoecccdata['Toronto City'].set_crs('EPSG:4326').to_crs('EPSG:3857').iloc[0:2]
    eccislandproj = geoecccdata['Center Island'].set_crs('EPSG:4326').to_crs('EPSG:3857').iloc[0:2]
    

    fig, ax = plt.subplots(figsize=(14, 7))

    if metric == 'temp':
        cmap = mpl.cm.coolwarm
        bounds = [-3,-2, -1, 0, 1,2, 3]
   
        cmaplist = [cmap(i) for i in range(cmap.N)]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap',cmaplist , cmap.N)
        
    elif metric =='count':

        cmap = mpl.cm.Reds
        bounds = [0,100,200,300,400,500]
        # bounds = [0,2,4,6,8,10,12,14,16,18,20]
        cmaplist = [cmap(i) for i in range(cmap.N)]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap',cmaplist , cmap.N)
        
    elif metric =='elevation':

        cmap = mpl.cm.coolwarm
        bounds = [80,90,100,110,120,130,140,150,160,170]
        # bounds = [0,2,4,6,8,10,12,14,16,18,20]
        cmaplist = [cmap(i) for i in range(cmap.N)]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap',cmaplist , cmap.N)
        ax.set_title('Elevation')
    
    elif metric =='imperm':
        
        gridproj['imperm']*=100
        tractproj['imperm']*=100
        cmap = mpl.cm.coolwarm
        bounds = [10,20,30,40,50,60,70,80]
        # bounds = [0,2,4,6,8,10,12,14,16,18,20]
        cmaplist = [cmap(i) for i in range(cmap.N)]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap',cmaplist , cmap.N)
        ax.set_title('Impermeable Surface')
    
    elif metric =='treedarea':
        
        gridproj['treedarea']*=100
        tractproj['treedarea']*=100
        cmap = mpl.cm.coolwarm
        bounds = [0,2.5,5,7.5,10,12.5,15]
        # bounds = [0,2,4,6,8,10,12,14,16,18,20]
        cmaplist = [cmap(i) for i in range(cmap.N)]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap',cmaplist , cmap.N)
        ax.set_title('Tree Cover')
    
    

    
    if df is bytract:
        
         if countoverzero:
            tractproj[tractproj['days']>2].plot(ax=ax, column=metric, linewidth=1, alpha=0.8, edgecolor='white', cmap=cmap, norm=norm)
         else:
            
            tractproj.plot(ax=ax, column=metric, linewidth=1, alpha=0.8, edgecolor='white', cmap=cmap, norm=norm)
            

   
    elif countoverzero and df is gridmap:
                gridproj[gridproj['day count']>=2].plot(ax=ax, column=metric, cmap=cmap, norm=norm, edgecolor='black', linewidth=0.1,)

    if df is gridmap and not countoverzero:
        gridproj.plot(ax=ax, column=metric, cmap=cmap, norm=norm, edgecolor='white', linewidth=0.1,)
        gridproj[gridproj['day count']>2].plot(ax=ax, column=metric, cmap=cmap, norm=norm, edgecolor='black', linewidth=0.3)
        
        
    
    if metric == 'temp':
        plt.colorbar(ax.collections[0], ax=ax, label='Average Anomaly (°C)', orientation='vertical', shrink=1)

    elif metric == 'count':
         plt.colorbar(ax.collections[0], ax=ax, label='Count', orientation='vertical', shrink=1)

    elif metric == 'elevation':
         plt.colorbar(ax.collections[0], ax=ax, label='Elevation (m)', orientation='vertical', shrink=1)

    elif metric == 'imperm':
         plt.colorbar(ax.collections[0], ax=ax, label='Impermeable Surface %', orientation='vertical', shrink=1)
    
    elif metric == 'treedarea':
         plt.colorbar(ax.collections[0], ax=ax, label='Tree Cover %', orientation='vertical', shrink=1)


    if tao:
        taopoint.plot(ax=ax, color='white', markersize=100, marker='*', label='TAO Station', zorder=500)

    if eccc:
        ecccityproj.plot(ax=ax, color='green', markersize=120, marker='*', label='ECCC Toronto City Station', zorder=500)
        eccislandproj.plot(ax=ax, color='blue', markersize=120, marker='*', label='ECCC Centre Island Station', zorder=500)

    
    xticks_merc = ax.get_xticks()
    yticks_merc = ax.get_yticks()

    # ax.set_xlim([xticks_merc[0] - 100, xticks_merc[-1] + 100])
    # ax.set_ylim([yticks_merc[0] - 100, yticks_merc[-1] + 100])

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    xticks_lon, _ = coordtransformer.transform(xticks_merc, [0]*len(xticks_merc))
    _, yticks_lat = coordtransformer.transform([0]*len(yticks_merc), yticks_merc)

    ax.set_xticklabels(xticks_lon.round(3))
    ax.set_yticklabels(yticks_lat.round(3))


    if routes:
        routesproj.plot(ax=ax, color='black', linewidth=1, label='Bike and Truck Routes', alpha=0.7, linestyle='--')

    ax.set_xlim(xticks_merc[0], xticks_merc[-1])
    ax.set_xticks(xticks_merc)

    ax.set_ylim(yticks_merc[0], yticks_merc[-1])
    ax.set_yticks(yticks_merc)

    
    ax.set_xticklabels(xticks_lon.round(3))

    ax.set_yticklabels(yticks_lat.round(3))

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    # plt.savefig("high_res_map2.png", dpi=300)
    fig.show()

