# Toronto-UHI-analysis-summer-project
Mobile data based analysis of the Urban Heat Island effect in Toronto.


This repository contains code for exploring and analyzing the Urban Heat
Island (UHI) effect in Toronto using a combination of mobile,
stationary, and geospatial datasets.

Two main modules:
- UHIdata.py → loads, cleans, and compiles datasets.
- UHIresults.py → generates visualizations, correlation plots, and maps
from processed data.

------------------------------------------------------------------------

Project Structure

    .
    ├── UHIdata.py       # Data loading, cleaning, and preprocessing
    ├── UHIresults.py    # Visualization and results generation
    ├── GeoData/         # Spatial datasets (land cover, parks, elevation, census, etc.)
    ├── stations/        # TAO and ECCC station data

------------------------------------------------------------------------

Install dependencies

        pip install numpy pandas matplotlib seaborn geopandas shapely pyproj contextily scipy scikit-learn pykrige

------------------------------------------------------------------------

Data Sources



Mobile Data (Bike & Truck)


-   Mobile emission and weather data collected via bike and truck.
-   Only the weather data is used. 
-   Stored in stations/Bike_Data/ and stations/Truck_Data/.
-   truck link (2022, 2023) : https://catalogue.ec.gc.ca/geonetwork/srv/eng/catalog.search#/metadata/e56ea8c3-1267-49a7-863b-8ae00eabf11f
  	https://catalogue.ec.gc.ca/geonetwork/srv/eng/catalog.search#/metadata/5b90420e-70dc-482a-9407-394529095a19
-   bike data link: https://borealisdata.ca/dataverse/wunchlab

Stationary Weather Data

-   Toronto Atmospheric Observatory (TAO)
	https://www.atmosp.physics.utoronto.ca/wstat/index.htm

-   Environment and Climate Change Canada (ECCC) Historical Weather Data
    — multiple stations (Toronto City, Centre Island, Airport, Oshawa,
    North York).
	https://climate.weather.gc.ca/historical_data/search_historic_data_e.html

GeoData

-   Census Boundaries & Demographics (StatsCan / CHASS)
    -   https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21
    -   https://datacentre.chass.utoronto.ca/census/
-   Toronto Open Data Portal
    -   Parks, Waterbodies, Land Cover, Elevation (TIN), etc.
    -   https://open.toronto.ca/
-   Land Cover (pre-processed into
    GeoData/landcover.geojson). Included in repo.

------------------------------------------------------------------------

Usage

1. Data Preparation

All datasets must be downloaded into the correct folders before running.
Paths are defined in UHIdata.py:

    MAIN_DIR = '/Users/<your-username>/Desktop/CGCS/Heat_Island/'
    STATION_DIR = MAIN_DIR + 'stations'
    GEODATA_DIR = MAIN_DIR + 'GeoData'

Modify these paths to point to your local data directories.

2. Loading Data

UHIdata.py will automatically load and process datasets into:
- fullbikedata — concatenated mobile bike data with anomalies
- fulltruckdata — concatenated truck data with anomalies
- TAOdatafull — TAO station dataset
- ecccdatafull — dictionary of ECCC station datasets
- gridmap — grid-averaged anomalies with geospatial features
- bytract — anomalies and features aggregated by census tract
- georoutes — GeoDataFrame of bike/truck routes (for visualization)

Run once to generate and cache these objects:

    python UHIdata.py

3. Visualizations

Use UHIresults.py to generate plots and maps. Example:

    # Plot a specific bike route relative to TAO station
    plot_route(day="2021-07-15", station="TAO")

    # Plot anomaly maps aggregated by census tract with overlays
    plot_maps(df=bytract, metric="temp", routes=True, eccc=True, tao=True)

