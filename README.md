# Entalpi public dataset Kona 2022

## Analysis description
This analysis has been set up to show how I would compare performances using time gaps over distance travelled. The jupyter notebook file [analysis.ipynb](./analysis.ipynb) contains a workflow and full explaination showing how I did this, but you can also just view the main outputs of the analysis in the [figures](./figures/) folder. Custom functions used in the analysis have been written and stored in the [functions.py](./functions.py) file. 

## Setting up the environment
This analysis environment is best setup using conda. A key package used to visualize the geospatial data is Geopandas – its installation is highly sensitive, prone to breaking and doesn't work with pip. As such, it is best to install the provided conda environment and run the analysis from there. 

```
conda env create -f environment.yml
```

## About the dataset
This repository contains a dataset with data from Gustav Iden's and Kristian Blummenfelt's run on the 2022 Ironman world championship in Kona.
All the data in this repository is published under the "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)" license. For more information about this license, read the full license [here](LICENSE) or more information about it [here](https://creativecommons.org/licenses/by-nc/4.0/).

## Description of the dataset
This repository contains data from Gustav's and Kristian's run during the 2022 Ironman world championships.
The data was recorded on their sport watches. Not all the data that was collected is published, for privacy and competive reasons.
The data that is published is:
- speed [m/s]
- cadence [/min]
- stride length [m]
- heartrate [bpm]
- elevation [m]
- gps location (latitude and longitude) [°]
- core temperature [°C] (only for Kristian)
- skin temperature [°C] (only for Kristian)

The data from both athletes is stored in 2 separate .csv files ([Gustav](./gustav_iden_copyright_entalpi_as.csv), [Kristian](./kristian_blummenfelt_copyright_entalpi_as.csv)) that are identical in format (order of columns might differ):
```csv
datetime,latitude,longitude,speed,elevation,heartrate,cadence,core_temperature,skin_temperature,stride_length
2022-10-08 21:27:33+00:00,19.63866636157036,-155.9970761463046,4.012,9.199999999999989,140.0,89.0,38.86000061035156,34.20000076293945,1.406
2022-10-08 21:27:34+00:00,19.63863903656602,-155.99709006026387,4.012,9.199999999999989,140.0,89.0,38.86000061035156,34.20000076293945,1.4
2022-10-08 21:27:35+00:00,19.6386088617146,-155.99710405804217,4.003,9.199999999999989,139.0,89.0,38.86000061035156,34.20000076293945,1.409
...
```

