# Earth's Infrared Background
Code for reproducing the analysis and figures in Shamir and Gerber (20??).

## Data
Interpolated Outgoing Longwave Radiation data provided by the NOAA PSL, Boulder, Colorado, USA, was sourced from their website at https://psl.noaa.gov/data/gridded/data.olrcdr.interp.html.

Twice daily estimates (individual observations) were used. The code expects the relevant file (olr.2xdaily.1979-2022.nc) to be in the data folder. 

## Random realization of the background
A random realization of the background was used for statistical analysis and for estimating the foreground, and is needed for the plotting code to work. 

To generate a random realization use: `./code/comp/ou-realization.py`

## Space/time analysis
Having obtained the data and generated a random realization of the background, the space/time analysis is computed in:
- `./code/comp/analysis-observations.py`
- `./code/comp/analysis-observations-raw.py`
- `./code/comp/analysis-ou.py`

## Statistical analysis
The plotting code loads pre-computed p-values, computed in:
- `./code/comp/bootstrap-grid-space.py`
- `./code/comp/bootstrap-spectral-space.py`

## Plotting code
The code for reproducing the figures is provided in Jupyter Notebooks, located in `./code/plot`. 

