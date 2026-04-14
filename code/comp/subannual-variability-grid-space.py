import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import pyshtools as sh
import xarray as xr
import netCDF4 as nc
from pathlib import Path
import time
from multiprocessing import Pool
import spectral_analysis


# grid
ntrunc=72
spd = 2

nDayWin = 360
nDaySkip = -180
nSampWin = nDayWin * spd


if __name__ == "__main__":

    base_dir = (Path(__file__).parent / "../../").resolve()
    data_dir = base_dir / "data"

    # obs
    file_name = "olr.2xdaily.1979-2022.nc"

    ds_observed = xr.open_dataset(str(data_dir / file_name))
    ds_observed = ds_observed.sel(time=ds_observed.time.dt.year.isin(range(1981, 2011, 1)))

    # extract
    F_obs = ds_observed.olr.values
    lond = ds_observed.lon.values
    latd = ds_observed.lat.values

    # ou
    file_name = "ou-realization-2024-epsilon0-5.8-lambda0-0.06-tau0-2.3.nc"

    ds_ou = xr.open_dataset(str(data_dir / file_name))

    # extract
    F_ou = ds_ou.F.values


    # empty SHGrid instance for lon lat
    x = sh.SHGrid.from_zeros(lmax=ntrunc, grid='GLQ')

    # add cyclic longitude (originally 0. to 357.5 every 2.5  -- adding 360)
    F_obs = np.dstack([F_obs, F_obs[:, :, 0]])
    lond = np.hstack([lond, 360.])

    # # interpolate longitudes to the gaussian grid
    F_out = interp1d(lond, F_obs, axis=2, fill_value='extrapolate')
    F_obs = F_out(x.lons())

    # # interpolate latitudes to the gaussian grid
    F_out = interp1d(latd, F_obs, axis=1, fill_value='extrapolate')
    F_obs = F_out(x.lats())

    lond = x.lons()
    latd = x.lats()

    # dimensions
    ntim = F_obs.shape[0]
    nlat = F_obs.shape[1]
    nlon = F_obs.shape[2]

    nDayTot = ntim // spd

    # Number of samples to skip between window segments.
    # Negative number means overlap
    nSampSkip = nDaySkip * spd

    # Count the number of available samples
    nWindow = (ntim - nSampWin) // (nSampWin + nSampSkip) + 1

    # remove dominant signals
    F_obs = spectral_analysis.remove_dominant_signals(F_obs, ntim, spd, nDayTot, nDayWin, conserve_mean=False)
    # F_ou = spectral_analysis.remove_dominant_signals(F_ou, ntim, spd, nDayTot, nDayWin, conserve_mean=False)

    # window
    Fw_obs = spectral_analysis.windows(F_obs, nSampWin, nSampSkip, nWindow, nlat, nlon)
    Fw_ou = spectral_analysis.windows(F_ou, nSampWin, nSampSkip, nWindow, nlat, nlon)

    # stds
    STD_obs = np.mean(np.std(Fw_obs ,axis=1), axis=0)
    STD_ou = np.mean(np.std(Fw_ou ,axis=1), axis=0)

    # some power is lost when using hanning window
    STD_obs *= (8. / 3.)**0.5
    STD_ou *= (8. / 3.)**0.5


    # write to netcdf file
    file_name = "subaanual-variability-grid-window-360-skip-180.nc"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(data_dir / file_name), 'w', format='NETCDF4')

    lat = ds.createDimension('latg', nlat)
    lon = ds.createDimension('long', nlon)

    lat = ds.createVariable('latg', 'f8', ('latg'))
    lon = ds.createVariable('long', 'f8', ('long'))

    STDs_obs = ds.createVariable('std_obs', 'f8', ('latg', 'long'))
    STDs_ou = ds.createVariable('std_ou', 'f8', ('latg', 'long'))

    lat[:] = latd[:]
    lon[:] = lond[:]

    STDs_obs[:, :] = STD_obs[:, :]
    STDs_ou[:, :] = STD_ou[:, :]

    ds.close()

