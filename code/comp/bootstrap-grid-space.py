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
nboots = 5000
ntrunc=72
spd = 2

nDayWin = 360
nDaySkip = -180
nSampWin = nDayWin * spd

frequency = np.fft.fftfreq(nSampWin, 1./spd)
ff = frequency[0:int(nSampWin/2)]
ll = np.arange(0, ntrunc+1, 1)
mm = np.arange(-ntrunc, ntrunc+1, 1)


def bootstrap(seed):

    rng = np.random.default_rng(seed)

    boot_statistic = np.zeros((nlat, nlon))

    for i in range(nlat):
        for j in range(nlon):

            samp = rng.choice(np.concatenate((FFw_obs[:, i, j], FFw_ou[:, i, j])), size=2*nWindow, replace=True)

            FFw_1 = samp[0:nWindow]
            FFw_2 = samp[nWindow:2*nWindow]

            boot_statistic[i, j] = np.mean(FFw_1, axis=0) / np.mean(FFw_2, axis=0)

    return boot_statistic


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
    x = sh.SHGrid.from_zeros(lmax=72, grid='GLQ')
    
    # add cyclic longitude (originally 0. to 357.5 every 2.5  -- adding 360)
    F_obs = np.dstack([F_obs, F_obs[:, :, 0]])
    lond = np.hstack([lond, 360.])
    
    # interpolate longitudes to the gaussian grid
    F_out = interp1d(lond, F_obs, axis=2, fill_value='extrapolate')
    F_obs = F_out(x.lons())
    
    # interpolate latitudes to the gaussian grid
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

    # variance
    FFw_obs = np.mean(np.abs(Fw_obs)**2 ,axis=1)
    FFw_ou = np.mean(np.abs(Fw_ou)**2 ,axis=1)

    # some power is lost when using hanning window
    FFw_obs *= 8. / 3.
    FFw_ou *= 8. / 3.

    # statistic
    statistic = np.zeros((nlat, nlon))

    for i in range(nlat):
        for j in range(nlon):
            statistic[i, j] = np.mean(FFw_obs, axis=0)[i, j] / np.mean(FFw_ou, axis=0)[i, j]

    # call bootstrap in parallel over boots
    start_time = time.time()
    
    seeds = np.array([2024 + i for i in range(nboots)])
    
    with Pool(10) as p:
        # boot_statistic_p = zip(*p.map(bootstrap, seeds))
        boot_statistic_p = p.map(bootstrap, seeds)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # unpack back to grid
    boot_statistic = np.zeros((nboots, nlat, nlon))
    
    for b in range(nboots):
        boot_statistic[b, :, :] = boot_statistic_p[b]


    # p-value
    p_value = np.zeros((nlat, nlon))
    
    for i in range(nlat):
        for j in range(nlon):
    
            if (statistic[i, j] > 1):
                p_value[i, j] = (len(boot_statistic[boot_statistic[:, i, j] > statistic[i, j], i, j]) / nboots +
                                 len(boot_statistic[boot_statistic[:, i, j] < 1/statistic[i, j], i, j]) / nboots)
    
            if (statistic[i, j] < 1):
                p_value[i, j] = (len(boot_statistic[boot_statistic[:, i, j] < statistic[i, j], i, j]) / nboots +
                                 len(boot_statistic[boot_statistic[:, i, j] > 1/statistic[i, j], i, j]) / nboots)



    # write to netcdf file
    file_name = "boot-statistics-grid-space-realization-2024-epsilon0-5.8-lambda0-0.06-tau0-2.3.nc"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(data_dir / file_name), 'w', format='NETCDF4')
    
    samples = ds.createDimension('sample', nboots)
    lat = ds.createDimension('latg', nlat)
    lon = ds.createDimension('long', nlon)
    
    samples = ds.createVariable('sample', 'f8', ('sample'))
    lat = ds.createVariable('latg', 'f8', ('latg'))
    lon = ds.createVariable('long', 'f8', ('long'))
    
    statistics = ds.createVariable('statistic', 'f8', ('latg', 'long'))
    boot_statistics = ds.createVariable('boot_statistic', 'f8', ('sample', 'latg', 'long'))
    p_values = ds.createVariable('p_value', 'f8', ('latg', 'long'))
    
    samples[:] = np.arange(0, nboots, 1)
    lat[:] = latd[:]
    lon[:] = lond[:]
    
    statistics[:, :] = statistic[:, :]
    boot_statistics[:, :, :] = boot_statistic[:, :, :]
    p_values[:, :] = p_value[:, :]
    
    ds.close()

