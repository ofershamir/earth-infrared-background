import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import pyshtools as sh
import xarray as xr
import netCDF4 as nc
from pathlib import Path
import spectral_analysis


if __name__ == "__main__":

    # import data
    base_dir = (Path(__file__).parent / "../../").resolve()
    data_dir = base_dir / "data"
    file_name = "olr.2xdaily.1979-2022.nc"

    ds_in = xr.open_dataset(str(data_dir / file_name))
    ds_in = ds_in.sel(time=ds_in.time.dt.year.isin(range(1981, 2011, 1)))
    ds_in = ds_in.convert_calendar('noleap')

    # extract fields
    F = ds_in.olr  # (time,lat,lon)
    latd = ds_in.lat.values
    lond = ds_in.lon.values

    # remove long-term temporal mean
    F -= np.mean(F, axis=0)

    # remove annual cycle
    # F_annual_cycle = F.groupby("time.dayofyear").mean("time")
    # F_anom = F.groupby("time.dayofyear") - F_annual_cycle
    F_anom = F

    # dimensions
    ntim = F_anom.shape[0]

    # parameters
    ntrunc = 72
    spd = 2
    nDayWin = 360
    nDaySkip = -180
    nSampWin = nDayWin * spd

    # Number of days
    nDayTot = ntim // spd

    # Number of samples per temporal window
    nSampWin = nDayWin * spd

    # Number of samples to skip between window segments.
    # Negative number means overlap
    nSampSkip = nDaySkip * spd

    # Count the number of available samples
    nWindow = (ntim - nSampWin) // (nSampWin + nSampSkip) + 1

    # empty SHGrid instance for lon lat
    x = sh.SHGrid.from_zeros(lmax=ntrunc, grid='GLQ')

    # add cyclic longitude (originally 0. to 357.5 every 2.5  -- adding 360)
    F_anom = np.dstack([F_anom, F_anom[:, :, 0]])
    lond = np.hstack([lond, 360.])

    # interpolate longitudes to the gaussian grid
    F_out = interp1d(lond, F_anom, axis=2, fill_value='extrapolate')
    F_anom = F_out(x.lons())

    # interpolate latitudes to the gaussian grid
    F_out = interp1d(latd, F_anom, axis=1, fill_value='extrapolate')
    F_anom = F_out(x.lats())

    # dimensions
    nlat = F_anom.shape[1]
    nlon = F_anom.shape[2]

    # remove dominant signals
    F_anom = spectral_analysis.remove_dominant_signals(F_anom, ntim, spd, nDayTot, nDayWin, conserve_mean=False)

    # window
    F_anom = spectral_analysis.windows(F_anom, nSampWin, nSampSkip, nWindow, nlat, nlon)

    # space analysis
    Fwtlm = spectral_analysis.space_only_analysis(F_anom, nSampWin, nWindow, ntrunc)

    # spece time analysis
    Fwflm = spectral_analysis.space_time_analysis(F_anom, nSampWin, nWindow, ntrunc)

    # keep track of frequency
    frequency = np.fft.fftfreq(nSampWin, 1./spd)
    # frequency = frequency[0:int(nSampWin/2)]

    # write to netcdf file
    file_name = "olr-2xdaily-1981-2010-space-time-analysis-window-360-skip-180" + ".nc"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(data_dir / file_name), 'w', format='NETCDF4')

    window = ds.createDimension('window', nWindow)
    times = ds.createDimension('time', nSampWin)
    frequencies = ds.createDimension('frequency', nSampWin)
    l = ds.createDimension('l', ntrunc+1)
    m = ds.createDimension('m', 2*ntrunc+1)

    windows = ds.createVariable('window', 'f8', ('window'))
    times = ds.createVariable('time', 'f8', ('time'))
    frequencies = ds.createVariable('frequency', 'f8', ('frequency'))
    l = ds.createVariable('l', 'f8', ('l'))
    m = ds.createVariable('m', 'f8', ('m'))

    Fwtlms_real = ds.createVariable('Fwtlm_real', 'f8', ('window', 'time', 'l', 'm'))
    Fwtlms_imag = ds.createVariable('Fwtlm_imag', 'f8', ('window', 'time', 'l', 'm'))
    Fwflms_real = ds.createVariable('Fwflm_real', 'f8', ('window', 'frequency', 'l', 'm'))
    Fwflms_imag = ds.createVariable('Fwflm_imag', 'f8', ('window', 'frequency', 'l', 'm'))

    windows[:] = np.arange(1, nWindow+1, 1)
    times[:] = np.arange(0, 360, 0.5)
    frequencies[:] = frequency[:]
    l[:] = np.arange(0, ntrunc+1, 1)
    m[:] = np.arange(-ntrunc, ntrunc+1, 1)

    Fwtlms_real[:, :, :, :] = np.real(Fwtlm[:, :, :, :])
    Fwtlms_imag[:, :, :, :] = np.imag(Fwtlm[:, :, :, :])
    Fwflms_real[:, :, :, :] = np.real(Fwflm[:, :, :, :])
    Fwflms_imag[:, :, :, :] = np.imag(Fwflm[:, :, :, :])

    ds.close()

