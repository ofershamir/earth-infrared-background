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
    file_name = "ou-realization-2024-epsilon0-5.8-lambda0-0.06-tau0-2.3.nc"

    ds_in = xr.open_dataset(str(data_dir / file_name))
    # ds_in = ds_in.sel(time=ds_in.time.dt.year.isin(range(1981, 2011, 1)))

    # extract fields
    F = ds_in.F.values  # (samples,time,lat,lon)
    latd = ds_in.lat.values
    lond = ds_in.lon.values

    # dimensions
    ntim = F.shape[0]

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

    # the ou process was already generated on a gaussian grid

    # dimensions
    nlat = F.shape[1]
    nlon = F.shape[2]

    # remove dominant signals
    # F = spectral_analysis.remove_dominant_signals(F, ntim, spd, nDayTot, nDayWin, conserve_mean=True)

    Ftlm = spectral_analysis.space_only_analysis_no_window(F, ntim, ntrunc)

    # window
    F = spectral_analysis.windows(F, nSampWin, nSampSkip, nWindow, nlat, nlon)

    # space analysis
    Fwtlm = spectral_analysis.space_only_analysis(F, nSampWin, nWindow, ntrunc)

    # spece time coefficients
    Fwflm = spectral_analysis.space_time_analysis(F, nSampWin, nWindow, ntrunc)

    # keep track of frequency
    frequency = np.fft.fftfreq(nSampWin, 1./spd)
    # frequency = frequency[0:int(nSampWin/2)]

    # write to netcdf file
    file_name = "ou-realization-2024-space-time-analysis-window-360-skip-180-epsilon0-5.8-lambda0-0.06-tau0-2.3" + ".nc"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(data_dir / file_name), 'w', format='NETCDF4')

    window = ds.createDimension('window', nWindow)
    times = ds.createDimension('time', nSampWin)
    times2 = ds.createDimension('time2', ntim)
    frequencies = ds.createDimension('frequency', nSampWin)
    l = ds.createDimension('l', ntrunc+1)
    m = ds.createDimension('m', 2*ntrunc+1)

    windows = ds.createVariable('window', 'f8', ('window'))
    times = ds.createVariable('time', 'f8', ('time'))
    times2 = ds.createVariable('time2', 'f8', ('time2'))
    frequencies = ds.createVariable('frequency', 'f8', ('frequency'))
    l = ds.createVariable('l', 'f8', ('l'))
    m = ds.createVariable('m', 'f8', ('m'))

    Ftlms_real = ds.createVariable('Ftlm_real', 'f8', ('time2', 'l', 'm'))
    Ftlms_imag = ds.createVariable('Ftlm_imag', 'f8', ('time2', 'l', 'm'))
    Fwtlms_real = ds.createVariable('Fwtlm_real', 'f8', ('window', 'time', 'l', 'm'))
    Fwtlms_imag = ds.createVariable('Fwtlm_imag', 'f8', ('window', 'time', 'l', 'm'))
    Fwflms_real = ds.createVariable('Fwflm_real', 'f8', ('window', 'frequency', 'l', 'm'))
    Fwflms_imag = ds.createVariable('Fwflm_imag', 'f8', ('window', 'frequency', 'l', 'm'))

    windows[:] = np.arange(1, nWindow+1, 1)
    times[:] = np.arange(0, 360, 0.5)
    times2[:] = np.arange(0, 30*365+7, 0.5)
    frequencies[:] = frequency[:]
    l[:] = np.arange(0, ntrunc+1, 1)
    m[:] = np.arange(-ntrunc, ntrunc+1, 1)

    Ftlms_real[:, :, :] = np.real(Ftlm[:, :, :])
    Ftlms_imag[:, :, :] = np.imag(Ftlm[:, :, :])
    Fwtlms_real[:, :, :, :] = np.real(Fwtlm[:, :, :, :])
    Fwtlms_imag[:, :, :, :] = np.imag(Fwtlm[:, :, :, :])
    Fwflms_real[:, :, :, :] = np.real(Fwflm[:, :, :, :])
    Fwflms_imag[:, :, :, :] = np.imag(Fwflm[:, :, :, :])

    ds.close()

