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
    F = ds_in.olr.values  # (time,lat,lon)
    latd = ds_in.lat.values
    lond = ds_in.lon.values

    # parameters
    ntrunc = 72
    spd = 2

    # empty SHGrid instance for lon lat
    x = sh.SHGrid.from_zeros(lmax=ntrunc, grid='GLQ')

    # add cyclic longitude (originally 0. to 357.5 every 2.5  -- adding 360)
    F = np.dstack([F, F[:, :, 0]])
    lond = np.hstack([lond, 360.])

    # interpolate longitudes to the gaussian grid
    F_out = interp1d(lond, F, axis=2, fill_value='extrapolate')
    F = F_out(x.lons())

    # interpolate latitudes to the gaussian grid
    F_out = interp1d(latd, F, axis=1, fill_value='extrapolate')
    F = F_out(x.lats())

    # remove long-term temporal mean
    F -= np.mean(F, axis=0)

    # dimensions
    ntim = F.shape[0]
    nlat = F.shape[1]
    nlon = F.shape[2]

    # space analysis
    Ftlm = spectral_analysis.space_only_analysis_no_window(F, ntim, ntrunc)

    # spece time analysis
    Fflm = spectral_analysis.space_time_analysis_no_window(F, ntim, ntrunc)

    # keep track of frequency
    frequency = np.fft.fftfreq(ntim, 1./spd)
    # frequency = frequency[0:int(ntim/2)]

    # write to netcdf file
    file_name = "olr-2xdaily-1981-2010-space-time-analysis-raw" + ".nc"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(data_dir / file_name), 'w', format='NETCDF4')

    times = ds.createDimension('time', ntim)
    frequencies = ds.createDimension('frequency', ntim)
    l = ds.createDimension('l', ntrunc+1)
    m = ds.createDimension('m', 2*ntrunc+1)

    times = ds.createVariable('time', 'f8', ('time'))
    frequencies = ds.createVariable('frequency', 'f8', ('frequency'))
    l = ds.createVariable('l', 'f8', ('l'))
    m = ds.createVariable('m', 'f8', ('m'))

    Ftlms_real = ds.createVariable('Ftlm_real', 'f8', ('time', 'l', 'm'))
    Ftlms_imag = ds.createVariable('Ftlm_imag', 'f8', ('time', 'l', 'm'))
    Fflms_real = ds.createVariable('Fflm_real', 'f8', ('frequency', 'l', 'm'))
    Fflms_imag = ds.createVariable('Fflm_imag', 'f8', ('frequency', 'l', 'm'))

    times[:] = np.arange(0, int(ntim/2), 0.5)
    frequencies[:] = frequency[:]
    l[:] = np.arange(0, ntrunc+1, 1)
    m[:] = np.arange(-ntrunc, ntrunc+1, 1)

    Ftlms_real[:, :, :] = np.real(Ftlm[:, :, :])
    Ftlms_imag[:, :, :] = np.imag(Ftlm[:, :, :])
    Fflms_real[:, :, :] = np.real(Fflm[:, :, :])
    Fflms_imag[:, :, :] = np.imag(Fflm[:, :, :])

    ds.close()

