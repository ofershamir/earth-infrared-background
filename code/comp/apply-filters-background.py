import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import xarray as xr
import netCDF4 as nc
from pathlib import Path
import spectral_analysis


def filter_omega_k(data, mask):

    ntim, nlat, nlon = data.shape

    # 2d fft across time (axis 0) and longitude (axis 2), for all 73 latitudes at once
    fft = np.fft.fft2(data, axes=(0, 2))

    # shift
    fft = np.fft.fftshift(fft, axes=(0, 2))

    # fourier e^[-i(kx+wt)] modes to waves e^[i(kx-wt)]
    fft = fft[:, :, ::-1]

    # mask is (720, 145), move to (720, 1, 145)
    mask_3d = mask[:, np.newaxis, :]

    # apply the mask
    fft_filtered = np.where(mask_3d.astype(bool), fft, 0.0)

    # waves e^[i(kx-wt)] to fourier e^[-i(kx+wt)] mode
    fft_filtered = fft_filtered[:, :, ::-1]

    # shift back
    fft_filtered = np.fft.ifftshift(fft_filtered, axes=(0, 2))

    # 2d ifft back to time-longitude space for all latitudes
    filtered_data = np.real(np.fft.ifft2(fft_filtered, axes=(0, 2)))

    return filtered_data


if __name__ == "__main__":

    # import data
    base_dir = (Path(__file__).parent / "../../").resolve()
    data_dir = base_dir / "data"
    file_name = "ou-realization-2024-epsilon0-5.8-lambda0-0.06-tau0-2.3.nc"

    ds_in = xr.open_dataset(str(data_dir / file_name))
    # ds_in = ds_in.sel(time=ds_in.time.dt.year.isin(range(1981, 2011, 1)))

    # extract fields
    F = ds_in.F.values  # (time,lat,lon)
    latg = ds_in.lat.values
    long = ds_in.lon.values

    # regular grid
    latd = np.linspace(90, -90, 73)
    lond = np.linspace(0, 360, 145)

    # interpolate longitudes to regular grid
    F_out = interp1d(long, F, axis=2, fill_value='extrapolate')
    F = F_out(lond)

    # interpolate latitudes to regular grid
    F_out = interp1d(latg, F, axis=1, fill_value='extrapolate')
    F = F_out(latd)

    # dimensions
    ntim, nlat, nlon = F.shape

    # remove long-term temporal mean
    # F -= np.mean(F, axis=0)

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

    # remove dominant signals
    F = spectral_analysis.remove_dominant_signals(F, ntim, spd, nDayTot, nDayWin, conserve_mean=False)

    # window
    Fw = spectral_analysis.windows(F, nSampWin, nSampSkip, nWindow, nlat, nlon)

    # correcting for the lost power
    # Fw *= (8. / 3.)**0.5

    # spectral grid
    frequency = np.fft.fftfreq(nSampWin, 1./spd)
    frequency = np.fft.fftshift(frequency)


    ## masks
    
    # kelvin wheeler and kiladis
    mask1 = np.zeros((nSampWin, 2*ntrunc+1))

    for i in range(nSampWin):
        for k in range(1, 7):
            if ( frequency[i] >= 0.05 + (8*9.8)**0.5 * (k-2) / 4e7 * 86400 ) and ( frequency[i] <= 0.05 + (90*9.8)**0.5 * (k-1) / 4e7 * 86400 ):
                mask1[i, ntrunc+k] = 1
        for k in range(7, 15):
            if ( frequency[i] >= 0.05 + (8*9.8)**0.5 * (k-2) / 4e7 * 86400 ) and ( frequency[i] <= 0.05 + (90*9.8)**0.5 * (6-1) / 4e7 * 86400 ):
                mask1[i, ntrunc+k] = 1


    # eastward baroclinic rossby
    mask2 = np.zeros((nSampWin, 2*ntrunc+1))

    for i in range(nSampWin):
        for k in range(1, 3):
            if (frequency[i] >= 0.05) and (frequency[i] <= 0.05 + ( (150*9.8)**0.5 * (k-1) ) * 86400 / 4e7):
                mask2[i, ntrunc+k] = 1
        for k in range(3, 10):
            if ( frequency[i] >= 0.05 + ( (8*9.8)**0.5 * (k-2) ) * 86400 / 4e7 ) and ( frequency[i] <= 0.05 + ( (150*9.8)**0.5 * (k-1) ) * 86400 / 4e7 ):
                mask2[i, ntrunc+k] = 1
        for k in range(10, 21):
            y1 = 0.05 + ( (150*9.8)**0.5 * (9-1) ) * 86400 / 4e7
            y2 = 0.05 + ( (8*9.8)**0.5 * (20-2) ) * 86400 / 4e7
            m1 = (y1 - y2) / (9 - 20)
            if ( frequency[i] >= 0.05 + ( (8*9.8)**0.5 * (k-2) ) * 86400 / 4e7 ) and ( frequency[i] <= y1 + m1 * (k-8) ):
                mask2[i, ntrunc+k] = 1

    
    # westeard baroclinic rossby
    mask3 = np.zeros((nSampWin, 2*ntrunc+1))

    for i in range(nSampWin):
        for k in range(-2, 0):
            if (frequency[i] >= 0.05) and (frequency[i] <= 0.05 - ( (150*9.8)**0.5 * (k+1) ) * 86400 / 4e7):
                mask3[i, ntrunc+k] = 1
        for k in range(-9, -2):
            if ( frequency[i] >= 0.05 - ( (8*9.8)**0.5 * (k+2) ) * 86400 / 4e7 ) and ( frequency[i] <= 0.05 - ( (150*9.8)**0.5 * (k+1) ) * 86400 / 4e7 ):
                mask3[i, ntrunc+k] = 1
        for k in range(-20, -9):
            y1 = 0.05 - ( (150*9.8)**0.5 * (-9+1) ) * 86400 / 4e7
            y2 = 0.05 - ( (8*9.8)**0.5 * (-20+2) ) * 86400 / 4e7
            m1 = (y1 - y2) / (9 - 20)
            if ( frequency[i] >= 0.05 - ( (8*9.8)**0.5 * (k+2) ) * 86400 / 4e7 ) and ( frequency[i] <= y1 - m1 * (k+9) ):
                mask3[i, ntrunc+k] = 1


    # center lobes
    mask4 = np.zeros((nSampWin, 2*ntrunc+1))

    for i in range(nSampWin):
        for k in range(1, 5):
            if ( frequency[i] >= 0.05 + ( (1400*9.8)**0.5 * (k-1) ) * 86400 / 4e7 ) and ( frequency[i] <= 0.05 + ( (1400*9.8)**0.5 * (4-1) ) * 86400 / 4e7 ):
                mask4[i, ntrunc+k] = 1
        for k in range(-4, 0):
            if ( frequency[i] >= 0.05 - ( (1400*9.8)**0.5 * (k+1) ) * 86400 / 4e7 ) and ( frequency[i] <= 0.05 + ( (1400*9.8)**0.5 * (4-1) ) * 86400 / 4e7 ):
                mask4[i, ntrunc+k] = 1
        for k in [0]:
            if ( frequency[i] >= 0.05 ) and ( frequency[i] <= 0.05 + ( (1400*9.8)**0.5 * (4-1) ) * 86400 / 4e7 ):
                mask4[i, ntrunc+k] = 1


    # eastward side lobe
    mask5 = np.zeros((nSampWin, 2*ntrunc+1))

    for i in range(nSampWin):
        for k in range(10, 56):
            if ( frequency[i] >= 0.05 ) and ( frequency[i] <= 0.05 + ( (5*9.8)**0.5 * (k-10) ) * 86400 / 4e7 ):
                mask5[i, ntrunc+k] = 1


    # westward side lobe
    mask6 = np.zeros((nSampWin, 2*ntrunc+1))

    for i in range(nSampWin):
        for k in range(-55, -9):
            if ( frequency[i] >= 0.05 ) and ( frequency[i] <= 0.05 - ( (5*9.8)**0.5 * (k+10) ) * 86400 / 4e7 ):
                mask6[i, ntrunc+k] = 1


    # mjo 
    mask7 = np.zeros((nSampWin, 2*ntrunc+1))

    for i in range(nSampWin):
        for k in range(1, 6):
            if ( frequency[i] >= 1./96. ) and ( frequency[i] <= 1./30. ):
                mask7[i, ntrunc+k] = 1


    # rossby equatorial
    mask8 = np.zeros((nSampWin, 2*ntrunc+1))

    for i in range(nSampWin):
        for k in range(-10, 0):
            if ( frequency[i] >= - k * 86400 / (2*np.pi) / 4e7 * 2.3e-11 / ( (k/4e7)**2 + 2.3e-11 / (8*9.8)**0.5 ) ) and ( frequency[i] <=  - k * 86400 / (2*np.pi) / 4e7 * 2.3e-11 / ( (k/4e7)**2 + 2.3e-11 / (90*9.8)**0.5 )  ):
                mask8[i, ntrunc+k] = 1


    # apply filters
    F1 = np.zeros_like(Fw)
    for n in range(nWindow):
        F1[n, :, :, :] = filter_omega_k(Fw[n, :, :, :], mask1)

    F2 = np.zeros_like(Fw)
    for n in range(nWindow):
        F2[n, :, :, :] = filter_omega_k(Fw[n, :, :, :], mask2)

    F3 = np.zeros_like(Fw)
    for n in range(nWindow):
        F3[n, :, :, :] = filter_omega_k(Fw[n, :, :, :], mask3)

    F4 = np.zeros_like(Fw)
    for n in range(nWindow):
        F4[n, :, :, :] = filter_omega_k(Fw[n, :, :, :], mask4)

    F5 = np.zeros_like(Fw)
    for n in range(nWindow):
        F5[n, :, :, :] = filter_omega_k(Fw[n, :, :, :], mask5)

    F6 = np.zeros_like(Fw)
    for n in range(nWindow):
        F6[n, :, :, :] = filter_omega_k(Fw[n, :, :, :], mask6)

    F7 = np.zeros_like(Fw)
    for n in range(nWindow):
        F7[n, :, :, :] = filter_omega_k(Fw[n, :, :, :], mask7)

    F8 = np.zeros_like(Fw)
    for n in range(nWindow):
        F8[n, :, :, :] = filter_omega_k(Fw[n, :, :, :], mask8)


    # write to netcdf file
    file_name = "ou-realization-2024-epsilon0-5.8-lambda0-0.06-tau0-2.3-filters" + ".nc"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(data_dir / file_name), 'w', format='NETCDF4')

    window = ds.createDimension('window', nWindow)
    times = ds.createDimension('time', nSampWin)
    lat = ds.createDimension('lat', nlat)
    lon = ds.createDimension('lon', nlon)

    windows = ds.createVariable('window', 'f8', ('window'))
    times = ds.createVariable('time', 'f8', ('time'))
    lats = ds.createVariable('lat', 'f8', ('lat'))
    lons = ds.createVariable('lon', 'f8', ('lon'))

    Fw1 = ds.createVariable('Fw1', 'f8', ('window', 'time', 'lat', 'lon'))
    Fw2 = ds.createVariable('Fw2', 'f8', ('window', 'time', 'lat', 'lon'))
    Fw3 = ds.createVariable('Fw3', 'f8', ('window', 'time', 'lat', 'lon'))
    Fw4 = ds.createVariable('Fw4', 'f8', ('window', 'time', 'lat', 'lon'))
    Fw5 = ds.createVariable('Fw5', 'f8', ('window', 'time', 'lat', 'lon'))
    Fw6 = ds.createVariable('Fw6', 'f8', ('window', 'time', 'lat', 'lon'))
    Fw7 = ds.createVariable('Fw7', 'f8', ('window', 'time', 'lat', 'lon'))
    Fw8 = ds.createVariable('Fw8', 'f8', ('window', 'time', 'lat', 'lon'))

    windows[:] = np.arange(1, nWindow+1, 1)
    times[:] = np.arange(0, 360, 0.5)
    lats[:] = latd
    lons[:] = lond

    Fw1[:, :, :, :] = F1[:, :, :, :]
    Fw2[:, :, :, :] = F2[:, :, :, :]
    Fw3[:, :, :, :] = F3[:, :, :, :]
    Fw4[:, :, :, :] = F4[:, :, :, :]
    Fw5[:, :, :, :] = F5[:, :, :, :]
    Fw6[:, :, :, :] = F6[:, :, :, :]
    Fw7[:, :, :, :] = F7[:, :, :, :]
    Fw8[:, :, :, :] = F8[:, :, :, :]

    ds.close()

