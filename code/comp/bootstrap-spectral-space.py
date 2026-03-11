import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import xarray as xr
import netCDF4 as nc
from pathlib import Path
import time
from multiprocessing import Pool


# grid
nboots = 5000
ntrunc=72
nSampWin = 720
nWindow = 59
spd = 2
frequency = np.fft.fftfreq(nSampWin, 1./spd)
ff = frequency[0:int(nSampWin/2)]
ll = np.arange(0, ntrunc+1, 1)
mm = np.arange(-ntrunc, ntrunc+1, 1)


def psd_l(Fwflm):

    # compute power
    Pwflm = np.abs(Fwflm[:, :, :, :])**2

    # to get psd need to scale (divide) by frequency spacing = spd/(nSampWin)
    Pwflm /= (spd / nSampWin)

    # some power is lost when using hanning window
    Pwflm *= 8. / 3.

    # average over m
    Pwfl_mean = np.zeros((nWindow, int(nSampWin/2), ntrunc+1))
    for l in range(0, ntrunc+1):
        Pwfl_mean[:, :, l] = np.mean(Pwflm[:, :, l, ntrunc-l:ntrunc+l+1], axis=2)

    # sum over m
    Pwfl_sum = np.zeros((nWindow, int(nSampWin/2), ntrunc+1))
    for l in range(0, ntrunc+1):
        Pwfl_sum[:, :, l] = np.sum(Pwflm[:, :, l, ntrunc-l:ntrunc+l+1], axis=2)

    return Pwfl_mean, Pwfl_sum


def psd_m(Fwflm):

    # compute power
    Pwflm = np.abs(Fwflm[:, :, :, :])**2

    # to get psd need to scale (divide) by frequency spacing = spd/(nSampWin)
    Pwflm /= (spd / nSampWin)

    # some power is lost when using hanning window
    Pwflm *= 8. / 3.

    # average over l
    Pwfm_mean = np.zeros((nWindow, int(nSampWin/2), 2*ntrunc+1))
    for m in range(-ntrunc, ntrunc+1):
        Pwfm_mean[:, :, ntrunc+m] = np.mean(Pwflm[:, :, np.abs(m):ntrunc+1, ntrunc+m], axis=2)

    # sum over l
    Pwfm_sum = np.zeros((nWindow, int(nSampWin/2), 2*ntrunc+1))
    for m in range(-ntrunc, ntrunc+1):
        Pwfm_sum[:, :, ntrunc+m] = np.sum(Pwflm[:, :, np.abs(m):ntrunc+1, ntrunc+m], axis=2)

    return Pwfm_mean, Pwfm_sum


def bootstrap_l(seed):

    rng = np.random.default_rng(seed)

    boot_statistic = np.zeros((int(nSampWin/2), ntrunc+1))

    for f in range(int(nSampWin/2)):
        for l in range(ntrunc+1):

            Fw_observed = Fwfl_observed_mean[:, f, l]
            Fw_ou = Fwfl_ou_mean[:, f, l]
            # Fw_observed = Fwfl_observed_sum[:, f, l]
            # Fw_ou = Fwfl_ou_sum[:, f, l]

            samp = rng.choice(np.concatenate((Fw_observed, Fw_ou)), size=2*nWindow, replace=True)

            Fw_1 = samp[0:nWindow]
            Fw_2 = samp[nWindow:2*nWindow]

            boot_statistic[f, l] = np.mean(Fw_1, axis=0) / np.mean(Fw_2, axis=0)

    return boot_statistic


def bootstrap_m(seed):

    rng = np.random.default_rng(seed)

    boot_statistic = np.zeros((int(nSampWin/2), 2*ntrunc+1))

    for f in range(int(nSampWin/2)):
        for m in range(2*ntrunc+1):

            Fw_observed = Fwfm_observed_mean[:, f, m]
            Fw_ou = Fwfm_ou_mean[:, f, m]
            # Fw_observed = Fwfm_observed_sum[:, f, m]
            # Fw_ou = Fwfm_ou_sum[:, f, m]

            samp = rng.choice(np.concatenate((Fw_observed, Fw_ou)), size=2*nWindow, replace=True)

            Fw_1 = samp[0:nWindow]
            Fw_2 = samp[nWindow:2*nWindow]

            boot_statistic[f, m] = np.mean(Fw_1, axis=0) / np.mean(Fw_2, axis=0)

    return boot_statistic


if __name__ == "__main__":

    base_dir = (Path(__file__).parent / "../../").resolve()
    data_dir = base_dir / "data"

    # obs    
    file_name = "olr-2xdaily-1981-2010-space-time-analysis-window-360-skip-180.nc"
    
    ds_observed = xr.open_dataset(str(data_dir / file_name))

    # extract
    Fwflm_observed = ds_observed.Fwflm_real.values[:, 0:int(nSampWin/2), :, :] +1j * ds_observed.Fwflm_imag.values[:, 0:int(nSampWin/2), :, :]

    # ou
    file_name = "ou-realization-2024-space-time-analysis-window-360-skip-180-epsilon0-5.8-lambda0-0.06-tau0-2.3.nc"
    
    ds_ou = xr.open_dataset(str(data_dir / file_name))

    # extract
    Fwflm_ou = ds_ou.Fwflm_real.values[:, 0:int(nSampWin/2), :, :] +1j * ds_ou.Fwflm_imag.values[:, 0:int(nSampWin/2), :, :]


    # statistic -- l
    Fwfl_observed_mean, _ = psd_l(Fwflm_observed)
    Fwfl_ou_mean, _ = psd_l(Fwflm_ou)
    
    statistic_l = np.zeros((int(nSampWin/2), ntrunc+1))
    
    for f in range(int(nSampWin/2)):
        for l in range(0, ntrunc+1):
    
            statistic_l[f, l] = np.mean(Fwfl_observed_mean, axis=0)[f, l] / np.mean(Fwfl_ou_mean, axis=0)[f, l]
            # statistic_l[f, l] = np.mean(Fwfl_observed_sum, axis=0)[f, l] / np.mean(Fwfl_ou_sum, axis=0)[f, l]

    # statistic -- m
    Fwfm_observed_mean, _ = psd_m(Fwflm_observed)
    Fwfm_ou_mean, _ = psd_m(Fwflm_ou)
    
    statistic_m = np.zeros((int(nSampWin/2), 2*ntrunc+1))
    
    for f in range(int(nSampWin/2)):
        for l in range(0, 2*ntrunc+1):
    
            statistic_m[f, l] = np.mean(Fwfm_observed_mean, axis=0)[f, l] / np.mean(Fwfm_ou_mean, axis=0)[f, l]
            # statistic_m[f, l] = np.mean(Fwfm_observed_sum, axis=0)[f, l] / np.mean(Fwfm_ou_sum, axis=0)[f, l]


    # call bootstrap in parallel over boots -- l
    start_time = time.time()
    
    seeds = np.array([2024 + i for i in range(nboots)])
    
    with Pool(10) as p:
        # boot_statistic_p = zip(*p.map(bootstrap, seeds))
        boot_statistic_p_l = p.map(bootstrap_l, seeds)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # unpack back to grid -- l
    boot_statistic_l = np.zeros((nboots, int(nSampWin/2), ntrunc+1))
    
    for b in range(nboots):
        boot_statistic_l[b, :, :] = boot_statistic_p_l[b]


    # call bootstrap in parallel over boots -- m
    start_time = time.time()
    
    seeds = np.array([2024 + i for i in range(nboots)])
    
    with Pool(10) as p:
        # boot_statistic_p = zip(*p.map(bootstrap, seeds))
        boot_statistic_p_m = p.map(bootstrap_m, seeds)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # unpack back to grid -- m
    boot_statistic_m = np.zeros((nboots, int(nSampWin/2), 2*ntrunc+1))
    
    for b in range(nboots):
        boot_statistic_m[b, :, :] = boot_statistic_p_m[b]


    # p-value -- l
    p_value_l = np.zeros((int(nSampWin/2), ntrunc+1))
    
    for f in range(int(nSampWin/2)):
        for l in range(ntrunc+1):
    
            if (statistic_l[f, l] > 1):
                p_value_l[f, l] = (len(boot_statistic_l[boot_statistic_l[:, f, l] > statistic_l[f, l], f, l]) / nboots +
                                 len(boot_statistic_l[boot_statistic_l[:, f, l] < 1/statistic_l[f, l], f, l]) / nboots)
            if (statistic_l[f, l] < 1):
                p_value_l[f, l] = (len(boot_statistic_l[boot_statistic_l[:, f, l] < statistic_l[f, l], f, l]) / nboots +
                                 len(boot_statistic_l[boot_statistic_l[:, f, l] > 1/statistic_l[f, l], f, l]) / nboots)


    # p-value -- m
    p_value_m = np.zeros((int(nSampWin/2), 2*ntrunc+1))
    
    for f in range(int(nSampWin/2)):
        for m in range(2*ntrunc+1):
    
            if (statistic_m[f, m] > 1):
                p_value_m[f, m] = (len(boot_statistic_m[boot_statistic_m[:, f, m] > statistic_m[f, m], f, m]) / nboots +
                                 len(boot_statistic_m[boot_statistic_m[:, f, m] < 1/statistic_m[f, m], f, m]) / nboots)
            if (statistic_m[f, m] < 1):
                p_value_m[f, m] = (len(boot_statistic_m[boot_statistic_m[:, f, m] < statistic_m[f, m], f, m]) / nboots +
                                 len(boot_statistic_m[boot_statistic_m[:, f, m] > 1/statistic_m[f, m], f, m]) / nboots)


    # write to netcdf file
    file_name = "boot-statistics-spectral-space-realization-2024-epsilon0-5.8-lambda0-0.06-tau0-2.3.nc"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(data_dir / file_name), 'w', format='NETCDF4')
    
    samples = ds.createDimension('sample', nboots)
    frequencies = ds.createDimension('frequency', int(nSampWin/2))
    l = ds.createDimension('l', ntrunc+1)
    m = ds.createDimension('m', 2*ntrunc+1)
    
    samples = ds.createVariable('sample', 'f8', ('sample'))
    frequencies = ds.createVariable('frequency', 'f8', ('frequency'))
    l = ds.createVariable('l', 'f8', ('l'))
    m = ds.createVariable('m', 'f8', ('m'))
    
    statistics_l = ds.createVariable('statistic_l', 'f8', ('frequency', 'l'))
    statistics_m = ds.createVariable('statistic_m', 'f8', ('frequency', 'm'))
    boot_statistics_l = ds.createVariable('boot_statistic_l', 'f8', ('sample', 'frequency', 'l'))
    boot_statistics_m = ds.createVariable('boot_statistic_m', 'f8', ('sample', 'frequency', 'm'))
    p_values_l = ds.createVariable('p_value_l', 'f8', ('frequency', 'l'))
    p_values_m = ds.createVariable('p_value_m', 'f8', ('frequency', 'm'))
    
    samples[:] = np.arange(0, nboots, 1)
    frequencies[:] = frequency[0:int(nSampWin/2)]
    l[:] = np.arange(0, ntrunc+1, 1)
    m[:] = np.arange(-ntrunc, ntrunc+1, 1)
    
    statistics_l[:, :] = statistic_l[:, :]
    statistics_m[:, :] = statistic_m[:, :]
    boot_statistics_l[:, :, :] = boot_statistic_l[:, :, :]
    boot_statistics_m[:, :, :] = boot_statistic_m[:, :, :]
    p_values_l[:, :] = p_value_l[:, :]
    p_values_m[:, :] = p_value_m[:, :]
    
    ds.close()

