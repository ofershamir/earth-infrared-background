import numpy as np
from scipy.interpolate import interp1d
import pyshtools as sh
import netCDF4 as nc
from pathlib import Path


# solve dX = 1/tau (theta - X) + sigma * sqrt(2/tau) dW
def solve_ou(tau, sigma, theta, dt, ntim, ntrunc, seed):
    # vectorized version, solving over all m>0 at once

    rng = np.random.default_rng(seed)

    F = np.zeros((ntim, ntrunc+1), dtype=np.complex128)

    # first sample
    F[0, :] = sigma * ( rng.standard_normal(size=ntrunc+1) + 1j * rng.standard_normal(size=ntrunc+1) ) / 2**0.5

    # following samples
    for n in range(1, ntim):
        F[n, :] = (theta + np.exp( - dt / tau ) * ( F[n-1, :] - theta ) +
                   ( sigma**2 * (1 - np.exp(- 2 * dt / tau) ) )**0.5 * ( rng.standard_normal(size=ntrunc+1) + 1j * rng.standard_normal(size=ntrunc+1) ) / 2**0.5)

    return F


if __name__ == "__main__":

    # parameters
    seed = 2024
    ntrunc = 72

    # time
    spd = 2
    ntim = (30 * 365 + 7 + 100) * spd  # 7 leap years + 100 day spinup
    dt = 1. / spd
    time = np.arange(0, ntim, 1, dtype=np.int32) * dt * 86400  # sec

    # empty SHGrid instance for lon lat
    x = sh.SHGrid.from_zeros(lmax=ntrunc, grid='GLQ')

    # grid
    nlon = x.lons().shape[0]
    nlat = x.lats().shape[0]
    lond = x.lons()
    latd = x.lats()
    lonr = np.deg2rad(lond)
    latr = np.deg2rad(latd)

    # field parameters
    epsilon_0 = 5.8
    lambda_0 = 0.060
    tau_0 = 2.3

    # spectral coefficients
    Flm = np.zeros((ntim, 2, ntrunc+1, ntrunc+1), dtype=np.complex128)

    # loop over spectral coefficients
    for l in range(ntrunc+1):
        tau_l = tau_0 / ( 1 + lambda_0**2 * l * (l+1) )
        sigma_l = epsilon_0 * tau_l / tau_0
        theta_l = 0

        # positive m
        Flm[:, 0, l, 0:l+1] = solve_ou(tau=tau_l, sigma=sigma_l, theta=theta_l, dt=dt, ntim=ntim, ntrunc=l, seed=seed)

    # negative m for real F
    for m in range(0, ntrunc+1):
        Flm[:, 1, :, m] = (-1)**m * np.conj(Flm[:, 0, :, m])

    # m == 0
    Flm[:, 0, :, 0] *= 2**0.5
    Flm[:, 1, :, 0] *= 2**0.5

    # grid space
    F = np.zeros((ntim, nlat, nlon), dtype=np.complex128)

    # loop over time
    for i in range(ntim):

        # creating SHCoeffs instance
        Clm = sh.SHCoeffs.from_zeros(normalization='ortho', csphase=-1, lmax=ntrunc, kind='complex')

        # populate
        for l in range(ntrunc+1):
            Clm.set_coeffs(ls=l, ms=np.arange(0, l+1), values=Flm[i, 0, l, 0:l+1])
            Clm.set_coeffs(ls=l, ms=np.arange(0, -l-1, -1), values=Flm[i, 1, l, 0:l+1])

        # sh transform
        C = Clm.expand(grid='GLQ', lmax_calc=ntrunc)

        # extract
        F[i, :, :] = C.to_array()


    # write to netcdf file
    base_dir = (Path(__file__).parent / "../../").resolve()
    data_dir = base_dir / "data"
    file_name = "ou-realization-" + str(seed) + "-epsilon0-5.8-lambda0-0.06-tau0-2.3.nc"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(str(data_dir / file_name), 'w', format='NETCDF4')

    times = ds.createDimension('time', (30 * 365 + 7) * spd)
    latitude = ds.createDimension('lat', nlat)
    lonngitude = ds.createDimension('lon', nlon)

    times = ds.createVariable('time', 'f8', ('time'))
    latitude = ds.createVariable('lat', 'f8', ('lat'))
    lonngitude = ds.createVariable('lon', 'f8', ('lon'))

    Fs = ds.createVariable('F', 'f8', ('time', 'lat', 'lon'))

    # save after spinup
    times[:] = time[100*spd:]
    latitude[:] = latd[:]
    lonngitude[:] = lond[:]

    Fs[:, :, :] = np.real(F[100*spd:, :, :]) # imaginary part is on the order of 1e-15

    ds.close()
