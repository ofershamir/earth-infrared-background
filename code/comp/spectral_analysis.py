import numpy as np
from scipy import signal
import pyshtools as sh

# based on code by Alejandro Jaramillo

def remove_dominant_signals(F_in, ntim, spd, nDayTot, nDayWin, conserve_mean=False):
    """
    This function removes the dominant signals by removing the long term
    linear trend and mean (by default) and eliminating the annual cycle
    by removing all time periods less than a corresponding critical
    frequency.
    """

    # F = F(time, lat, lon)

    # Critical frequency
    fCrit   = 1. / nDayWin

    # Remove long term linear trend
    long_mean = np.mean(F_in, axis=0)
    detrend = signal.detrend(F_in, axis=0, type='linear')

    if conserve_mean == True:
        # F_out = detrend + long_mean
        F_out = F_in
    elif conserve_mean == False:
        # F_out = detrend
        F_out = F_in - long_mean

    fourier = np.fft.rfft(F_out, axis=0)
    fourier_mean = np.copy(fourier[0, :, :])
    freq = np.fft.rfftfreq(ntim, 1./spd)

    # removing also the dominant harmonics of the annual cycle
    ind1 = np.argmin(np.abs(freq - 1./365))
    ind2 = np.argmin(np.abs(freq - 2./365))
    ind3 = np.argmin(np.abs(freq - 3./365))
    ind4 = np.argmin(np.abs(freq - 4./365))
    fourier[ind1, :, :] = 0.0
    fourier[ind2, :, :] = 0.0
    fourier[ind3, :, :] = 0.0
    fourier[ind4, :, :] = 0.0

    fourier[0, :, :] = fourier_mean
    F_out = np.fft.irfft(fourier, axis=0)

    return F_out


def smooth121(F_in):
    """
    Smoothing function that takes a 1D array and pass it through a 1-2-1 filter.
    This function is a modified version of the wk_smooth121 from NCL.
    The weights for the first and last points are  3-1 (1st) or 1-3 (last) conserving the total sum.
    :param F_in:
        Input array
    :type F_in: Numpy array
    :return: F_out
    :rtype: Numpy array
    """

    F_out = np.zeros(len(F_in))
    weights = np.array([1.0, 2.0, 1.0]) / 4.0
    sma = np.convolve(F_in, weights, 'valid')
    F_out[1:-1] = sma

    # Now its time to correct the borders
    if (np.isnan(F_in[1])):
        if (np.isnan(F_in[0])):
            F_out[0] = np.nan
        else:
            F_out[0] = F_in[0]
    else:
        if (np.isnan(F_in[0])):
            F_out[0] = np.nan
        else:
            F_out[0] = (F_in[1] + 3.0 * F_in[0]) / 4.0
    if (np.isnan(F_in[-2])):
        if (np.isnan(F_in[-1])):
            F_out[-1] = np.nan
        else:
            F_out[-2] = F_out[-2]
    else:
        if (np.isnan(F_in[-1])):
            F_out[-1] = np.nan
        else:
            F_out[-1] = (F_in[-2] + 3.0 * F_in[-1]) / 4.0

    return F_out


def windows(F, nSampWin, nSampSkip, nWindow, nlat, nlon):

    # F = F(time, lat, lon)

    # allocate
    F_windows = np.zeros((nWindow, nSampWin, nlat, nlon), dtype=np.complex128)

    # Create a tapering window(nSampWin, nlat, nlon) using the hanning function.
    # For more information about the hanning function see:
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.hanning.html
    # Note: Hanning function is different from Hamming function.
    tapering_window = np.repeat(np.hanning(nSampWin), nlat*nlon).reshape(nSampWin, nlat, nlon)

    ntStrt = 0
    ntLast = nSampWin

    for nw in range(nWindow):

        # Detrend temporal window
        # temp_window = signal.detrend(F[ntStrt:ntLast, :, :], axis=0, type='linear')
        temp_window = F[ntStrt:ntLast, :, :]

        # Taper temporal window in the time dimension
        temp_window = temp_window * tapering_window

        # extract
        F_windows[nw, :, :, :] = temp_window

        # Set index for next temporal window
        ntStrt = ntLast + nSampSkip
        ntLast = ntStrt + nSampWin

        del temp_window

    return F_windows


def spatial_analysis(F, ntrunc):

    # F = F(lat, lon), where latitudes are on a gaussian grid

    # allocate
    Flm = np.zeros((ntrunc+1, 2*ntrunc+1), dtype=np.complex128)

    # creating SHGrid instance
    C = sh.SHGrid.from_array(F[:, :], grid='GLQ')

    # sh transform
    Clm = C.expand(normalization='ortho', csphase=-1, lmax_calc=ntrunc)
    Clm = Clm.convert(kind='complex')

    # extract
    Flm[:, ntrunc:2*ntrunc+1] = Clm.to_array()[0, :, :]

    # negative m for real F
    for m in range(1, ntrunc+1):
        Flm[:, ntrunc-m] = (-1)**m * np.conj(Flm[:, ntrunc+m])

    return Flm


def space_time_analysis(F, nSampWin, nWindow, ntrunc):

    # F = F(window, time, lat, lon), where latitudes are on a gaussian grid

    # allocate
    Fwflm = np.zeros((nWindow, nSampWin, ntrunc+1, 2*ntrunc+1), dtype=np.complex128)

    # loop over windows
    for w in range(nWindow):
        # loop over time to compute Flm
        for i in range(nSampWin):
            Fwflm[w, i, :, :] = spatial_analysis(F[w, i, :, :], ntrunc)

    # fourier in time
    Fwflm = np.fft.fft(Fwflm[:, :, :, :], axis=1)

    # normalize by # time samples
    Fwflm /= nSampWin

    return Fwflm


def space_time_analysis_no_window(F, ntim, ntrunc):

    # F = F(time, lat, lon), where latitudes are on a gaussian grid

    # allocate
    Fflm = np.zeros((ntim, ntrunc+1, 2*ntrunc+1), dtype=np.complex128)

    # loop over time to compute Flm
    for i in range(ntim):
        Fflm[i, :, :] = spatial_analysis(F[i, :, :], ntrunc)

    # fourier in time
    Fflm = np.fft.fft(Fflm[:, :, :], axis=0)

    # normalize by # time samples
    Fflm /= ntim

    return Fflm


def space_only_analysis(F, nSampWin, nWindow, ntrunc):

    # F = F(window, time, lat, lon), where latitudes are on a gaussian grid

    # allocate
    Fwtlm = np.zeros((nWindow, nSampWin, ntrunc+1, 2*ntrunc+1), dtype=np.complex128)

    # loop over windows
    for w in range(nWindow):
        # loop over time to compute Flm
        for i in range(nSampWin):
            Fwtlm[w, i, :, :] = spatial_analysis(F[w, i, :, :], ntrunc)

    return Fwtlm


def space_only_analysis_no_window(F, ntim, ntrunc):

    # F = F(time, lat, lon), where latitudes are on a gaussian grid

    # allocate
    Ftlm = np.zeros((ntim, ntrunc+1, 2*ntrunc+1), dtype=np.complex128)

    # loop over time to compute Flm
    for i in range(ntim):
        Ftlm[i, :, :] = spatial_analysis(F[i, :, :], ntrunc)

    return Ftlm

