
import logging
import numpy as np
import tables as tb
from scipy.optimize import curve_fit
from scipy.special import erf

def scurve(x, A, mu, sigma):
    return 0.5 * A * erf((x - mu) / (np.sqrt(2) * sigma)) + 0.5 * A

def fit_scurve(scurve_data, PlsrDAC):  # data of some pixels to fit, has to be global for the multiprocessing module
    index = np.argmax(np.diff(scurve_data))
    max_occ = np.median(scurve_data[index:])
    threshold = PlsrDAC[index] 
    if abs(max_occ) <= 1e-08:  # or index == 0: occupancy is zero or close to zero
        popt = [0, 0, 0]
    else:
        try:
            popt, _ = curve_fit(scurve, PlsrDAC, scurve_data, p0=[max_occ, threshold, 0.01], check_finite=False)
        except RuntimeError:  # fit failed
            popt = [0, 0, 0]
    if popt[1] < 0:  # threshold < 0 rarely happens if fit does not work
        popt = [0, 0, 0]
    return popt[1:3]


if __name__ == "__main__":
    pass
