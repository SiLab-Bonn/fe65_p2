﻿
import logging
import numpy as np
import tables as tb
from scipy.optimize import curve_fit
from scipy.special import erf
import yaml


def cap_fac():
    return 7.9891


def analyze_threshold_scan(h5_file_name):
    with tb.open_file(h5_file_name, 'r+') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        hit_data = in_file_h5.root.hit_data[:]
        en_mask = in_file_h5.root.scan_results.en_mask[:]
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        scan_range = scan_args['scan_range']
        scan_range_inx = np.arange(scan_range[0], scan_range[1], scan_range[2])

        repeat_command = scan_args['repeat_command']

        np.set_printoptions(threshold=np.nan)
        param = np.unique(meta_data['scan_param_id'])
        ret = []
        for i in param:
            # this can be faster and multi threaded
            wh = np.where(hit_data['scan_param_id'] == i)
            hd = hit_data[wh[0]]
            hits = hd['col'].astype(np.uint16)
            hits = hits * 64
            hits = hits + hd['row']
            value = np.bincount(hits)
            value = np.pad(value, (0, 64 * 64 - value.shape[0]), 'constant')
            if len(ret):
                ret = np.vstack((ret, value))
            else:
                ret = value

        s_hist = np.swapaxes(ret, 0, 1)
        indices = np.indices(s_hist.shape)

        param_inx = np.ravel(indices[1].astype(np.float64))  # *0.05 - 0.6)

        pix_scan_hist = np.empty((s_hist.shape[1], repeat_command + 10))
        for param in range(s_hist.shape[1]):
            h_count = np.bincount(s_hist[:, param])
            h_count = h_count[:repeat_command + 10]
            pix_scan_hist[param] = np.pad(
                h_count, (0, (repeat_command + 10) - h_count.shape[0]), 'constant')

        log_hist = np.log10(pix_scan_hist)
        log_hist[~np.isfinite(log_hist)] = 0

        threshold = np.empty(64 * 64)
        noise = np.empty(64 * 64)
        chi2 = np.empty(64 * 64)
        x = scan_range_inx
        for pix in range(64 * 64):
            # this can multi threaded
            fitOut = fit_scurve(s_hist[pix], x, repeat_command)
            # starting at 1, 0 is scalling->should be fixed at repeat_command
            threshold[pix] = fitOut[1]
            noise[pix] = fitOut[2]
            chi2[pix] = fitOut[3]
        shape = en_mask.shape
        ges = 1
        for i in range(2):
            ges = ges * shape[i]
        Noise_pure = ()
        Threshold_pure = ()
        en_mask = en_mask.reshape(ges)
        Noise = noise.reshape(ges)
        Threshold = threshold.reshape(ges)
        ChisqS = chi2.reshape(ges)
        for n in range(ges):
            if (str(en_mask[n]) == 'True'):
                Noise_pure = np.append(Noise_pure, Noise[n])
                Threshold_pure = np.append(Threshold_pure, Threshold[n])

        # TODO: weird, check
        Threshold_pure[Threshold_pure > scan_range_inx[-1]] = 0

        hist_thresh_y, hist_thresh_x = np.histogram(
            Threshold_pure, density=False, bins=50)
        Noise_pure[Noise_pure > 0.02] = 0.02
        hist_noise_y, hist_noise_x = np.histogram(
            Noise_pure, density=False, bins=50)
        new_x = ()
        for entries in range(len(hist_thresh_x) - 1):
            new_x = np.append(
                new_x, (hist_thresh_x[entries] + hist_thresh_x[entries + 1]) / 2)
        hist_thresh_x = new_x
        new_x = ()
        for entries in range(len(hist_noise_x) - 1):
            new_x = np.append(
                new_x, (hist_noise_x[entries] + hist_noise_x[entries + 1]) / 2)
        hist_noise_x = new_x
        gauss_thresh = fit_gauss(hist_thresh_x[2:-4], hist_thresh_y[2:-4])
        gauss_noise = fit_gauss(hist_noise_x[2:-4], hist_noise_y[2:-4])
        thresh_fit_values = {}
        noise_fit_values = {}
        thresh_fit_values['height'] = gauss_thresh[0]
        thresh_fit_values['mu'] = gauss_thresh[1]
        thresh_fit_values['sigma'] = gauss_thresh[2]

        noise_fit_values['height'] = gauss_noise[0]
        noise_fit_values['mu'] = gauss_noise[1]
        noise_fit_values['sigma'] = gauss_noise[2]

        threshold = threshold.reshape(64, 64)
        noise = noise.reshape(64, 64)
        chi2 = chi2.reshape(64, 64)

        Thresh_results = in_file_h5.create_group(
            "/", 'Thresh_results', 'Thresh_results')
        Noise_results = in_file_h5.create_group(
            "/", 'Noise_results', 'Noise_results')
        Chisq_results = in_file_h5.create_group(
            "/", 'Chisq_results', 'ChiSq_results')
        Scurves_results = in_file_h5.create_group(
            "/", 'Scurves_results', 'Scurves_results')

        scurve_hist = in_file_h5.create_carray(Scurves_results, name='Scurve', title='Scurve Histogram',
                                               obj=s_hist.reshape((64, 64, scan_range_inx.shape[0])))
        # atom=tb.Atom.from_dtype(
        #    Scurves_results.dtype),
        # shape=Scurves_results.shape,

        #scurve_hist[:] = Scurves_results

        threshold_hist = in_file_h5.create_carray(Thresh_results, name='Threshold', title='Threshold Histogram',
                                                  atom=tb.Atom.from_dtype(
                                                      threshold.dtype),
                                                  shape=threshold.shape)
        threshold_hist[:] = threshold

        threshold_pure_hist = in_file_h5.create_carray(Thresh_results, name='Threshold_pure',
                                                       title='Threshold_pure Histogram',
                                                       atom=tb.Atom.from_dtype(
                                                           Threshold_pure.dtype),
                                                       shape=Threshold_pure.shape)
        threshold_pure_hist[:] = Threshold_pure
        threshold_pure_hist.attrs.fitdata_thresh = thresh_fit_values

        chisq_hist = in_file_h5.create_carray(where=Chisq_results, obj=chi2, name='Chisq_scurve',
                                              title='Chisq results per pix', atom=tb.Atom.from_dtype(chi2.dtype), shape=chi2.shape)
        chisq_hist_full = in_file_h5.create_carray(where=Chisq_results, obj=ChisqS, name='Chisq_scurve_unformatted',
                                                   title='chisq unformatted', atom=tb.Atom.from_dtype(ChisqS.dtype), shape=ChisqS.shape)

        noise_pure_hist = in_file_h5.create_carray(Noise_results, name='Noise_pure',
                                                   title='Noise_pure Histogram',
                                                   atom=tb.Atom.from_dtype(
                                                       Noise_pure.dtype),
                                                   shape=Noise_pure.shape)
        noise_pure_hist[:] = Noise_pure
        noise_hist = in_file_h5.create_carray(Noise_results, name='Noise', title='noise Histogram',
                                              atom=tb.Atom.from_dtype(
                                                  noise.dtype),
                                              shape=noise.shape)
        noise_hist[:] = noise
        noise_pure_hist.attrs.fitdata_noise = noise_fit_values


def scurve(x, A, mu, sigma):
    return 0.5 * A * erf((x - mu) / (np.sqrt(2) * sigma)) + 0.5 * A


# data of some pixels to fit, has to be global for the multiprocessing module
def fit_scurve(scurve_data, PlsrDAC, repeat_command):
    index = np.argmax(np.diff(scurve_data))
    maxInject = repeat_command
    q_min = min(PlsrDAC)
    q_max = max(PlsrDAC)
    M = np.sum(PlsrDAC)
    mu_guess = q_max - M / maxInject
    '''
    i = 0
    while i in len(PlsrDAC):
        if mu_und < mu_guess:
            mu_und += PlsrDAC[i]
        else:
            mu_ovr += maxInject - PlsrDAC
    sigma_guess = ((mu_ovr + mu_und) / maxInject) * np.sqrt(np.pi / 2)
    '''

    max_occ = np.median(scurve_data[index:])
    threshold = PlsrDAC[index]
    if abs(max_occ) <= 1e-08:  # or index == 0: occupancy is zero or close to zero
        popt = [0, 0, 0]
    else:
        try:
            popt, _ = curve_fit(scurve, PlsrDAC, scurve_data, p0=[
                                repeat_command, mu_guess, 0.01], check_finite=False)  # 0.01 vorher
            logging.info('Fit-params-scurve: %s %s %s ',
                         str(popt[0]), str(popt[1]), str(popt[2]))
        except RuntimeError:  # fit failed
            popt = [0, 0, 0]
            logging.info('Fit did not work scurve: %s %s %s', str(popt[0]),
                         str(popt[1]), str(popt[2]))

    chi2 = np.sum(np.diff(PlsrDAC - scurve(scurve_data, *popt))**2)

    if popt[1] < 0:  # threshold < 0 rarely happens if fit does not work
        popt = [0, 0, 0]
    return popt[0], popt[1], popt[2], chi2


def gauss(x_data, *parameters):
    """Gauss function"""
    A_gauss, mu_gauss, sigma_gauss = parameters
    return A_gauss * np.exp(-(x_data - mu_gauss)**2 / (2. * sigma_gauss**2))


def fit_gauss(x_data, y_data):
    """Fit gauss"""
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    y_maxima = x_data[np.where(y_data[:] == np.max(y_data))[0]]
    params_guess = np.array(
        [np.max(y_data), y_maxima[0], np.std(x_data)])  # np.mean(y_data)
    logging.info('Params guessed: %s ', str(params_guess))
    try:
        params_from_fit = curve_fit(gauss, x_data, y_data, p0=params_guess)
        logging.info('Fit-params-gauss: %s %s %s ', str(params_from_fit[0][0]), str(
            params_from_fit[0][1]), str(params_from_fit[0][2]))
    except RuntimeError:
        logging.info('Fit did not work gauss: %s %s %s', str(np.max(y_data)), str(
            x_data[np.where(y_data[:] == np.max(y_data))[0]][0]), str(np.std(x_data)))
        return params_guess[0], params_guess[1], params_guess[2]
    A_fit = params_from_fit[0][0]
    mu_fit = params_from_fit[0][1]
    sigma_fit = np.abs(params_from_fit[0][2])
    return A_fit, mu_fit, sigma_fit


if __name__ == "__main__":
    pass


def exp(x, *parameters):
    a, b, c, d = parameters
    return d + c * np.exp(-(x + b) / a)


def fit_exp(x_data, y_data, thresh, decline):
    a = (x_data[decline] - x_data[0]) / 4
    b = -1 * thresh
    c = np.max(y_data)
    d = np.min(y_data)
    params_guess = np.array([a, b, c, d])
    if len(x_data) < 4:
        return 0, 0, 0, 0
    try:
        params_from_fit = curve_fit(exp, x_data, y_data, p0=params_guess)
        logging.info('Fit worked exp: %s %s %s %s', str(params_from_fit[0][0]),
                     str(params_from_fit[0][1]), str(params_from_fit[0][2]), str(params_from_fit[0][3]))
    except RuntimeError:
        logging.info('Fit did not work exp: %s %s %s %s', str(a),
                     str(b), str(c), str(d))
        return a, b, c, d
    a_fit = params_from_fit[0][0]
    b_fit = params_from_fit[0][1]
    c_fit = params_from_fit[0][2]
    d_fit = params_from_fit[0][3]
    return a_fit, b_fit, c_fit, d_fit


def cosh(x, *parameters):
    a, b, c, d = parameters
    return d + c / (np.cosh((x + b) / a))


def fit_cosh(x_data, y_data, thresh, decline):
    a = (x_data[decline] - x_data[0]) / 4
    b = -1 * thresh
    c = np.max(y_data)
    d = np.min(y_data)
    params_guess = np.array([a, b, c, d])
    # print "params_guessed: ", params_guess
    try:
        params_from_fit = curve_fit(cosh, x_data, y_data, p0=params_guess)
    except RuntimeError:
        logging.info('Fit did not work: %s %s %s %s', str(a),
                     str(b), str(c), str(d))
        return a, b, c, d
    a_fit = params_from_fit[0][0]
    b_fit = params_from_fit[0][1]
    c_fit = params_from_fit[0][2]
    d_fit = params_from_fit[0][3]
    return a_fit, b_fit, c_fit, d_fit
