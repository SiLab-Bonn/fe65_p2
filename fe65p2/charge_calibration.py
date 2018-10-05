#===============================================================================
#
# File to do the charge calibration analysis, included are the following:
#     single hit selection (in analysis)
#     fit of tdc peaks after single hit selection for each source type
#         need common file name system
#     background fit and reduction
#     creation of splines for the conversion from Vinj to TDC channel
#     plotting of keVs from peak vs the Vinj corresponding to tdc channels
#     fit of linear function to the charge calibration
#     calculation of true threshold and true injection capacitance
#
# Created by Daniel Coquelin on 15.6.2018
#===============================================================================

import fe65p2
import analysis as analysis
import yaml
import numpy as np
import tables as tb
import glob
from os import chdir
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy import stats as ss
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate.fitpack2 import UnivariateSpline
import uncertainties as unc


src_list = ["Am241", "Cd109", "Tb56"]
# src peaks all in keV
src_peaks = {"Am241": [21.16, 26.345, 59.541],  # L_gamma, Gamma_2,1(Np), Gamma_2,0(Np)
             "Cd109": [8.64, 22.103, 25.05],     # avg K_alpha of copper, K_alpha, K_beta
             "Sn": [8.64, 25.157, 28.680],       # avg K_alpha of copper, K_alpha, K_beta
             "Mo": [8.64, 17.427, 19.852],       # avg K_alpha of copper, K_alpha, K_beta
             "Nb": [8.64, 16.583, 18.697]}       # avg K_alpha of copper, K_alpha, K_beta
#"Cu": [8.0378, 8.941]


pix_list = [[11, 6], [10, 12], [11, 20], [9, 37], [44, 5], [44, 12], [41, 19], [42, 42],
            [5, 25], [10, 25], [19, 25], [29, 25], [33, 25], [46, 24], [49, 25], [60, 25],
            [2, 7], [10, 7], [20, 7], [29, 4], [35, 5], [42, 6], [54, 6], [58, 6],
            [3, 49], [12, 50], [19, 54], [24, 55], [37, 51], [45, 46], [49, 49], [58, 44]]


def df_basics(pix):
    # get the sensor flavor, fe flavor, threshond and noise here, returned as a list of strings
    #------------------------------------------------------------------------------
    # pixel flavor
    if pix[0] < 8 * 4:
        if pix[1] < 8 + 1:
            pix_flav = 'nw15'
        elif pix[1] < 1 + 8 * 2:
            pix_flav = 'nw20'
        elif pix[1] < 1 + 8 * 3:
            pix_flav = 'nw25'
        else:
            pix_flav = 'nw30'

    else:
        if pix[1] < 8 + 1:
            pix_flav = 'dnw15'
        elif pix[1] < 1 + 8 * 2:
            pix_flav = 'dnw20'
        elif pix[1] < 1 + 8 * 3:
            pix_flav = 'dnw25'
        else:
            pix_flav = 'dnw30'

    #------------------------------------------------------------------------------
    # fe flavor
    if pix[0] < 8:
        fe_flav = 1
    elif pix[0] < 8 * 2:
        fe_flav = 2
    elif pix[0] < 8 * 3:
        fe_flav = 3
    elif pix[0] < 8 * 4:
        fe_flav = 4
    elif pix[0] < 8 * 5:
        fe_flav = 5
    elif pix[0] < 8 * 6:
        fe_flav = 6
    elif pix[0] < 8 * 7:
        fe_flav = 7
    else:
        fe_flav = 8
    #------------------------------------------------------------------------------
    # threshold and noise
    thresh_h5 = '/home/daniel/Documents/InterestingPlots/chip6/20180706_133647_threshold_scan (copy).h5'
    with tb.open_file(thresh_h5, 'r+') as in_h5:
        thresholds = in_h5.root.Thresh_results.Threshold[:]
        thresholds_err = in_h5.root.Thresh_errs.Threshold[:]
        Noise = in_h5.root.Noise_results.Noise[:]
        Noise_err = in_h5.root.Noise_errs.Noise_Errs[:]

        thresh = thresholds[pix[0], pix[1]]
        thresh_err = thresholds_err[pix[0], pix[1]]
        noise = Noise[pix[0], pix[1]]
        noise_err = Noise_err[pix[0], pix[1]]
    return pix_flav, 'fe' + str(fe_flav), thresh, thresh_err, noise, noise_err


def get_singles(src, flav=None):
    # get single hits from
    chdir("/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/" + str(src))
    h5_files = glob.glob('*.h5')
#     print src, h5_files

    for i, h5_name in enumerate(h5_files):
        with tb.open_file(h5_name, 'r+') as io_file_h5:
            #             print h5_name
            try:
                hit_data = io_file_h5.root.hit_data[:]
            except:
                fe65 = fe65p2.fe65p2()
                hit_data = fe65.interpret_raw_data_w_tdc(io_file_h5.root.raw_data[:], io_file_h5.root.meta_data[:])

        if flav:
            if i == 0:
                singles_stacked = analysis.singular_hits_tdc_pix_flav(hit_data=hit_data, flav=flav)
            else:
                singles_stacked = np.append(singles_stacked, analysis.singular_hits_tdc_pix_flav(hit_data=hit_data, flav=flav))
        else:
            if i == 0:
                singles_stacked = analysis.singular_hits_tdc_pix_flav(hit_data=hit_data)
            else:
                singles_stacked = np.append(singles_stacked, analysis.singular_hits_tdc_pix_flav(hit_data=hit_data))

    return singles_stacked


def bkg_func(x, *params):
    # linear to a limit, then at limit go to
    p1, m, b, p2, p3, p4, p5, peak1a, peak1u, peak1s, peak2a, peak2u, peak2s, peak3a, peak3u, peak3s, peak4a, peak4u, peak4s, peak5a, peak5u, peak5s = params

    if x <= p1:
        return analysis.linear(x, m, b)
    else:
        return (analysis.exp(x, p2, p3, p4, p5) +
                analysis.gauss(x, peak1a, peak1u, peak1s) +
                analysis.gauss(x, peak2a, peak2u, peak2s) +
                analysis.gauss(x, peak3a, peak3u, peak3s) +
                analysis.gauss(x, peak4a, peak4u, peak4s) +
                analysis.gauss(x, peak5a, peak5u, peak5s))


bkg_func_vec = np.vectorize(bkg_func)


def make_singles_file(h5_file):
    with tb.open_file(h5_file, 'r+') as io_file_h5:
        hit_data = io_file_h5.root.hit_data[:]
        singles = analysis.singular_hits_tdc_pix_flav(hit_data=hit_data)
        io_file_h5.create_table(io_file_h5.root, 'singles', singles, filters=tb.Filters(complib='zlib', complevel=5, fletcher32=False))
    return singles


v_i_per_c_i = np.arange(0.001, 1.21, 0.025) * analysis.cap_fac() * 1000
v_i = np.arange(0.001, 1.21, 0.025)


def tdc_to_v_inj_spline(h5_file, pix=None, load=False):
    #------------------------------------------------------------------------------
    # returns a spline to convert tdc channel to V_inj/C_inj
    #------------------------------------------------------------------------------
    if load:
        ld = np.load('/home/daniel/Documents/InterestingPlots/chip6/hitor_calibration/' + str(pix) + '.npy')
        spl = ld.item()['spl']
        average_tdcs = ld.item()['tdcs']
        errs = ld.item()['errs']
    else:
        try:
            with tb.open_file(h5_file, 'r+') as io_file_h5:
                singles = io_file_h5.root.singles[:]
        except:
            singles = make_singles_file(h5_file)

        average_tdcs = [np.mean(singles[singles['scan_param_id'] == x]['tdc']) for x in np.unique(singles['scan_param_id'])]  # x axis
        errs = [np.std(singles[singles['scan_param_id'] == x]['tdc']) / 2 for x in np.unique(singles['scan_param_id'])]

        # use spline to get the converstion from inj -> TDC
        to_be_removed = []
        for x in range(len(average_tdcs) - 1):
            if average_tdcs[x + 1] - average_tdcs[x] < 0:
                to_be_removed.append(x)
        for i in to_be_removed:
            del average_tdcs[i]
            del errs[i]
        spl = UnivariateSpline(average_tdcs, v_i[len(v_i) - len(average_tdcs):], k=3, s=0.0004)
#         spl.set_smoothing_factor(0.5)
        chdir('/home/daniel/Documents/InterestingPlots/chip6/hitor_calibration/')
        np.save(str(pix) + '.npy', {'spl': spl, 'tdcs': average_tdcs, 'errs': errs})

    return spl, average_tdcs, errs


def fit_bkg(data, pix):
    # fit background of x-ray tube
    # returns function parameters for bkg_func()
    peaks = yaml.load(open('/home/daniel/Documents/InterestingPlots/chip6/peaks_guess.yaml'))['bkg'][str(pix)]

    wh = np.where(data <= 3.)[0]
    start = wh[wh < 50][-1]

    try:
        popt, _ = curve_fit(bkg_func_vec, np.arange(start, data.shape[0]), data[start:],  sigma=1 / np.sqrt(data[start:]),
                            p0=[peaks[0], 3, -10,  # switch + lin
                                100., 300., 8., 0.,  # exp
                                750., peaks[1], 4.,  # gaus1
                                600., peaks[2], 50.,  # gaus2 (underlaying)
                                600., peaks[3], 6.,  # gaus3
                                500., peaks[4], 4.,  # gaus4
                                35., 220, 40.])  # gaus5
#                             bounds=([10,   0, -200, -np.inf, -np.inf, -np.inf, -1000, 100, peaks[1] - 30, -20,   0, peaks[2] - 75, -100, -150, peaks[3] - 20, -10, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
#                                     [50, 1000,   0,  np.inf,  np.inf,  np.inf,  1000, 1000, peaks[1] + 30,  20, 2000, peaks[2] + 75,  100,  4000, peaks[3] + 20,  10,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]))

        print popt
        return popt

    except Exception as e:
        print(e)
        print "bkg fit fail"

        return [peaks[0], 3, -10,  # switch + lin
                100., 300., 8., 0.,  # exp
                250., peaks[1], 4.,  # gaus1
                100., peaks[2], 50.,  # gaus2 (underlaying)
                300., peaks[3], 6.,  # gaus3
                250., peaks[4], 4.,  # gaus4
                35., 220, 40.]


def remove_bkg(data, pix, scan_time, bins, bkg_data=None):
    # get the average background /s /bin /pixel and subtract from data with a target
    # option for implementation of fit available if a better fitting algorythim is found
    try:
        bkg_singles = bkg_data
    except bkg_data:
        bkg_singles = get_singles('bkg')
    bkg_singles = bkg_singles[(bkg_singles['col'] == pix[0]) & (bkg_singles['row'] == pix[1])]
    tdc_data = bkg_singles['tdc']
    bkg_singles, bins_bkg = np.histogram(tdc_data, (max(tdc_data)), range=(0, max(tdc_data)))

    bkg_per_s = bkg_singles / (10. * 3600. + 60. * 2 + 31. + 14165. + 21600.)

    exp_bkg = bkg_per_s * scan_time

    if len(exp_bkg) > len(data):
        diff = len(exp_bkg) - len(data)
        data = np.pad(data, (0, diff), 'constant')
        return np.round(np.subtract(np.array(data), np.array(exp_bkg)), 0), bins_bkg, bkg_singles, bins_bkg
    if len(exp_bkg) < len(data):
        diff = len(data) - len(exp_bkg)
        exp_bkg = np.pad(exp_bkg, (0, diff), 'constant')
        return np.round(np.subtract(np.array(data), np.array(exp_bkg)), 0), bins, bkg_singles, bins_bkg
    return np.round(np.subtract(np.array(data), np.array(exp_bkg)), 0), bins, bkg_singles, bins_bkg


#         return data - exp_bkg

def peak_finder(data):
    # peaks finder for the 1st two peaks in Cd...may need to adjust the cut values
    peaks = []
    first = []
    first_idx = []
    second = []
    second_idx = []

    for i in range(data.shape[0] - 2):
        r = range(i, i + 7)
        try:
            loop_data = data[r]
        except IndexError:
            loop_data = data[r[0]:]

        m = (loop_data[-1] - loop_data[0]) / len(r)  # /10 because its m/2 after integration
        b = loop_data[-1] - (m * r[-1])
        expect_integral = ((m / 2) * (r[-1]**2) + r[-1] * b) - ((m / 2) * (r[0]**2) + r[0] * b)
        s = np.sum(loop_data)
        comp = float(s) - float(expect_integral)
        # in each loop need to choose the ones which
        if comp > 500. and i < 70:
            first.append(comp)
            first_idx.append(i)
        if comp > 4000. and i > 90:
            second.append(comp)
            second_idx.append(i)

    peaks.append([first_idx[np.argmax(first)] + 2, second_idx[np.argmax(second)] + 2])
    return peaks


def get_fits(data, src, pix):
    peaks_guess = yaml.load(open('/home/daniel/Documents/InterestingPlots/chip6/peaks_guess.yaml'))[src]

    if src == 'Cd109':
        fit_results = []
        error_list = []
        p = peaks_guess[str(pix)]

        try:
            # double gaussian
            fit, errors = analysis.fit_double_gauss(range(p[1] - 5, p[1] + 28), data[p[1] - 5:p[1] + 28],
                                                    [12000, p[1], 3, 3000, p[2], 2],
                                                    bounds=([0, p[1] - 5, 1, 0, p[2] - 5, 0.5], [np.inf, p[1] + 5, 10, np.inf, p[2] + 5, 10]))
            fit_results.append(fit)
            error_list.append(errors)
#             print "double", fit
        except Exception as e:
            print(e)
            print "double gause fit fail"

        try:
            fit, errors = analysis.fit_gauss_lin(range(p[0] - 15, p[0] + 15), data[p[0] - 15:p[0] + 15], [2000, p[0], 3.1, -0.5, 2000],
                                                 bounds=([0, p[0] - 5, 0, -np.inf, 0], [np.inf, p[0] + 5, 8, np.inf, np.inf]))
            fit_results.append(fit)
            error_list.append(errors)
#             print "single", fit
        except:
            print "gause fit fail"
        return fit_results, error_list

    #------------------------------------------------------------------------------
    if src == 'Am241':
        fit_results = []
        error_list = []
        peaks = peaks_guess[str(pix)]

        try:
            # double gaussian
            if peaks[0] == 0:
                # fit single gaussian to second peak
                fit, errors = analysis.fit_gauss(range(peaks[1] - 4, peaks[1] + 8), data[peaks[1] - 4: peaks[1] + 8],
                                                 [60, peaks[1], 3], errors=True)
                fit_results.append(fit)
                error_list.append(errors)
#                 print "single", fit
            else:
                fit, errors = analysis.fit_double_gauss(range(peaks[0] - 4, peaks[1] + 4), data[peaks[0] - 4:peaks[1] + 4], [50, peaks[0] + 1.5, 3, 50, peaks[1] + 2, 3],
                                                        bounds=([0, peaks[0] - 4, 0, 0, peaks[1] - 4, 0], [np.inf, peaks[0] + 4, 8.5, np.inf, peaks[1] + 4, 10]))
                fit_results.append(fit)
                error_list.append(errors)
#                 print "double", fit

        except Exception as e:
            print(e)
            print "double gause fit fail"

        try:
            fit, errors = analysis.fit_gauss(range(peaks[2] - 8, len(data)), data[peaks[2] - 8:], [150, peaks[2], 2], errors=True,
                                             bounds=([0, peaks[2] - 2, 0], [np.inf, peaks[2] + 2, 6.0]))
            fit_results.append(fit)
            error_list.append(errors)
#             print "single", fit
        except Exception as e:
            print(e)
            print "gause fit fail"

        return fit_results, error_list
    #------------------------------------------------------------------------------
    if src == 'Sn':
        fit_results = []
        error_list = []
        p = peaks_guess[str(pix)]

        try:
            # double gaussian
            fit, errors = analysis.fit_double_gauss(range(p[1] - 10, p[1] + 28), data[p[1] - 10:p[1] + 28],
                                                    [1500, p[1], 3, 350, p[2], 3.],
                                                    bounds=([1000, p[1] - 6, -10, 0, p[2] - 5, 1.1], [np.inf, p[1] + 6, 10, np.inf, p[2] + 5, 5.5]))
            fit_results.append(fit)
            error_list.append(errors)
#             print "double", fit

        except Exception as e:
            print(e)
            print "double gause fit fail"

        try:
            fit, errors = analysis.fit_gauss_lin(range(p[0] - 10, p[0] + 10), data[p[0] - 10:p[0] + 10], [300, p[0], 2, -0.1, 100],
                                                 bounds=([0, p[0] - 5, 0, -np.inf, 0], [np.inf, p[0] + 5, 8, np.inf, np.inf]))
            fit_results.append(fit)
            error_list.append(errors)
#             print "single", fit
        except:
            print "gause fit fail"
        return fit_results, error_list
    #------------------------------------------------------------------------------
    if src == 'Mo':
        fit_results = []
        error_list = []
        p = peaks_guess[str(pix)]

        try:
            # double gaussian
            fit, errors = analysis.fit_double_gauss(range(p[1] - 10, p[1] + 28), data[p[1] - 10:p[1] + 28],
                                                    [2500, p[1], 3, 750, p[1] + 10, 3.1],
                                                    bounds=([1000, p[1] - 4, 1, 100, p[1] + 2, 1.5], [np.inf, p[1] + 5, 10, 1500, p[1] + 15, 5]))
            fit_results.append(fit)
            error_list.append(errors)
#             print "double", fit

        except Exception as e:
            print(e)
            print "double gause fit fail"

        try:
            fit, errors = analysis.fit_gauss_lin(range(p[0] - 10, p[0] + 10), data[p[0] - 10:p[0] + 10], [300, p[0], 2, -0.1, 100],
                                                 bounds=([0, p[0] - 5, 0, -np.inf, 0], [np.inf, p[0] + 5, 8, np.inf, np.inf]))
            fit_results.append(fit)
            error_list.append(errors)
#             print "single", fit
        except:
            print "gause fit fail"

        return fit_results, error_list

    #------------------------------------------------------------------------------
    if src == 'Nb':
        fit_results = []
        error_list = []
        p = peaks_guess[str(pix)]

        try:
            # double gaussian
            fit, errors = analysis.fit_double_gauss(range(p[1] - 10, p[2] + 15), data[p[1] - 10:p[2] + 15],
                                                    [4000, p[1], 2.1, 800, p[2], 2],
                                                    bounds=([1000, p[1] - 5, 1, 100, p[2] - 4, 1.5], [np.inf, p[1] + 5, 4.0, 2000, p[2] + 5, 5]))
            fit_results.append(fit)
            error_list.append(errors)
#             print "double", fit

        except Exception as e:
            print(e)
            print "double gause fit fail"

        try:
            fit, errors = analysis.fit_gauss_lin(range(p[0] - 10, p[0] + 10), data[p[0] - 10:p[0] + 10], [300, p[0], 2, -0.1, 100],
                                                 bounds=([0, p[0] - 5, 0, -np.inf, 0], [np.inf, p[0] + 5, 8, np.inf, np.inf]))
            fit_results.append(fit)
            error_list.append(errors)
#             print "single", fit
        except:
            print "gause fit fail"

        return fit_results, error_list


def peak_fits(bkg_reduction=False):
    if bkg_reduction == True:
        print 'doing fits with background reduction'
        pdfName = "/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/charge_calib_fits_wo_bkg.pdf"
        bkg_singles = get_singles('bkg')
        print 'finished bkg_singles'
        pdfName2 = "/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/charge_calib_bkg.pdf"
        pp2 = PdfPages(pdfName2)
        bkg_fitted = [False for _ in range(len(pix_list))]
    else:
        print 'doing fits without backgroud reduction'
        pdfName = "/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/charge_calib_fits_w_bkg.pdf"
    pp = PdfPages(pdfName)

    cols = ['pix_flav', 'fe_flav', 'thresh', 'thresh_err', 'noise', 'noise_err', 'pixel', 'source', 'peak_energy',
            'A', 'A_err', 'mu', 'mu_err', 'sigma', 'sigma_err', 'chi2', 'p-value', 's/n']
    out_list = []

    for src in ['Nb', 'Sn', 'Mo', 'Cd109']:  # , 'Am241']:
        singles = get_singles(src)
        print "\n---------------------------------------------------------------------------------------\n\t\t", src
        print "\n---------------------------------------------------------------------------------------\n"
        for i, pix in enumerate(pix_list):
            print pix

            base_list = df_basics(pix)

            hit_data2 = singles[(singles['col'] == pix[0]) & (singles['row'] == pix[1])]
            #------------------------------------------------------------------------------
            fig1 = Figure()
            _ = FigureCanvas(fig1)
            ax1 = fig1.add_subplot(111)

            tdc_data = hit_data2['tdc']
            bar_data, bins = np.histogram(tdc_data, (max(tdc_data)) / 1, range=(0, max(tdc_data)))
#             peak_finder(bar_data)

            if src in ['Nb', 'Sn', 'Mo'] and bkg_reduction == True:
                if src == 'Sn':
                    bar_data, bins, pix_bkg, bins_bkg = remove_bkg(bar_data, pix, bins=bins, scan_time=4 * 3600, bkg_data=bkg_singles)
                else:
                    bar_data, bins, pix_bkg, bins_bkg = remove_bkg(bar_data, pix, bins=bins, scan_time=6 * 3600, bkg_data=bkg_singles)

                file_bkg = '/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/bkg/bkg_hists/hist' + \
                    src + '_' + str(pix) + '.txt'
                file_bins = '/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/bkg/bkg_hists/bins' + \
                    src + '_' + str(pix) + '.txt'
                np.save(file_bkg, pix_bkg)
                np.save(file_bins, bins_bkg)

                if bkg_fitted[i] == False:
                    fig2 = Figure()
                    _ = FigureCanvas(fig2)
                    ax2 = fig2.add_subplot(111)
                    ax2.bar(x=bins_bkg[:-1], height=pix_bkg, width=np.diff(bins_bkg[:-1])[0], align="center", alpha=0.7)
#                     bkg_fit = fit_bkg(pix_bkg, pix)
#                     lin1 = np.linspace(0, pix_bkg.shape[0], 2000)
#                     ax2.plot(lin1, bkg_func_vec(lin1, *bkg_fit), 'r--')

                    ax2.set_title("Background Spectrum\n Pixel: (%s,%s)" % (str(pix[0]), str(pix[1])))
                    ax2.set_xlabel("TDC channel")
                    ax2.set_ylabel("Counts")
                    ax2.grid()
                    pp2.savefig(fig2, layout='tight')
                    bkg_fitted[i] = True

            bar_data[bar_data < 0] = 0
            bin_left = bins[:-1]
            ax1.bar(x=bin_left, height=bar_data, width=np.diff(bin_left)[0], align="center", alpha=0.7)
            ax1.set_title("Spectrum of %s\n Pixel: (%s,%s)" % (str(src), str(pix[0]), str(pix[1])))
            ax1.set_xlabel("TDC channel")
            ax1.set_ylabel("Counts")
#             ax1.set_yscale('log')

            fits, errors = get_fits(bar_data, src, pix)
            lin = np.linspace(0, bar_data.shape[0], 2000)
            if fits:
                for f, e in zip(fits, errors):
                    if len(f) == 3:
                        r = np.where((lin > f[1] - 15) & (lin < f[1] + 15))

                        ax1.plot(lin[r], analysis.gauss(lin[r], *f), 'r--', label=((r'$\mu_{1}: %s' % str(np.round(f[1], 2))) + r'\pm %s' % str(np.round(e[1], 2)) +
                                                                                   r'$  $' +
                                                                                   (r'\sigma_{1}:%s' % str(np.round(f[2], 2))) + r'\pm %s$' % str(np.round(e[2], 2))))

                        chi2 = ss.chisquare(bar_data[int(f[1] - f[2] * 1):int(f[1] + f[2] * 1)] / 10.,
                                            analysis.gauss(np.arange(int(f[1] - f[2] * 1), int(f[1] + f[2] * 1)), *f) / 10.,
                                            ddof=(bar_data[int(f[1] - f[2] * 1):int(f[1] + f[2] * 1)].shape[0] - 1 - 3))
                        print 'single chi2:', chi2

                        # update df here
                        if src == 'Am241':
                            hold5 = list(base_list)
                            if f[1] > 170.:
                                hold5.extend([str(pix), src, src_peaks[src][-1]])
                            else:
                                hold5.extend([str(pix), src, src_peaks[src][-2]])
                            hold5.extend([f[0], e[0], f[1], e[1], f[2], e[2]])
                            hold5.extend([chi2[0], chi2[1], f[0] / f[2]])
                            out_list.append(hold5)
                    elif len(f) == 6:

                        ax1.plot(lin, analysis.double_gauss(lin, *f), 'm-', linewidth=0.75, label=((r'$\mu_{1}: %s' % str(np.round(f[1], 2))) + r'\pm %s' % str(np.round(e[1], 2)) +
                                                                                                   r'$  $' +
                                                                                                   (r'\sigma_{1}:%s' % str(np.round(f[2], 2))) + r'\pm %s$' % str(np.round(e[2], 2)) +
                                                                                                   '\n' +
                                                                                                   (r'$\mu_{2}: %s' % str(np.round(f[4], 2))) + r'\pm %s' % str(np.round(e[4], 2)) +
                                                                                                   r'$  $' +
                                                                                                   (r'\sigma_{2}:%s' % str(np.round(f[5], 2))) + r'\pm %s$' % str(np.round(e[5], 2))))

                        chi2 = ss.chisquare(bar_data[int(f[1] - f[2] * 1. - 1):int(f[4] + f[5] * 1.)] / 100.,
                                            analysis.double_gauss(np.arange(int(f[1] - f[2] * 1.) - 1, int(f[4] + f[5] * 1.)), *f) / 100.,
                                            ddof=(bar_data[int(f[1] - f[2] * 1. - 1):int(f[4] + f[5] * 1.)].shape[0] - 7))
                        print 'double gauss chi2:', chi2
                        if src in ['Sn', 'Mo', 'Nb', 'Cd109']:
                            hold = list(base_list)
                            hold.extend([str(pix), src, src_peaks[src][1]])
                            hold.extend([f[0], e[0], f[1], e[1], f[2], e[2]])
                            hold.extend([chi2[0], chi2[1], f[0] / f[2]])
                            out_list.append(hold)

                            hold2 = list(base_list)
                            hold2.extend([str(pix), src, src_peaks[src][2]])
                            hold2.extend([f[3], e[3], f[4], e[4], f[5], e[5]])
                            hold2.extend([chi2[0], chi2[1], f[3] / f[5]])
                            out_list.append(hold2)

                        if src in ['Am241']:
                            hold = list(base_list)
                            hold.extend([str(pix), src, src_peaks[src][0]])
                            hold.extend([f[0], e[0], f[1], e[1], f[2], e[2]])
                            hold.extend([chi2[0], chi2[1], f[0] / f[2]])
                            out_list.append(hold)

                            hold2 = list(base_list)
                            hold2.extend([str(pix), src, src_peaks[src][1]])
                            hold2.extend([f[3], e[3], f[4], e[4], f[5], e[5]])
                            hold2.extend([chi2[0], chi2[1], f[3] / f[5]])
                            out_list.append(hold2)

                    elif len(f) == 5:
                        r = np.where((lin > f[1] - 12) & (lin < f[1] + 12))
                        ax1.plot(lin[r], analysis.gauss_lin(lin[r], *f), 'g-', label=(r'$\mu: %s' % str(np.round(f[1], 2)) + r'\pm %s' % str(np.round(e[1], 2)) + r'$  $' +
                                                                                      r'\sigma:%s' % str(np.round(np.abs(f[2]), 2)) + r'\pm %s$' % str(np.round(e[2], 2))))

                        chisq = ss.chisquare(bar_data[int(f[1] - f[2] * 1.5 + 1):int(f[1] + f[2] * 1.5)] / 100.,
                                             analysis.gauss_lin(np.arange(int(f[1] - f[2] * 1.5) + 1, int(f[1] + f[2] * 1.5)), *f) / 100.,
                                             ddof=(bar_data[int(f[1] - f[2] * 1.5 + 1):int(f[1] + f[2] * 1.5)].shape[0] - 1 - 5))

                        if src in ['Sn', 'Mo', 'Nb', 'Cd109']:
                            hold5 = list(base_list)
                            hold5.extend([str(pix), src, src_peaks[src][0]])
                            hold5.extend([f[0], e[0], f[1], e[1], f[2], e[2]])
                            hold5.extend([chisq[0], chisq[1], f[0] / f[2]])
                            out_list.append(hold5)
                    else:
                        print "returned fit value not in parameter space, len=", len(f)
#             print out_list
            ax1.grid()
            ax1.legend()
            fig1.tight_layout()
            pp.savefig(fig1, layout='tight')
            plt.clf()
            print

#     print out_list
    fit_df = pd.DataFrame(out_list, columns=cols)
    if bkg_reduction == True:
        fit_df.to_csv("/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/fits_wo_bkg.csv", sep='\t')
    else:
        fit_df.to_csv("/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/fits_w_bkg.csv", sep='\t')

    pp.close()
    if bkg_reduction == True:
        pp2.close()

    print 'finished fits'
    return fit_df


def constant_func(x, c):
    return x + c


def make_Vinj_v_kev_plots(fit_df, bkg_reduction=False):
    print 'starting Vinj vs keV plots'
    if bkg_reduction:
        pdfName = "/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/charge_calib_vinj_v_kev_wo_bkg.pdf"
    else:
        pdfName = "/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/charge_calib_vinj_v_kev_w_bkg.pdf"
    pp = PdfPages(pdfName)
    q_e = 1.6021766E-19
    avg = 0
    avg2 = 0
#     w = 0.00365  # keV/(e/h pair)
    w = 0.00365
    cols = ['pix_flav', 'fe_flav', 'thresh', 'thresh_err', 'noise', 'noise_err', 'pixel',
            'm', 'm_err', 'b', 'b_err', 'chisq', 'p', 'r2', 'true_thresh', 'true_thresh_err', 'inj_cap', 'inj_cap_err']
    labels = [r'Cu $K$', r'Cu $K$', r'Cu $K$', r'Cu $K$',
              r'Nb $K_{\alpha}$', r'Mo $K_{\alpha}$',
              r'Nb $K_{\beta}$', r'Mo $K_{\beta}$',
              r'Cd $K_{\alpha}$', r'Cd $K_{\beta}$',
              r'Sn $K_{\alpha}$',
              r'Sn $K_{\beta}$']
    out_list = []
    inj_cap_list = []
    for pix in pix_list:
        base_list = list(df_basics(pix))
        base_list.append(str(pix))
        pix_df = fit_df[fit_df['pixel'] == str(pix)][['peak_energy', 'mu', 'sigma', 'y_err']]
        pix_df['peak_energy'] /= w
        print '\n', pix
        fig1 = Figure()
        _ = FigureCanvas(fig1)
        ax = fig1.add_subplot(111)
        pix_df = pix_df.sort_values(by=['peak_energy'])
        ax.errorbar(pix_df['mu'], pix_df['peak_energy'], xerr=pix_df['y_err'],
                    fmt='o', markersize='3', label='Peak Positions')

        for inj, kev in zip(pix_df['mu'], pix_df['peak_energy']):
            inj_cap_list.append(kev / 0.00365 / inj * q_e)
#             print

        flip = False
        for label, x, y in zip(labels, pix_df['mu'], pix_df['peak_energy']):
            if flip == True:
                ax.annotate(label,
                            xy=(x, y), xytext=(40, -30),
                            textcoords='offset points', ha='right', va='bottom',
                            #                             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                flip = False
            else:
                ax.annotate(label,
                            xy=(x, y), xytext=(-20, 20),
                            textcoords='offset points', ha='right', va='bottom',
                            #                             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                flip = True
        ax.set_title(r'$V_{inj}$ vs Peak Energy, Pixel: %s' % str(pix))

        try:
            fit, pcov = curve_fit(analysis.linear, pix_df['mu'], pix_df['peak_energy'],  # sigma=pix_df['y_err'],
                                  p0=[0.0336, -0.028])  # ,  absolute_sigma=True)
            errors = np.sqrt(np.diagonal(pcov))
            chisq = ss.chisquare(pix_df['peak_energy'], f_exp=analysis.linear(pix_df['mu'], *fit),
                                 ddof=len(pix_df['mu']) - 2 - 1)
            r2 = np.sum((pix_df['peak_energy'][0:-1] - analysis.linear(pix_df['mu'], *fit))**2)
#             print chisq, np.sum(((pix_df['mu'][0:-1] - analysis.linear(pix_df['peak_energy'], *fit))**2) / analysis.linear(pix_df['peak_energy'], *fit))
#             fit_results.append(fit)
            print "lin", fit
#             print pcov
#             print np.linalg.eig(pcov)
#             print
#             print uncertainties.correlated_values(fit, pcov)
            errors = np.sqrt(np.linalg.eig(pcov)[0])

            base_list.extend([fit[0], errors[0], fit[1], errors[1]])
            base_list.extend(chisq)
            base_list.extend([r2])

        except Exception as e:
            print(e)
            print "lin fit fail"
            base_list.extend([0, 0])
            base_list.extend(0.0)

        lin = np.linspace(min(pix_df['mu']) - 0.2, max(pix_df['mu']) + 0.2, 10000)

        label = (r'$y = ' + r'(%s' % str(np.round(fit[0], 4)) + r'\pm %s)$x$' % str(np.round(errors[0], 4)) +
                 r'+(%s' % str(np.round(fit[1], 3)) + r'\pm %s)$' % str(np.round(errors[1], 3)))
        ax.plot(lin, analysis.linear(lin, *fit), 'g-.', label=label)

        avg += ((pix_df['mu'] - fit[1]) / fit[0]).iloc[0] - pix_df['peak_energy'].iloc[0]
        avg2 += ((pix_df['mu'] - fit[1]) / fit[0]).iloc[0]
        ax.set_ylabel("Peak Energy, [e]")
        ax.set_xlabel(r'$V_{inj}$, [V]')
        ax.legend()
        ax.grid()
        vinj = unc.ufloat(base_list[2], base_list[3])
        m = unc.ufloat(fit[0], errors[0])
        b = unc.ufloat(fit[1], errors[1])
        true_thres_w_err = m * vinj + b
        cap_err = (true_thres_w_err - b) * q_e / vinj
#         cap_err = (true_thres_w_err * q_e) / (vinj - b)

        fig1.tight_layout()
        pp.savefig(fig1, layout='tight')
        plt.clf()
        base_list.extend([true_thres_w_err.n, true_thres_w_err.s, cap_err.n, cap_err.s])
        out_list.append(base_list)
    calib_df = pd.DataFrame(out_list, columns=cols)
    calib_df.to_csv("/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/calib_values.csv", sep='\t')
    print calib_df.loc[:, ['pixel', 'm', 'b', 'true_thresh', 'true_thresh_err', 'inj_cap', 'inj_cap_err']]
    print 'average threshold:', np.mean(calib_df['true_thresh']), '+/-', np.mean(calib_df['true_thresh_err']), 'std', np.std(calib_df['true_thresh'])
    print 'average cap:', np.mean(calib_df['inj_cap']), '+/-', np.mean(calib_df['inj_cap_err']), 'std', np.std(calib_df['inj_cap'])
    print 'inj_cap_list:', np.mean(inj_cap_list), np.std(inj_cap_list)
    pp.close()
    print 'finished inj v keV plots'

    return calib_df


def convert_tdc_to_vinj(fit_df, background_reduction=False, load_splines=True):
    pp = PdfPages("/home/daniel/Documents/InterestingPlots/chip6/hitor_calibration/splines.pdf")
    hitor_files = yaml.load(open('/home/daniel/Documents/InterestingPlots/chip6/hitor_calibration/files_dict.yaml'))['files']
    print 'starting splines'
    fit_df["y_err"] = np.nan

    for pix in pix_list:
        print pix
        spl, tdcs, errs = tdc_to_v_inj_spline(hitor_files[str(pix)], pix=pix, load=load_splines)
        #------------------------------------------------------------------------------
        # plotting of splines to check data
        # x -> tdc vals
        # y-> Vinj/Cinj
        lin = np.linspace(0, max(tdcs), 1000)
        fig1 = Figure()
        _ = FigureCanvas(fig1)
        ax = fig1.add_subplot(111)
        ax.plot(lin, spl(lin), 'r--', label='Spline')
        ax.errorbar(tdcs, v_i[len(v_i) - len(tdcs):], xerr=errs, fmt='go', markersize=2.5, label='Average TDC Channel')

        ax.set_title(r'$V_{inj}$ vs TDC Channel, Pixel %s' % str(pix))
        ax.set_xlabel("TDC Channel")
        ax.set_ylabel(r"$V_{inj}, [V]$")
        ax.set_ylim([0, max(spl(lin)) + 0.02])
        ax.legend()
        ax.grid()
        fig1.tight_layout()
        pp.savefig(fig1, layout='tight')
        plt.clf()
        ch2_errs = [(x * 2)**2 for x in errs]
#         print v_i[len(v_i) - len(tdcs):] - spl(tdcs)**2
#         print ch2_errs**2
#         chisq = np.sum(((v_i[len(v_i) - len(tdcs):] - spl(tdcs))**2) / ch2_errs)
#         print chisq
#         #------------------------------------------------------------------------------
#         print spl.get_residual() / np.sum(ch2_errs)
        # error on pts are derivative(spl)
        yerr_list = []
        for x in fit_df.loc[fit_df['pixel'] == str(pix), 'mu']:
            yerr_list.append(spl.derivatives(x)[1] / 2)
        fit_df.loc[fit_df['pixel'] == str(pix), 'y_err'] = yerr_list
        fit_df.loc[fit_df['pixel'] == str(pix), 'mu'] = spl(fit_df.loc[fit_df['pixel'] == str(pix), 'mu'])

    pp.close()

    if background_reduction == True:
        fit_df.to_csv("/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/fits_converted_wo_bkg.csv", sep='\t')
    else:
        fit_df.to_csv("/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/fits_converted_w_bkg.csv", sep='\t')
    print 'finished splines'
    return fit_df


if __name__ == "__main__":

    load_fits_from_file = False
    background_reduction = False

    if load_fits_from_file:
        print 'loading fits from file'
        if background_reduction:
            fit_df = pd.read_csv('/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/fits_wo_bkg.csv', sep='\t')
        else:
            fit_df = pd.read_csv('/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/fits_w_bkg.csv', sep='\t')
    else:
        fit_df = peak_fits(bkg_reduction=background_reduction)

    load_tdc_conv = False
    load_splines = True
    if not load_tdc_conv:
        fit_df = convert_tdc_to_vinj(fit_df, load_splines=load_splines)
    else:
        if background_reduction:
            fit_df = pd.read_csv("/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/fits_converted_wo_bkg.csv", sep='\t')
        else:
            fit_df = pd.read_csv("/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/fits_converted_w_bkg.csv", sep='\t')

    load_calib_data = False
    if not load_calib_data:
        calib_df = make_Vinj_v_kev_plots(fit_df, bkg_reduction=background_reduction)
    else:
        calib_df = pd.read_csv('/home/daniel/Documents/InterestingPlots/chip6/chrage_calibration/calib_values.csv', sep='\t')
