
import logging
import numpy as np
import tables as tb
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy import stats as ss
import yaml


def cap_fac():
    return 7.989  # capacitance converted to 1/V (divided by 1000) (calculated by divided by elem charge)


def inj_cap():
    return 1.18  # unit: fF


class TDCTable(tb.IsDescription):
    pix_num = tb.UInt16Col(pos=0)
    elecs = tb.Float64Col(pos=1)
    mu_tdc = tb.Float64Col(pos=2)
    sigma_tdc = tb.Float64Col(pos=3)
    mu_delay = tb.Float64Col(pos=4)
    sigma_delay = tb.Float64Col(pos=5)
    num_rec = tb.Float64Col(pos=6)
    #trigger = tb.Float64Col(pos=7)


class TDCFitTable(tb.IsDescription):
    pix_num = tb.UInt16Col(pos=0)
    m = tb.Float64Col(pos=1)
    b = tb.Float64Col(pos=2)
    chisq = tb.Float64Col(pos=3)
    p_val = tb.Float64Col(pos=4)


def tdc_table_w_srcs(h5_file_in, h5_file_old, out_file_name='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/tdc_calib.h5'):
    # need to have the data from the current scan, old data from inj/other srcs, and outputfile

    # load tdc table from prev inj scan
    # append the previous data with the collected data
    #     data to add: pix_num, elecs, mu_tdc, sigma_tdc, mu_delay, sigma_delay, num_rec
    #      = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/tdc_calib.h5'

    with tb.open_file(h5_file_old, '+r') as old_h5:
        old_tdc_table = old_h5.root.tdc_data[:]
        old_fit_data = old_h5.root.fit_data[:]
    with tb.open_file(h5_file_in, '+r') as curr_h5:
        meta_data_in = curr_h5.root.meta_data[:]
        raw_data_in = curr_h5.root.raw_data[:]
        scan_args_in = yaml.load(curr_h5.root.meta_data.attrs.kwargs)
        src_elecs = scan_args_in['source_elecs']
        pix_list = range(scan_args_in['pix_range'][0], scan_args_in['pix_range'][1])

    with tb.open_file(out_file_name, mode='w', title='characterization') as out_file:
        new_tdc_table = out_file.create_table(out_file.root, name='tdc_data', description=TDCTable, title='tdc_data')
        for pix in pix_list:
            # need to getthe old tdc table and add it to the new one
            for table_rows in range(old_tdc_table[old_tdc_table['pix_num'] == pix].shape[0]):
                new_tdc_table.row['pix_num'] = old_tdc_table['pix_num'][table_rows]
                new_tdc_table.row['elecs'] = old_tdc_table['elecs'][table_rows]
                new_tdc_table.row['mu_tdc'] = old_tdc_table['mu_tdc'][table_rows]
                new_tdc_table.row['sigma_tdc'] = old_tdc_table['sigma_tdc'][table_rows]
                new_tdc_table.row['mu_delay'] = old_tdc_table['mu_delay'][table_rows]
                new_tdc_table.row['sigma_delay'] = old_tdc_table['sigma_delay'][table_rows]
                new_tdc_table.row['num_rec'] = old_tdc_table['num_rec'][table_rows]
                new_tdc_table.row.append()
                new_tdc_table.flush()

            # code to get the data for the src run
            # mu is the peak of the spectrum of the source, must see how the spectrum of the source looks first
            meta_data_hold = meta_data_in[meta_data_in['scan_param_id'] == pix]
            if meta_data_hold.shape[0] != 0:
                data_start = min(meta_data_hold['index_start'])
                data_stop = max(meta_data_hold['index_stop'])
                raw_data_range = np.arange(data_start, data_stop)
                raw_data_idx = raw_data_in[raw_data_range]
                tdc_data_idx = raw_data_idx & 0x0FFF
                tdc_delay_idx = (raw_data_idx & 0x0FF00000) >> 20
                # fit gaussian(s) to source data
                # TODO: fix this for multiple peaks
                y_data = np.histogram(tdc_data_idx, (max(tdc_data_idx) - min(tdc_data_idx)))
                gauss_fit_params = fit_gauss(tdc_data_idx, y_data, params_guess=np.ndarray(100, tdc_data_idx[y_data == max(y_data)], 0.1))
                mu_data_idx = gauss_fit_params[1]
                std_data_idx = gauss_fit_params[2]
                mu_delay_idx = np.mean(tdc_delay_idx)
                std_delay_idx = np.std(tdc_delay_idx)

                new_tdc_table.row['pix_num'] = pix
                new_tdc_table.row['elecs'] = src_elecs
                new_tdc_table.row['mu_tdc'] = mu_data_idx
                new_tdc_table.row['sigma_tdc'] = std_data_idx
                new_tdc_table.row['mu_delay'] = mu_delay_idx
                new_tdc_table.row['sigma_delay'] = std_delay_idx
                new_tdc_table.row['num_rec'] = raw_data_idx.shape[0]
                new_tdc_table.row.append()
                new_tdc_table.flush()


def create_tdc_inj_table(h5_file_name):
    with tb.open_file(h5_file_name, 'r+') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        raw_data = in_file_h5.root.raw_data[:]
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        scan_range = scan_args['scan_range']
        if scan_range.shape[0] == 3:
            scan_range_inx = np.arange(scan_range[0], scan_range[1], scan_range[2])
            len_range = scan_range_inx.shape[0]
        else:
            scan_range_inx = scan_range
            len_range = len(scan_range_inx)
        pixel_range = scan_args['pixel_range']
        tdc_table = in_file_h5.create_table(in_file_h5.root, name='tdc_data', description=TDCTable, title='tdc_data')

        for pix in range(pixel_range[0], pixel_range[1]):
            for i in range(len_range):
                local_scan_idx = pix * len_range + i
                meta_data_idx = meta_data[(meta_data['scan_param_id'] == int(local_scan_idx))]

                if meta_data_idx.shape[0] != 0:
                    data_start = min(meta_data_idx['index_start'])
                    data_stop = max(meta_data_idx['index_stop'])
                    raw_data_range = np.arange(data_start, data_stop)
                    raw_data_idx = raw_data[raw_data_range]
                    tdc_data_idx = raw_data_idx & 0x0FFF
                    tdc_delay_idx = (raw_data_idx & 0x0FF00000) >> 20
                    if tdc_data_idx[tdc_delay_idx < 253].shape[0] != 0:
                        mu_data_idx = np.mean(tdc_data_idx[tdc_delay_idx < 253])
                        std_data_idx = np.std(tdc_data_idx[tdc_delay_idx < 253])
                        mu_delay_idx = np.mean(tdc_delay_idx[tdc_delay_idx < 253])
                        std_delay_idx = np.std(tdc_delay_idx[tdc_delay_idx < 253])

                        # save everything in tdc_tble
                        tdc_table.row['pix_num'] = pix
                        tdc_table.row['elecs'] = scan_range_inx[i]
                        tdc_table.row['mu_tdc'] = mu_data_idx
                        tdc_table.row['sigma_tdc'] = std_data_idx
                        tdc_table.row['mu_delay'] = mu_delay_idx
                        tdc_table.row['sigma_delay'] = std_delay_idx
                        tdc_table.row['num_rec'] = tdc_data_idx[tdc_delay_idx < 253].shape[0]
                        tdc_table.row.append()
                        tdc_table.flush()
            if pix % 100 == 0:
                print "finished pixel:", pix


def analyze_pixel_calib_inj(h5_file_name):
    # want to save the data in tables with the following:
    # pixel number | inj electrons | mu_data | sigma_data | mu_delay | sigma_delay |percent of repeats seen
    #     create_tdc_inj_table(h5_file_name)
    with tb.open_file(h5_file_name, 'r+') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        tdc_data = in_file_h5.root.tdc_data[:]
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        scan_range = scan_args['scan_range']
        if scan_range.shape[0] == 3:
            scan_range_inx = np.arange(scan_range[0], scan_range[1], scan_range[2])
            len_range = scan_range_inx.shape[0]
        else:
            scan_range_inx = scan_range
            len_range = len(scan_range_inx)
        repeats = scan_args['repeat_command']
        pixel_range = scan_args['pixel_range']
        fit_data = in_file_h5.create_table(in_file_h5.root, name='fit_data', description=TDCFitTable, title='fit_data')
        for pix in range(pixel_range[0], pixel_range[1]):
            tdc_data_hold = tdc_data[(tdc_data['pix_num'] == pix) & (tdc_data['num_rec'] / repeats >= 0.98)
                                     & (tdc_data['num_rec'] / repeats <= 1.02) & (tdc_data['elecs'] >= 3000)]
            # & (tdc_data['elecs'] >= 3000)]
            if tdc_data_hold['elecs'].shape[0] != 0:
                min_inj = min(tdc_data_hold['elecs'])
                max_inj = scan_range_inx[-1]
                max_mu = max(tdc_data_hold['mu_tdc'])
                min_mu = min(tdc_data_hold['mu_tdc'])
                x = tdc_data_hold['elecs']
                y = tdc_data_hold['mu_tdc']
                err = tdc_data_hold['sigma_tdc']
                m_guess = (max_mu - min_mu) / (max_inj - min_inj)
                b_guess = -100.
                if m_guess <= 0:
                    print pix
                try:
                    # print m_guess
                    popt, _ = curve_fit(linear, x, y, p0=[0.5, -10], sigma=err)
                    chi2 = ss.chisquare(y, linear(x, *popt))
                    # print chi2
                except:
                    print 'fit failed'
                    popt = [m_guess, 0.]
            else:
                popt = [0, 0]
                chi2 = [0, 0]

            fit_data.row['pix_num'] = pix
            fit_data.row['m'] = popt[0]
            fit_data.row['b'] = popt[1]
            fit_data.row['chisq'] = chi2[0]
            fit_data.row['p_val'] = chi2[1]
            fit_data.row.append()
            fit_data.flush()


def analyze_threshold_scan(h5_file_name, vth1=False):
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

#         print s_hist.shape
        param_inx = np.ravel(indices[1].astype(np.float64))  # *0.05 - 0.6)

        pix_scan_hist = np.empty((s_hist.shape[1], repeat_command + 10))
        for param in range(s_hist.shape[1]):
            h_count = np.bincount(s_hist[:, param])
            h_count = h_count[:repeat_command + 10]
            pix_scan_hist[param] = np.pad(
                h_count, (0, (repeat_command + 10) - h_count.shape[0]), 'constant')

        log_hist = np.log10(pix_scan_hist)
        log_hist[~np.isfinite(log_hist)] = 0
        #pix_scan_hist_flip = np.swapaxes(pix_scan_hist,0,1)

        threshold = np.empty(64 * 64)
        noise = np.empty(64 * 64)
        chi2 = np.empty(64 * 64)
        x = scan_range_inx

#         en_mask = np.reshape(en_mask, 4096)
#         for i in range(s_hist.shape[1]):
#             pix = np.where(s_hist[:, i] > 105)
#             for j in pix:
#                 en_mask[j] = False
#         en_mask = np.reshape(en_mask, (64, 64))

        for pix in range(64 * 64):
            # this can multi threaded
            if vth1:
                fitOut = fit_scurve(s_hist[pix], x, repeat_command, vth1=False)
            else:
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

        hist_thresh_y, hist_thresh_x = np.histogram(Threshold_pure, density=False, bins=50)
        Noise_pure[Noise_pure > 0.02] = 0.02
        hist_noise_y, hist_noise_x = np.histogram(Noise_pure, density=False, bins=50)
        new_x = ()
        for entries in range(len(hist_thresh_x) - 1):
            new_x = np.append(new_x, (hist_thresh_x[entries] + hist_thresh_x[entries + 1]) / 2)
        hist_thresh_x = new_x
        new_x = ()
        for entries in range(len(hist_noise_x) - 1):
            new_x = np.append(new_x, (hist_noise_x[entries] + hist_noise_x[entries + 1]) / 2)
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

        Thresh_results = in_file_h5.create_group("/", 'Thresh_results', 'Thresh_results')
        Noise_results = in_file_h5.create_group("/", 'Noise_results', 'Noise_results')
        Chisq_results = in_file_h5.create_group("/", 'Chisq_results', 'ChiSq_results')
        Scurves_Measurments = in_file_h5.create_group("/", 'Scurves_Measurments', 'Scurves_Measurments')

        scurve_hist_unform = in_file_h5.create_carray(Scurves_Measurments, name='Scurve', title='Scurve Measurements',
                                                      obj=s_hist)
        # scurve_hist_unform[:]=
        scurve_hist = in_file_h5.create_carray(Scurves_Measurments, name='Scurve_formatted', title='Scurve Histogram',
                                               obj=s_hist.reshape((64, 64, scan_range_inx.shape[0])))
        # need one that is num of occurances vs scan param ... go from the orginal s_hist -> pix_scan_hist
        scurve_thresh_hm = in_file_h5.create_carray(
            Scurves_Measurments, name='threshold_hm', title='numver of pix in bins/scan param', obj=pix_scan_hist)

        # scurve_hist[:]=scurve_formatted

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

#         print 'percent disabled', 100 * (en_mask[en_mask == False].shape[0] / 4096.)
#         en_mask[...] = en_mask


def analyze_tdac_scan(h5_file_name):
    with tb.open_file(h5_file_name, 'r+') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        hit_data = in_file_h5.root.hit_data[:]
        en_mask = in_file_h5.root.scan_results.en_mask[:]
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        scan_range = np.arange(0, 32, 1)
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
#         indices = np.indices(s_hist.shape)

#         param_inx = np.ravel(indices[1].astype(np.float64))  # *0.05 - 0.6)

        pix_scan_hist = np.empty((s_hist.shape[1], repeat_command + 10))
        for param in range(s_hist.shape[1]):
            h_count = np.bincount(s_hist[:, param])
            h_count = h_count[:repeat_command + 10]
            pix_scan_hist[param] = np.pad(
                h_count, (0, (repeat_command + 10) - h_count.shape[0]), 'constant')

        log_hist = np.log10(pix_scan_hist)
        log_hist[~np.isfinite(log_hist)] = 0
        #pix_scan_hist_flip = np.swapaxes(pix_scan_hist,0,1)

        threshold = np.empty(64 * 64)
        sigma = np.empty(64 * 64)
        chi2 = np.empty(64 * 64)
        tdac_val = np.empty(64 * 64)
        en_list = np.empty(64 * 64, dtype=bool)
        x = scan_range_inx

        for pix in range(64 * 64):
            fitOut = fit_zcurve_tdac(s_hist[pix], x, repeat_command)

            # starting at 1, 0 is scalling->should be fixed at repeat_command
            threshold[pix] = fitOut[1]
            sigma[pix] = fitOut[2]
            chi2[pix] = fitOut[3]
            tdac_val[pix] = fitOut[4]
            en_list[pix] = fitOut[5]
        shape = en_mask.shape
        ges = 1

        for i in range(2):
            ges = ges * shape[i]
        Noise_pure = ()
        Threshold_pure = ()
        en_mask = en_mask.reshape(ges)
        Sigma = sigma.reshape(ges)
        Threshold = threshold.reshape(ges)
        ChisqS = chi2.reshape(ges)
        tdac_mask = tdac_val.reshape(ges)
        en_mask_out = en_list.reshape(ges)
        Noise_pure = Sigma[en_mask == True]
        Threshold_pure = Threshold[en_mask == True]
        # print Threshold_pure
#         for n in range(ges):
#             if en_mask[n] == True:
#                 Noise_pure = np.append(Noise_pure, Sigma[n])
#                 Threshold_pure = np.append(Threshold_pure, Threshold[n])

        # TODO: weird, check
#         Threshold_pure[Threshold_pure > scan_range_inx[-1]] = 0

        hist_thresh_y, hist_thresh_x = np.histogram(Threshold_pure, density=False, bins=50)
        Noise_pure[Noise_pure > 0.02] = 0.02
        hist_noise_y, hist_noise_x = np.histogram(Noise_pure, density=False, bins=50)
        new_x = ()
        for entries in range(len(hist_thresh_x) - 1):
            new_x = np.append(new_x, (hist_thresh_x[entries] + hist_thresh_x[entries + 1]) / 2)
        hist_thresh_x = new_x
        new_x = ()
        for entries in range(len(hist_noise_x) - 1):
            new_x = np.append(new_x, (hist_noise_x[entries] + hist_noise_x[entries + 1]) / 2)
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
        sigma = sigma.reshape(64, 64)
        chi2 = chi2.reshape(64, 64)

        Thresh_results = in_file_h5.create_group(
            "/", 'Thresh_results', 'Thresh_results')
        Noise_results = in_file_h5.create_group(
            "/", 'Noise_results', 'Noise_results')
        Chisq_results = in_file_h5.create_group(
            "/", 'Chisq_results', 'ChiSq_results')
        Scurves_Measurments = in_file_h5.create_group(
            "/", 'Scurves_Measurments', 'Scurves_Measurments')

        zcurve_hist_unform = in_file_h5.create_carray(Scurves_Measurments, name='Scurve', title='Scurves_Measurments', obj=s_hist)
        # scurve_hist_unform[:]=
        zcurve_hist = in_file_h5.create_carray(Scurves_Measurments, name='Scurves_Measurments',
                                               title='zcurve Histogram', obj=s_hist.reshape((64, 64, 32)))
        # need one that is num of occurances vs scan param ... go from the orginal s_hist -> pix_scan_hist
        zcurve_thresh_hm = in_file_h5.create_carray(Scurves_Measurments, name='threshold_hm',
                                                    title='numver of pix in bins/scan param', obj=pix_scan_hist)

        threshold_hist = in_file_h5.create_carray(Thresh_results, name='Threshold', title='Threshold Histogram',
                                                  atom=tb.Atom.from_dtype(threshold.dtype), shape=threshold.shape)
        threshold_hist[:] = threshold

        threshold_pure_hist = in_file_h5.create_carray(Thresh_results, name='Threshold_pure', title='Threshold_pure Histogram',
                                                       atom=tb.Atom.from_dtype(Threshold_pure.dtype), shape=Threshold_pure.shape)
        threshold_pure_hist[:] = Threshold_pure
        threshold_pure_hist.attrs.fitdata_thresh = thresh_fit_values

        chisq_hist = in_file_h5.create_carray(where=Chisq_results, obj=chi2, name='Chisq_scurve',
                                              title='Chisq results per pix', atom=tb.Atom.from_dtype(chi2.dtype), shape=chi2.shape)
        chisq_hist_full = in_file_h5.create_carray(where=Chisq_results, obj=ChisqS, name='Chisq_scurve_unformatted',
                                                   title='chisq unformatted', atom=tb.Atom.from_dtype(ChisqS.dtype), shape=ChisqS.shape)

        noise_pure_hist = in_file_h5.create_carray(Noise_results, name='Noise_pure', title='Noise_pure Histogram',
                                                   atom=tb.Atom.from_dtype(Noise_pure.dtype), shape=Noise_pure.shape)
        noise_pure_hist[:] = Noise_pure
        noise_hist = in_file_h5.create_carray(Noise_results, name='Noise', title='noise Histogram',
                                              atom=tb.Atom.from_dtype(Sigma.dtype), shape=sigma.shape)
        noise_hist[:] = sigma
        noise_pure_hist.attrs.fitdata_noise = noise_fit_values

        scan_results = in_file_h5.create_group("/", 'analysis_results', 'Scan Masks')
        in_file_h5.create_carray(scan_results, 'tdac_mask', obj=tdac_mask.reshape(64, 64))
        in_file_h5.create_carray(scan_results, 'en_mask', obj=en_mask_out.reshape(64, 64))


def linear(x, m, b):
    return m * x + b


def scurve(x, A, mu, sigma):
    return 0.5 * A * erf((x - mu) / (np.sqrt(2) * sigma)) + 0.5 * A


def zcurve(x, A, mu, sigma):
    return 0.5 * A * erf((x - mu) / (np.sqrt(2) * sigma)) + 0.5 * A


def fit_zcurve_tdac(zcurve_data, PlsrDAC, repeat_command):
    index = np.argmax(np.diff(zcurve_data))
    maxInject = repeat_command
    q_min = min(PlsrDAC)
    q_max = max(PlsrDAC)
    M = np.sum(PlsrDAC)
    mu_guess = q_max - M / maxInject
    data_errors = np.sqrt(zcurve_data * (1 - (zcurve_data / repeat_command)))

    max_occ = np.median(zcurve_data[index:])
#     if abs(max_occ) <= 1e-08 and not vth1:  # or index == 0: occupancy is zero or close to zero
#         popt = [0, 0, 0]
#     else:
    try:
        popt, _ = curve_fit(zcurve, PlsrDAC, zcurve_data, p0=[repeat_command, 33, -1 * np.std(zcurve_data)], check_finite=False)
        logging.info('Fit-params-zcurve: %s %s %s ', str(popt[0]), str(popt[1]), str(popt[2]))
    except RuntimeError:  # fit failed
        popt = [0, 0, 0]
        logging.info('Fit did not work scurve: %s %s %s', str(popt[0]), str(popt[1]), str(popt[2]))
    chi2 = ss.chisquare(zcurve_data, zcurve(PlsrDAC, *popt))
#     chi2 = np.sum((np.diff(zcurve_data - zcurve(PlsrDAC, *popt))**2) * repeat_command)

    if popt[1] < 0:  # threshold < 0 rarely happens if fit does not work
        popt = [0, 0, 0]

    # cases:
    # 1: 100->0
    # 2: 100+->0
    # 3: 100-> repeats/2
    # 4: 100->0->noise
    # 5: all noise
    # 6: all ~0
    # 7: all = repeats
    thresh = 0

    if 0 in zcurve_data:
        # cases 1 2 4 6
        if max(zcurve_data) <= (repeat_command * 0.4) and np.mean(zcurve_data) <= (repeat_command / 10):
            # case 6
            en = True
            tdac_val = 0
        else:
            for itt in range(len(zcurve_data)):
                if zcurve_data[itt] <= (repeat_command / 2):
                    thresh1 = zcurve_data[itt - 1]
                    thresh2 = zcurve_data[itt]
                    if abs(thresh1 - 50) < abs(thresh2 - 50):
                        thresh = itt - 1
                    else:
                        thresh = itt
                    break
            if popt[1] != 0:
                if 0.9 <= popt[1] / thresh <= 1.1:
                    # cases 1 2 4
                    en = True
                    tdac_val = round(popt[1])
                else:
                    en = True
                    tdac_val = thresh
            else:
                en = True
                tdac_val = thresh
    else:
        # cases 3 5 7
        if zcurve_data[-1] <= (repeat_command * 0.5):
            # case 3
            for itt in range(len(zcurve_data)):
                if zcurve_data[itt] <= (repeat_command / 2):
                    thresh1 = zcurve_data[itt - 1]
                    thresh2 = zcurve_data[itt]
                    if abs(thresh1 - (repeat_command / 2)) < abs(thresh2 - (repeat_command / 2)):
                        thresh = itt - 1
                    else:
                        thresh = itt
                    break
            if popt[1] != 0:  # if there is a fit
                if 0.9 <= popt[1] / thresh <= 1.1:
                    en = True
                    tdac_val = round(popt[1])
                else:
                    en = True
                    tdac_val = thresh
            else:
                en = True
                tdac_val = thresh
        elif 0.97 <= (np.mean(zcurve_data) / repeat_command) <= 1.03:
            # case 7
            en = True
            tdac_val = 31
        elif np.mean(zcurve_data) > repeat_command * 1.1:
            # case 5
            en = False
            tdac_val = 31
        else:
            print "special case!!! thresh: ", popt[1]
            en = True
            tdac_val = 16
    if tdac_val > 31:
        tdac_val = 31
    if tdac_val == -1:
        tdac_val = 0
#     print "threshold of fit", popt[1], " tdac value ", tdac_val
    return [popt[0], popt[1], popt[2], chi2[0], tdac_val, en]

# data of some pixels to fit, has to be global for the multiprocessing module


def fit_scurve(scurve_data, PlsrDAC, repeat_command, vth1=False):
    index = np.argmax(np.diff(scurve_data))
    A = repeat_command
    M = np.sum(scurve_data)
    n = len(PlsrDAC)
    d = (max(PlsrDAC) - min(PlsrDAC)) / (len(PlsrDAC))
    q_min = min(PlsrDAC) - d / 2.
    q_max = max(PlsrDAC) + d / 2.

    data_errors = np.sqrt(scurve_data * (1 - (scurve_data / repeat_command)))
    min_err = np.sqrt(0.5 - 0.5 / repeat_command)
    data_errors[data_errors < min_err] = min_err
    sel_bad = scurve_data > repeat_command
    data_errors[sel_bad] = (scurve_data - repeat_command)[sel_bad]

    mu1 = np.sum(scurve_data[scurve_data <= (repeat_command / 2.)])
    mu2 = np.sum(scurve_data[scurve_data > (repeat_command / 2.)])
    mu_guess = q_max - ((q_max - q_min) * M / (n * A))
    sig_guess = d * ((mu1 + mu2) / A) * np.sqrt(np.pi / 2.)
    max_occ = np.median(scurve_data[index:])

    p0 = [repeat_command, mu_guess, 0.01]
    # print p0
    if abs(max_occ) <= 1e-08 and not vth1:  # or index == 0: occupancy is zero or close to zero
        popt = [0, 0, 0]
    else:
        try:
            if vth1:
                popt, _ = curve_fit(zcurve, PlsrDAC, scurve_data, p0=[repeat_command, mu_guess, -0.5], check_finite=False)

                logging.info('Fit-params-zcurve: %s %s %s ', str(popt[0]), str(popt[1]), str(popt[2]))
            else:
                popt, _ = curve_fit(scurve, PlsrDAC, scurve_data, p0=[repeat_command,
                                                                      mu_guess, sig_guess], sigma=data_errors, check_finite=False)
                if popt[1] < 0:
                    popt[1] = 0
                logging.info('Fit-params-scurve: %s %s %s ', str(popt[0]), str(popt[1]), str(popt[2]))
        except RuntimeError:  # fit failed
            popt = [repeat_command, mu_guess, sig_guess]
            logging.info('*****Fit did not work scurve: %s %s %s', str(popt[0]), str(popt[1]), str(popt[2]))

#     chi2 = np.sum((np.diff(scurve_data - scurve(PlsrDAC, *popt))**2) * repeat_command)
    chi2 = ss.chisquare(scurve_data, scurve(PlsrDAC, *popt))

    if popt[1] < 0:  # threshold < 0 rarely happens if fit does not work
        popt = p0

    # conditions for if fit fails but good data like zcurve fit for tdac
#     if vth1 == False and bad_fit == True:
#         if scurve_data[0] == 0 and scurve_data[-1] >= repeat_command:
#             for itt in range(len(scurve_data)):
#                 if scurve_data[itt] >= (repeat_command / 2):
#                     thresh1 = scurve_data[itt - 1]
#                     thresh2 = scurve_data[itt]
#                     thresh = (scurve_data[itt] / repeat_command)
#                     if abs(thresh1 - (repeat_command / 2)) < abs(thresh2 - (repeat_command / 2)):
#                         thresh = itt - 1
#                     else:
#                         thresh = itt
#                     break
#
#             logging.info('Fit failed on good data, params: %s %s %s', str(popt[0]), str(popt[1]), str(popt[2]))

    return [popt[0], popt[1], popt[2], chi2[0]]


def gauss(x_data, *parameters):
    """Gauss function"""
    A_gauss, mu_gauss, sigma_gauss = parameters
    return A_gauss * np.exp(-(x_data - mu_gauss)**2 / (2. * sigma_gauss**2))


def fit_gauss(x_data, y_data, params_guess=None):
    """Fit gauss"""
    # params_guess -> ndarray
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    y_maxima = x_data[np.where(y_data[:] == np.max(y_data))[0]]
    if not params_guess:
        params_guess = np.array([np.max(y_data), y_maxima[0], np.std(x_data)])  # np.mean(y_data)
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


# plt.tight_layout()
# plt.save_fig(, layout="tight")
