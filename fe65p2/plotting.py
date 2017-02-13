import numpy as np
from bokeh.charts import HeatMap, bins, output_file, show
from bokeh.models.layouts import Column, Row
from bokeh.palettes import RdYlGn6, RdYlGn9, BuPu9, Spectral11
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import tables as tb
import analysis as analysis
import yaml


def plot_timewalk(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        try:
            tdc_data = in_file_h5.root.tdc_data[:]
            td_threshold = in_file_h5.root.td_threshold[:]
            repeat_command_dict = in_file_h5.root.tdc_data.attrs.repeat_command
            repeat_command= repeat_command_dict['repeat_command']
        except RuntimeError:
            logging.info('tdc_data not present in file')
            return
        time_thresh = td_threshold['td_threshold']
        expfit0 = td_threshold['expfit0']
        expfit1 = td_threshold['expfit1']
        expfit2 = td_threshold['expfit2']
        expfit3 = td_threshold['expfit3']

        tot = tdc_data['tot_ns']
        tot_err = tdc_data['err_tot_ns']
        delay = tdc_data['delay_ns']
        delay_err = tdc_data['err_delay_ns']
        pixel_no = tdc_data['pixel_no']
        pulse = tdc_data['charge']
        hits = tdc_data['hits']
        pix, stop = np.unique(pixel_no, return_index=True)
        stop = np.sort(stop)
        single_scan = figure(title="Single pixel scan ")
        single_scan.xaxis.axis_label = "Charge (electrons)"
        single_scan.yaxis.axis_label = "Hits"

        TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"
        p1 = figure(title="Timewalk", tools=TOOLS)
        p1.xaxis.axis_label = "Charge (electrons)"
        p1.yaxis.axis_label = "Delay (ns)"
        p2 = figure(title="TOT linearity", tools=TOOLS)
        p2.xaxis.axis_label = "Charge (electrons)"
        p2.yaxis.axis_label = "TOT (ns)"

        stop = list(stop)
        stop.append(len(tot))
        for i in range(len(stop) - 1):
            s1 = int(stop[i])
            s2 = int(stop[i + 1])
            if time_thresh[i] == 0:
                continue
            single_scan.diamond(x=pulse[s1:s2], y=hits[s1:s2], size=5, color=Spectral11[i - 1], line_width=2)
            A, mu, sigma = analysis.fit_scurve(hits[s1:s2], pulse[s1:s2], repeat_command)
            for values in range(s1, s2):
                if pulse[values] >= 5 / 4 * mu:
                    s1 = values
                    break
            p1.circle(pulse[s1:s2], delay[s1:s2], legend=str("pixel " + str(pixel_no[s1])), color=Spectral11[i - 1],
                      size=8)
            if len(time_thresh) != 0:
                p1.line(pulse[s1:s2], analysis.exp(pulse[s1:s2], expfit0[i], expfit1[i], expfit2[i], expfit3[i]),
                        color=Spectral11[i - 1])
                p1.asterisk(time_thresh[i], np.min(delay[s1:s2]) + 25, color=Spectral11[i - 1], size=20,
                            legend="Time dependent Threshold: " + str(round(time_thresh[i], 2)))
            else:
                logging.info('No fit possible only Data plotted')
            err_x1 = [(pulse[s], pulse[s]) for s in range(s1, s2)]
            err_y1 = [[float(delay[s] - delay_err[s]), float(delay[s] + delay_err[s])] for s in range(s1, s2)]
            p1.multi_line(err_x1, err_y1, color=Spectral11[i - 1], line_width=2)

            p2.circle(pulse[s1:s2], tot[s1:s2], legend=str("pixel " + str(pix[i])), color=Spectral11[i - 1], size=8)
            err_x1 = [(pulse[s], pulse[s]) for s in range(s1, s2)]
            err_y1 = [[float(tot[s] - tot_err[s]), float(tot[s] + tot_err[s])] for s in range(s1, s2)]
            p2.multi_line(err_x1, err_y1, color=Spectral11[i - 1], line_width=2)

        return p1, p2, single_scan


def plot_status(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        kwargs = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
        power_status = yaml.load(in_file_h5.root.meta_data.attrs.power_status)

    data = {'nx': [], 'value': []};

    data['nx'].append('Scan Parameters:')
    data['value'].append('')

    for key, value in kwargs.iteritems():
        data['nx'].append(key)
        data['value'].append(value)

    data['nx'].append('DACs settings:')
    data['value'].append('')

    for key, value in dac_status.iteritems():
        data['nx'].append(key)
        data['value'].append(value)

    data['nx'].append('Power Status:')
    data['value'].append('')

    for key, value in power_status.iteritems():
        data['nx'].append(key)
        data['value'].append("{:.2f}".format(value))

    source = ColumnDataSource(data)

    columns = [
        TableColumn(field="nx", title="Name"),
        TableColumn(field="value", title="Value"),
    ]

    data_table = DataTable(source=source, columns=columns, width=300)

    return data_table


def plot_occupancy(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]

        hits = hit_data['col'].astype(np.uint16)
        hits = hits * 64
        hits = hits + hit_data['row']
        value = np.bincount(hits)
        value = np.pad(value, (0, 64 * 64 - value.shape[0]), 'constant')

        indices = np.indices(value.shape)
        col = indices[0] / 64
        row = indices[0] % 64

        data = {'column': col,
                'row': row,
                'value': value
                }

        hm = HeatMap(data, x='column', y='row', values='value', legend='top_right', title='Occupancy',
                     palette=RdYlGn6[::-1], stat=None)

    return hm, value.reshape((64, 64))


def plot_tot_dist(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]

        tot_count = np.bincount(hit_data['tot'])
        tot_count = np.pad(tot_count, (0, 15 - tot_count.shape[0]), 'constant')
        tot_plot = figure(title="ToT Distribution")
        tot_plot.quad(top=tot_count, bottom=0, left=np.arange(-0.5, 14.5, 1), right=np.arange(0.5, 15.5, 1))

    return tot_plot, tot_count


def plot_lv1id_dist(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]

        lv1id_count = np.bincount(hit_data['lv1id'])
        lv1id_count = np.pad(lv1id_count, (0, 16 - lv1id_count.shape[0]), 'constant')
        lv1id_plot = figure(title="lv1id Distribution")
        lv1id_plot.quad(top=lv1id_count, bottom=0, left=np.arange(-0.5, 15.5, 1), right=np.arange(0.5, 16.5, 1))

    return lv1id_plot, lv1id_count


def t_dac_plot(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        t_dac = in_file_h5.root.scan_results.tdac_mask[:]
        en_mask = in_file_h5.root.scan_results.en_mask[:]
    shape = en_mask.shape
    ges = 1
    for i in range(2):
        ges = ges * shape[i]
    T_Dac_pure = ()
    t_dac = t_dac.reshape(ges)
    en_mask = en_mask.reshape(ges)
    for i in range(ges):
        if (str(en_mask[i]) == 'True'):
            T_Dac_pure = np.append(T_Dac_pure, t_dac[i])
    T_Dac_pure = T_Dac_pure.astype(int)
    T_Dac_hist_y = np.bincount(T_Dac_pure)
    T_Dac_hist_x = np.arange(0, T_Dac_hist_y.size, 1)
    gauss_TDAC = analysis.fit_gauss(T_Dac_hist_x, T_Dac_hist_y)
    plt_t_dac = figure(title='T-Dac-distribution ', x_axis_label="T-Dac", y_axis_label="#Pixel")
    plt_t_dac.quad(top=T_Dac_hist_y, bottom=0, left=T_Dac_hist_x[:-1], right=T_Dac_hist_x[1:], fill_color="#036564",
                   line_color="#033649", legend="#{0:d}  mean={1:.2f}  std={2:.2f}".format(int(np.sum(T_Dac_hist_y[:])), gauss_TDAC[1] , gauss_TDAC[2] ))
    plt_t_dac.line(T_Dac_hist_x, analysis.gauss(T_Dac_hist_x, gauss_TDAC[0], gauss_TDAC[1], gauss_TDAC[2]),
                   line_color="#D95B43", line_width=8, alpha=0.7)
    return plt_t_dac


def scan_pix_hist(h5_file_name, scurve_sel_pix=200):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        hit_data = in_file_h5.root.hit_data[:]
        en_mask = in_file_h5.root.scan_results.en_mask[:]
        Noise_gauss = in_file_h5.root.Noise_results.Noise_pure.attrs.fitdata_noise
        Noise_pure = in_file_h5.root.Noise_results.Noise_pure[:]
        Thresh_gauss = in_file_h5.root.Thresh_results.Threshold_pure.attrs.fitdata_thresh
        Threshold_pure = in_file_h5.root.Thresh_results.Threshold_pure[:]
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        scan_range = scan_args['scan_range']
        scan_range_inx = np.arange(scan_range[0], scan_range[1], scan_range[2])

        # np.set_printoptions(threshold=np.nan)
        param = np.unique(meta_data['scan_param_id'])
        ret = []
        for i in param:
            wh = np.where(hit_data['scan_param_id'] == i)  # this can be faster and multi threaded
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
        repeat_command = max(ret[-3])
        shape = en_mask.shape
        ges = 1
        for i in range(2):
            ges = ges * shape[i]
        ret_pure = ()
        en_mask = en_mask.reshape(ges)
        for n in range(param[-1]):
            ret_pure1 = ()
            for i in range(ges):
                if (str(en_mask[i]) == 'True'):
                    ret_pure1 = np.append(ret_pure1, ret[n][i])
            if n == 0:
                ret_pure = ret_pure1
                continue
            ret_pure = np.vstack((ret_pure, ret_pure1))

        ret_pure = ret_pure.astype(int)
        s_hist = np.swapaxes(ret_pure, 0, 1)

        pix_scan_hist = np.empty((s_hist.shape[1], repeat_command + 10))
        for param in range(s_hist.shape[1]):
            h_count = np.bincount(s_hist[:, param])
            h_count = h_count[:repeat_command + 10]
            pix_scan_hist[param] = np.pad(h_count, (0, (repeat_command + 10) - h_count.shape[0]), 'constant')

        log_hist = np.log10(pix_scan_hist)
        log_hist[~np.isfinite(log_hist)] = 0
        data = {
            'scan_param': np.ravel(np.indices(pix_scan_hist.shape)[0]),
            'count': np.ravel(np.indices(pix_scan_hist.shape)[1]),
            'value': np.ravel(log_hist)
        }

        x = scan_range_inx
        px = scurve_sel_pix  # 1110 #1539
        single_scan = figure(title="Single pixel scan " + str(px), x_axis_label="Injection[V]", y_axis_label="Hits")
        single_scan.diamond(x=x, y=s_hist[px], size=5, color="#1C9099", line_width=2)
        yf = analysis.scurve(x, 100, Threshold_pure[px], Noise_pure[px])
        single_scan.cross(x=x, y=yf, size=5, color="#E6550D", line_width=2)

        hist, edges = np.histogram(Threshold_pure, density=False, bins=50)

        hm1 = HeatMap(data, x='scan_param', y='count', values='value', title='Threshold Heatmap',
                      palette=Spectral11[::-1], stat=None, plot_width=1000)  # , height=4100)
        hm1.extra_x_ranges = {
            "e": Range1d(start=edges[0] * 1000 * analysis.cap_fac(), end=edges[-1] * 1000 * analysis.cap_fac())}

        hm_th = figure(title="Threshold", x_axis_label="pixel #", y_axis_label="threshold [V]",
                       y_range=(scan_range_inx[0], scan_range_inx[-1]), plot_width=1000)
        hm_th.diamond(y=Threshold_pure, x=range(64 * 64), size=1, color="#1C9099", line_width=2)
        hm_th.extra_y_ranges = {"e": Range1d(start=scan_range_inx[0] * 1000 * analysis.cap_fac(),
                                             end=scan_range_inx[-1] * 1000 * analysis.cap_fac())}
        hm_th.add_layout(LinearAxis(y_range_name="e"), 'right')
        plt_th_dist = figure(title='Threshold Distribution ', x_axis_label="threshold [V]", y_axis_label="#Pixel",
                             y_range=(0, 1.1 * np.max(hist[1:])))
        plt_th_dist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#036564",
                         line_color="#033649",
                         legend="# {0:d}  mean={1:.2f}  std={2:.2f}".format(int(np.sum(hist[:])), round(
                             Thresh_gauss['mu'] * 1000 * analysis.cap_fac(), 4),
                                                                            round(Thresh_gauss[
                                                                                      'sigma'] * 1000 * analysis.cap_fac(),
                                                                                  4)))
        plt_th_dist.extra_x_ranges = {"e": Range1d(start=edges[0] * 1000 * analysis.cap_fac(),
                                                   end=edges[-1] * 1000 * analysis.cap_fac())}  # better 7.4?
        plt_th_dist.add_layout(LinearAxis(x_range_name="e"), 'above')
        plt_th_dist.line(np.arange(edges[1], edges[-1], 0.0001),
                         analysis.gauss(np.arange(edges[1], edges[-1], 0.0001), Thresh_gauss['height'],
                                        Thresh_gauss['mu'], Thresh_gauss['sigma']), line_color="#D95B43", line_width=8,
                         alpha=0.7)

        hist, edges = np.histogram(Noise_pure, density=False, bins=50)
        hm_noise = figure(title="Noise", x_axis_label="pixel #", y_axis_label="noise [V]", y_range=(0, edges[-1]),
                          plot_width=1000)
        hm_noise.diamond(y=Noise_pure, x=range(64 * 64), size=2, color="#1C9099", line_width=2)
        hm_noise.extra_y_ranges = {"e": Range1d(start=0, end=edges[-1] * 1000 * analysis.cap_fac())}  # default 7.6
        hm_noise.add_layout(LinearAxis(y_range_name="e"), 'right')

        plt_noise_dist = figure(title='Noise Distribution ', x_axis_label="noise [V]", y_axis_label="#Pixel",
                                y_range=(0, 1.1 * np.max(hist[1:])))
        plt_noise_dist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#036564",
                            line_color="#033649",
                            legend="# {0:d}  mean={1:.2f}  std={2:.2f}".format(int(np.sum(hist[:])), round(
                                Noise_gauss['mu'] * 1000 * analysis.cap_fac(), 4), round(
                                Noise_gauss['sigma'] * 1000 * analysis.cap_fac(), 4)))
        plt_noise_dist.extra_x_ranges = {
            "e": Range1d(start=edges[0] * 1000 * analysis.cap_fac(), end=edges[-1] * 1000 * analysis.cap_fac())}
        plt_noise_dist.add_layout(LinearAxis(x_range_name="e"), 'above')
        plt_noise_dist.line(np.arange(edges[0], edges[-1], 0.0001),
                            analysis.gauss(np.arange(edges[0], edges[-1], 0.0001), Noise_gauss['height'],
                                           Noise_gauss['mu'], Noise_gauss['sigma']), line_color="#D95B43", line_width=8,
                            alpha=0.7)

        return Column(Row(hm_th, plt_th_dist), Row(hm_noise, plt_noise_dist), Row(hm1, single_scan)), s_hist


if __name__ == "__main__":
    pass