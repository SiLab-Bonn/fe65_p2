'''
Author: Daniel Coquelin
Created: 23 Okt 2017
Purpose:
import data from scans for fe65p2 and plot data
'''
import numpy as np
import matplotlib.pyplot as plt
import logging
from numpy import uint16
from docutils.parsers.rst.directives.tables import align

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import tables as tb
import analysis as analysis
import yaml
import matplotlib as mpl
import matplotlib.mlab as mlab
import analysis as analysis
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from scipy import optimize
from math import ceil


# timewalk plotting not completed yet as i do not fully understand the
# plotting functions


def plot_timewalk(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        try:
            tdc_data = in_file_h5.root.tdc_data[:]
            td_threshold = in_file_h5.root.td_threshold[:]
            repeat_command_dict = in_file_h5.root.tdc_data.attrs.repeat_command
            repeat_command = repeat_command_dict['repeat_command']
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
        # stop tells the code where a set of tests on one pixel begin and end
        stop = np.sort(stop)
        stop = list(stop)
        stop.append(len(tot))

        fig1 = Figure()  # timewalk
        _ = FigureCanvas(fig1)
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        # n, bins, patchs = ax.hist(T_Dac_pure, 18)
        ax1.set_title('Timewalk')
        ax1.set_xlabel('Charge (e-)')
        ax1.set_ylabel('Delay (ns)')
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(4))

        fig2 = Figure()  # timewalk
        _ = FigureCanvas(fig2)
        fig2.clear()
        ax2 = fig2.add_subplot(111)
        # n, bins, patchs = ax.hist(T_Dac_pure, 18)
        ax2.set_title('TOT Linearity')
        ax2.set_xlabel('Charge (e-)')
        ax2.set_ylabel('TOT (ns)')
        ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(4))

        fig3 = Figure()  # timewalk
        _ = FigureCanvas(fig3)
        fig3.clear()
        ax3 = fig3.add_subplot(111)
        # n, bins, patchs = ax.hist(T_Dac_pure, 18)
        ax3.set_title('Single Pixel Scan')
        ax3.set_xlabel('Charge (e-)')
        ax3.set_ylabel('Hits')
        ax3.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax3.yaxis.set_minor_locator(AutoMinorLocator(4))
        colors = iter(cm.rainbow(np.linspace(0, 1, len(stop) * 3)))
        for i in range(len(stop) - 1):
            s1 = int(stop[i])
            s2 = int(stop[i + 1])
            if time_thresh[i] == 0:
                continue
            ax3.scatter(pulse[s1:s2], hits[s1:s2], color=next(
                colors), label=str("pixel " + str(pixel_no[s1])))
            A, mu, sigma, chi2 = analysis.fit_scurve(
                hits[s1:s2], pulse[s1:s2], repeat_command)
            for values in range(s1, s2):
                if pulse[values] >= 5 / 4 * mu:
                    s1 = values
                    break
            if len(time_thresh) != 0:
                lnspc = np.linspace(
                    time_thresh[i] - 10, pulse[s1:s2].max(), 200)
                expline = expfit3[i] + expfit2[i] * \
                    np.exp(-(lnspc + expfit1[i]) / expfit0[i])
                ax1.plot(lnspc, expline, label=str(
                    "exp fit for pixel " + str(pixel_no[s1])))
                ax1.scatter(time_thresh[i], np.min(delay[s1:s2]) + 25, color=next(
                    colors), label="Time dependent Threshold: " + str(round(time_thresh[i], 2)))
            else:
                logging.info('No fit possible, no fits plotted')

            err_x1 = [(pulse[s], pulse[s]) for s in range(s1, s2)]
            err_y1 = [[float(delay[s] - delay_err[s]),
                       float(delay[s] + delay_err[s])] for s in range(s1, s2)]
            # ax1.scatter(err_x1, err_y1, color=next(colors), label='errors')
            # TODO: make these errorbars work... err must be [ scalar | N, Nx1
            # or 2xN array-like ]
            ax1.errorbar(pulse[s1:s2], delay[s1:s2], xerr=err_x1, yerr=err_y1,  color=next(
                colors), label=str("pixel " + str(pixel_no[s1])))

            ax2.scatter(pulse[s1:s2], tot[s1:s2], color=next(
                colors), label=str("pixel " + str(pix[i])))
            err_x1 = [(pulse[s], pulse[s]) for s in range(s1, s2)]
            err_y1 = [[float(tot[s] - tot_err[s]), float(tot[s] + tot_err[s])]
                      for s in range(s1, s2)]
            print err_x1
            print err_y1
            # ax2.scatter(err_x1, err_y1, color=next(colors))

            ax1.legend(loc='upper right')
            ax2.legend(loc='upper left')

            print 'passed timewalk, single pix scan, and TOT'
            return fig1, fig2, fig3


def plot_status(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        kwargs = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
        power_status = yaml.load(in_file_h5.root.meta_data.attrs.power_status)
    '''
    def write_parameters(scan_id, run_name, attributes, dacs, sw_ver, filename=None):
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.axis('off')

    ax.text(0.01, 1, 'This is a bdaq53 %s with run name\n%s\nusing parameters:' % (scan_id, run_name), fontsize=10)
    ax.text(0.9, -0.11, 'Software version: %s' % (sw_ver), fontsize=3)
   
    tb_dict = OrderedDict(sorted(dacs.items()))
    attr = OrderedDict(sorted(attributes.items()))
    for key, value in attr.items():
        tb_dict[key] = str(value) + '*'
   
    tb_list = []
    for i in range(0,len(tb_dict.keys()),2):
        try:
            key1 = tb_dict.keys()[i]
            key2 = tb_dict.keys()[i+1]
            value1 = tb_dict[key1]
            value2 = tb_dict[key2]
            tb_list.append([key1, value1, '', key2, value2])
        except:
            pass
       
    widths = [0.3,0.1, 0.1, 0.3,0.1]
    labels = ['Parameter', 'Value', '', 'Parameter', 'Value']
    table = ax.table(cellText=tb_list, colWidths=widths, colLabels=labels, cellLoc='left', loc='center')
    table.scale(0.8, 0.8)

    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:
            cell.set_color('grey')
        if col == 2:
            cell.set_color('white')

    if not filename:
        fig.show()
    elif isinstance(filename, PdfPages):
        filename.savefig(fig)
    else:
        fig.savefig(filename)
    '''
    data = {'nx': [], 'value': []}

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

    tab_labels = data['nx']
    labelsArray = np.asarray(tab_labels)
    valArray = np.empty_like(labelsArray)
    labelsArray = labelsArray.transpose()

    tab_values = data['value']

    for i in range(0, len(tab_values)):
        valArray[i] = tab_labels[i]
    # valArray=np.asarray(tab_values)
    #

    fig = Figure()
    _ = FigureCanvas(fig)
    fig.clear()
    ax = fig.add_subplot(111)
    # print tab_values.shape
    # print valArray.shape
    # print type(tab_values)
    # print tab_values
    # print type(tab_labels)
    # print type(tab_values)
    ax.table(cellText=[tab_labels, tab_values], loc='center')
    ax.axis('off')
    # ax.set_title('ToT Dist.')
    print 'passed status'
    # need to fix below, data tabled with Bokeh

    # source = ColumnDataSource(data)

    # columns = [
    #    TableColumn(field="nx", title="Name"),
    #    TableColumn(field="value", title="Value"),
    #]

    # data_table = DataTable(source=source, columns=columns, width=300)

    return fig


def combi_calib_plots(h5_file):
    # plots needed for the combi:
    # difference between fitted line and average tdc for source
    # new fits with chisq for each line
    # full plot of random pixels for the whole thing

    pass


def tdc_src_spectrum(h5_file):
    with tb.open_file(h5_file, 'r+') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        raw_data = in_file_h5.root.raw_data[:]
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
    tdc_data = raw_data & 0x0FFF
    tdc_delay = (raw_data & 0x0FF00000) >> 20

    # fig1 -> spectrum of data (fit to come from analysis later
    fig1 = Figure()
    _ = FigureCanvas(fig1)
    ax1 = fig1.add_subplot(111)
    bar_data, bins = np.histogram(tdc_data[tdc_delay < 254], (max(tdc_data[tdc_delay < 254]) - min(tdc_data[tdc_delay < 254])),
                                  range=(min(tdc_data[tdc_delay < 254]), max(tdc_data[tdc_delay < 254])))
    bin_left = bins[:-1]
    ax1.bar(left=bin_left, height=bar_data, width=np.diff(bin_left)[0], align="edge")
    ax1.set_title("Spectrum of Source")
    ax1.grid()
    fig1.tight_layout()
    print"passed spectrum"
    return fig1


def pix_inj_calib_lines(h5_file):
    with tb.open_file(h5_file, 'r+') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        tdc_data = in_file_h5.root.tdc_data[:]
        fit_data = in_file_h5.root.fit_data[:]
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
    scan_range = scan_args['scan_range']
    # want to make lines out of the fit data, then put them into a 2d plot

    line_space = np.linspace(0, scan_range[-1], 300)
#     xlabels = np.linspace(0, scan_range[-1], 6)
    # x -> line_space is the # of electrons
    # y -> tdc bins
    max_tdc = 300
    pixel_range = scan_args['pixel_range']
    for pix in range(pixel_range[0], pixel_range[1]):
        #         print fit_data[fit_data['pix_num'] == pix]['m']
        lines_array_hold = analysis.linear(line_space, fit_data[fit_data['pix_num'] == pix]['m'], fit_data[fit_data['pix_num'] == pix]['b'])
        max_tdc_hold = max(lines_array_hold)
        if pix == pixel_range[0]:
            lines_array = lines_array_hold
        if pix != pixel_range[0]:
            lines_array = np.concatenate((lines_array, lines_array_hold), axis=0)
        if max_tdc_hold > max_tdc:
            max_tdc = max_tdc_hold

    lines_array = lines_array.reshape(len(fit_data['pix_num']), line_space.shape[0])
    lines_array = lines_array.astype(int)
#     print max_tdc
    for i in range(line_space.shape[0]):
        # count the bin fillings for each element of the line_space
        x_per_inj, _ = np.histogram(lines_array[:, i], bins=300, range=(0, np.round(max_tdc + 5.)))
#         print x_per_inj
        if i == 0:
            weight = x_per_inj
        if i != 0:
            weight = np.concatenate((weight, x_per_inj), axis=0)
    weight = weight.reshape(300, line_space.shape[0])
    weight = np.swapaxes(weight, 1, 0)

#     print lines_array
    fig = Figure()
    _ = FigureCanvas(fig)
    fig.clear()
    cmap = plt.cm.viridis
    cmap.set_under(color='white')
    ax = fig.add_subplot(111)
    h1 = ax.imshow(weight, origin='lower', vmin=0.00000001,
                   extent=[line_space[0], line_space[-1], 0, 300])
    ax.set_aspect(30)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
#     ax.set_xticklabels(np.round(xlabels, 0))
    ax.set_xlabel("Electrons")
    ax.set_ylabel("TDC Channel")
    ax.set_title("Overlay of Fitted Lines when 100% data collected")
    ax.grid()
    fig.tight_layout()
    print "finished fitted lines plot"

    fig2 = Figure()
    _ = FigureCanvas(fig2)
    fig2.clear()
    ax2 = fig2.add_subplot(111)
    bar_data, bins = np.histogram(fit_data['chisq'], 20, range=(min(fit_data['chisq']), max(fit_data['chisq'])))
    bin_left = bins[:-1]
    ax2.bar(left=bin_left, height=bar_data, width=np.diff(bin_left)[0], align="edge")
    ax2.set_title("Chi Squared values of fits (elecs>=3000)")
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.grid()
#     fig2.tight_layout()
    print "passing chisq and fitted lines"
    return fig, fig2


def pix_inj_calib_tdc_v_elec(h5_file):
    with tb.open_file(h5_file, 'r+') as in_file_h5:
        raw_data = in_file_h5.root.raw_data[:]
        tdc_data = in_file_h5.root.tdc_data[:]
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
    scan_range = scan_args['scan_range']
    num_pix = scan_args['pixel_stop']
    pixel_range = scan_args['pixel_range']
    repeats = scan_args['repeat_command']
#     inj_elecs = scan_args['scan_range']
#     inj_elecs = np.arange(inj_elecs[0],inj_elecs[1],inj_elecs[2])
    fit_list = []
    if pixel_range[1] - pixel_range[0] > 20:
        pixel_set = np.random.choice(np.arange(pixel_range[0], pixel_range[1]), size=20)
    else:
        pixel_set = range(pixel_range[0], pixel_range[1])
    fig1 = Figure()
    _ = FigureCanvas(fig1)
    fig1.clear()
    ax = fig1.add_subplot(111)
#     ax.color_cycle
#     color = iter(cm.rainbow(np.linspace(0, 1, 10)))
    print "enter pixel_set loop"
    for pix in pixel_set:
        # need: inj_elecs, mu_tdc, sigma_tdc
        #          y       x        x_err
        # need fit as well...hopefully linear or at least will assume it is for the moment
        # TODO: what is best fit?
        tdc_data_hold = tdc_data[tdc_data['pix_num'] == pix]
        y = tdc_data_hold['mu_tdc'][:]
        yerr = tdc_data_hold['sigma_tdc'][:]
        x = tdc_data_hold['elecs'][:]
        ax.errorbar(x, y, yerr=yerr, xerr=(1. / np.sqrt(repeats)), fmt='o', label=('Pixel %s' % str(pix)), markersize=0.1)
    # ax.legend()
    ax.set_xlabel("Injected Electrons")
    ax.set_ylabel("TDC Channel")
    ax.set_title("Spectrum of 20 Random Pixels")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid()
    print "finished tdc full 20"

    fig2 = Figure()
    _ = FigureCanvas(fig2)
    fig2.clear()
    ax2 = fig2.add_subplot(111)
    cmap = plt.cm.viridis
    cmap.set_under(color='white')
#     print tdc_data['mu_tdc'][tdc_data['inj_elecs'] <= 3000]
    h = ax2.hist2d(tdc_data['elecs'][tdc_data['elecs'] <= 3000],
                   tdc_data['mu_tdc'][tdc_data['elecs'] <= 3000],
                   bins=((max(tdc_data['elecs'][(tdc_data['elecs'] <= 3000)]) - min(tdc_data['elecs'][(tdc_data['elecs'] <= 3000)])) / 50,
                         max(tdc_data['mu_tdc'][tdc_data['elecs'] <= 3000])),
                   range=((min(tdc_data['elecs'][(tdc_data['elecs'] <= 3000)]), max(tdc_data['elecs'][(tdc_data['elecs'] <= 3000)]) + 50),
                          (0, max(tdc_data['mu_tdc'][tdc_data['elecs'] <= 3000]))),
                   cmap=cmap, vmin=0.00000001)
    fig2.colorbar(h[3], ax=ax2, pad=0.01)
    ax2.set_xlabel("Injected Electrons")
    ax2.set_ylabel("TDC Channel")
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.grid()
#     ax2.set_title("")
    print "passing tdc full and zoom"

    fig3 = Figure()
    _ = FigureCanvas(fig3)
    fig3.clear()
    ax3 = fig3.add_subplot(111)
    cmap1 = plt.cm.viridis
    cmap1.set_under(color='white')
#     print tdc_data['mu_tdc'][tdc_data['inj_elecs'] <= 3000]
    h = ax3.hist2d(tdc_data['elecs'], tdc_data['mu_tdc'], bins=(
        (max(tdc_data['elecs'] - min(tdc_data['elecs'])) / 100, max(tdc_data['mu_tdc']))), cmap=cmap1, vmin=0.00000001)
    fig3.colorbar(h[3], ax=ax3, pad=0.01)
    ax3.set_xlabel("Injected Electrons")
    ax3.set_ylabel("TDC Channel")
    ax3.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax3.grid()

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    return fig1, fig2, fig3


def pixel_inj_calib_delay_vs_data(h5_file):
    with tb.open_file(h5_file, 'r+') as in_file:
        raw_data = in_file.root.raw_data[:]
        tdc_data = raw_data & 0x0FFF  # only want last 12 bit
        tdc_delay = (raw_data & 0x0FF00000) >> 20
        fig = Figure()
        _ = FigureCanvas(fig)
        fig.clear()
        cmap = plt.cm.viridis
        cmap.set_under(color='white')
        ax = fig.add_subplot(111)
        h = ax.hist2d(tdc_data, tdc_delay, bins=((max(tdc_data) - min(tdc_data)) / 2,
                                                 max(tdc_delay) - min(tdc_delay)), cmap=cmap, vmin=0.00000001)
        fig.colorbar(h[3], ax=ax, pad=0.01)
        ax.set_title('TDC Data vs TDC Delay')
        ax.set_xlabel('TDC Data')
        ax.set_ylabel('TDC Delay')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid()
        fig.tight_layout()
        print "passing tdc vs delay"
        return fig


def plot_occupancy(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]

        hitsX_col = hit_data['col']
        hitsY_row = hit_data['row']

        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_main = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width - 0.105, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        fig = Figure(figsize=(8, 8))
        _ = FigureCanvas(fig)
        fig.clear()
        fig.patch.set_facecolor('white')
        cmap = plt.cm.viridis
        cmap.set_under(color='white')
        # ax = fig.add_subplot(111)
        ax_main = fig.add_axes(rect_main)
        ax_X = fig.add_axes(rect_histx)
        ax_Y = fig.add_axes(rect_histy)
        ax_X.xaxis.set_major_formatter(NullFormatter())
        ax_Y.yaxis.set_major_formatter(NullFormatter())

        ax_X.set_title('Occupancy' + str(h5_file_name))
        ax_main.set_xlabel('column')
        ax_main.set_ylabel('row')
        h = ax_main.hist2d(hitsX_col, hitsY_row, bins=(max(hitsX_col) + 1, max(hitsY_row) + 1),
                           range=((min(hitsX_col) - 0.5, max(hitsX_col) + 0.5), (min(hitsY_row) - 0.5, (max(hitsY_row) + 0.5))),
                           cmap=cmap, vmin=0.00000001)
        ax_main.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_main.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax_X.hist(hitsX_col, bins=65, range=(-0.5, 64.5))
        ax_Y.hist(hitsY_row, bins=65, range=(-0.5, 64.5),
                  orientation='horizontal')
        ax_X.set_xlim(ax_main.get_xlim())
        ax_Y.set_ylim(ax_main.get_ylim())
        fig.colorbar(h[3], ax=ax_main, pad=0.01)

        print 'passed occ plot'
#         fig.tight_layout()
        return fig


def thresh_pix_heatmap(h5_file_name):
    # output two heatmaps with projections on the sides
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        Noise = in_file_h5.root.Noise_results.Noise
        Thresh = in_file_h5.root.Thresh_results.Threshold
        Thresh = np.swapaxes(Thresh, 0, 1)
        Noise = np.swapaxes(Noise, 0, 1)
        fig1 = Figure()
        _ = FigureCanvas(fig1)
        fig1.clear()
        fig1.patch.set_facecolor("white")
        ax1_main = fig1.add_subplot(111)
        ax1_main.set_title('Thresholds for each pixel')
        ax1_main.set_xlabel('column')
        ax1_main.set_ylabel('row')
        # ax1_main.xaxis.set_label_position('bottom')
        # ax1_main.xaxis.tick_bottom()
        cmap = plt.cm.viridis
        cmap.set_under(color='white')
        h1 = ax1_main.imshow(Thresh, origin='lower', interpolation='none', cmap=cmap, vmin=0.00000001)
        # TODO: show color map
        cbar = fig1.colorbar(h1)
        fig2 = Figure()
        _ = FigureCanvas(fig2)
        fig2.clear()
        fig2.patch.set_facecolor("white")
        ax2_main = fig2.add_subplot(111)
        ax2_main.set_title('Noise levels for each pixel')
        ax2_main.set_xlabel('column')
        ax2_main.set_ylabel('row')
        # ax2_main.xaxis.set_label_position('bottom')
        # ax2_main.xaxis.set_ticks_position('bottom')
        cmap = plt.cm.viridis
        cmap.set_under(color='white')
        h1 = ax2_main.imshow(Noise, origin='lower left', interpolation='none', cmap=cmap, vmin=0.00000001)
        cbar = fig2.colorbar(h1)

        print 'passing threshold and noise heatmaps'
        fig1.tight_layout()
        fig2.tight_layout()
        return fig1, fig2


def tdac_heatmap(h5_file_name, en_mask_in=None, tdac_mask_in=None):
    if h5_file_name:
        with tb.open_file(h5_file_name, 'r') as in_file_h5:
            tdac = in_file_h5.root.analysis_results.tdac_mask[:]
            en_mask = in_file_h5.root.analysis_results.en_mask[:]
    else:
        tdac = tdac_mask_in
        en_mask = en_mask_in

    tdac[en_mask == False] = -1

    fig1 = Figure()
    _ = FigureCanvas(fig1)
    fig1.clear()
    ax1_main = fig1.add_subplot(111)
    ax1_main.set_title('TDAC for each pixel')
    ax1_main.set_xlabel('column')
    ax1_main.set_ylabel('row')
    cmap = plt.cm.viridis
    cmap.set_under(color='white')
    h1 = ax1_main.imshow(tdac, origin='lower', interpolation='none', cmap=cmap, vmin=-0.0001)
    cbar = fig1.colorbar(h1)

    fig2 = Figure()
    _ = FigureCanvas(fig2)
    fig2.clear()
    ax2 = fig2.add_subplot(111)
    print 'bad pixels: ', len(tdac[tdac == -1]), ' tdac = 31 or 0: ', len(tdac[(tdac == 31) | (tdac == 0)])
#     print (tdac.reshape(64 * 64)).astype(int)
#     tdac_counts = np.bincount((tdac.reshape(64 * 64)).astype(int))
#     print np.arange(len(tdac_counts))
#     print tdac_counts
    ax2.set_title('tdac histogram')
    for flav in range(8):
        tdac_per_flav = tdac[:, flav:(flav + 8)]
        bar_data = 0
        print "finished flavor ", flav
        bar_data, bins = np.histogram(tdac_per_flav, 34, range=(-1, 32))
        bin_left = bins[:-1] + (0.1 * flav)
        ax2.bar(left=bin_left, height=bar_data, width=0.1, label=('Flavor %s' % str(flav + 1)), align="edge")
    ax2.legend()
#         ax2.hist(tdac_per_flav, bins=34, range=(-1.5, 32.5), rwidth=1.5)
    fig1.tight_layout()
    fig2.tight_layout()
    return fig1, fig2


def plot_tot_dist(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]
        hit_data_tot = hit_data['tot']

        fig = Figure()
        _ = FigureCanvas(fig)
        fig.clear()
        ax = fig.add_subplot(111)
        ax.hist(hit_data_tot, bins=np.arange(min(hit_data_tot) - 0.5, max(hit_data_tot) + 1.5, 1))
        ax.set_title('ToT Dist.')
        ax.set_xlabel('units of 25 ns')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        print 'average tot: ', np.mean(hit_data_tot), ' max tot: ', max(hit_data_tot), " std tot: ", hit_data_tot.std()
        print 'passed tot_dist'
        ax.grid()
        fig.tight_layout()
        return fig


def plot_lv1id_dist(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]
        hit_data_lv1id = hit_data['lv1id']

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
#         ax.hist(hit_data_lv1id, 20, range=(-1.5, 17.5))
        bar_data, bins = np.histogram(hit_data_lv1id, max(hit_data_lv1id) + 1, range=(0, max(hit_data_lv1id) + 1))
        print bar_data
        bin_left = bins[:-1]
        ax.bar(left=bin_left, height=bar_data, width=np.diff(bin_left)[0], align="edge")
        ax.set_title('lv1id Dist')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        print 'passed lv1id average: %d' % hit_data_lv1id.mean()
        ax.grid()
        fig.tight_layout()
        return fig


def tdac_plot_for_tdac_scan(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        t_dac = in_file_h5.root.analysis_results.tdac_mask[:]
        en_mask_after = in_file_h5.root.analysis_results.en_mask[:]
        en_mask = in_file_h5.root.scan_results.en_mask[:]

    shape = en_mask.shape
    ges = 1
    for i in range(2):
        ges = ges * shape[i]
    T_Dac_pure = ()

    t_dac = t_dac.reshape(ges)
    en_mask = en_mask.reshape(ges)
    en_mask_after = en_mask_after.reshape(ges)
    for i in range(ges):
        if (str(en_mask_after[i]) == 'True') and (str(en_mask[i]) == 'True'):
            #         if en_mask[i] == 1.0:
            T_Dac_pure = np.append(T_Dac_pure, t_dac[i])
    print "total length of tested columns minus disabled pixels", len(en_mask_after[(en_mask_after == True) & (en_mask == True)])
    T_Dac_pure = T_Dac_pure.astype(int)
    #gauss_TDAC = analysis.fit_gauss(T_Dac_hist_x, T_Dac_hist_y)
    # code for the fitting is above

    # code for plotting below
    xTDAC = T_Dac_pure
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    n, bins, _ = ax.hist(xTDAC, np.arange(-0.5, 32.5))
    ax.set_title('T-DAC Distro')
    ax.set_xlabel('T-DAC')
    ax.set_ylabel('No. of Pixels')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

#     A = gauss_TDAC[0]
#     mu = gauss_TDAC[1]
#     sigma = gauss_TDAC[2]
    try:
        lnspc = np.linspace(min(xTDAC) - 0.5, max(xTDAC) + 0.5, len(bins) - 1)
        popt, _ = optimize.curve_fit(analysis.gauss, lnspc, n, p0=(10, 16, np.std(xTDAC)), maxfev=1000, bounds=(2, 30))
        y = analysis.gauss(lnspc, *popt)
        ax.plot(lnspc, y, 'r--')
        print "T-DAC fit: ", popt
    except (RuntimeError, TypeError):
        print "error in fitting of T-DAC gaussian"

    print 'passed t-dac plot'
    ax.grid()
    fig.tight_layout()
    return fig


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
            #         if en_mask[i] == 1.0:
            T_Dac_pure = np.append(T_Dac_pure, t_dac[i])
    T_Dac_pure[T_Dac_pure < 0] = 0
    T_Dac_pure = T_Dac_pure.astype(int)
    T_Dac_hist_y = np.bincount(T_Dac_pure)
    T_Dac_hist_x = np.arange(0, T_Dac_hist_y.size, 1)
    #gauss_TDAC = analysis.fit_gauss(T_Dac_hist_x, T_Dac_hist_y)
    # code for the fitting is above

    # code for plotting below
    xTDAC = T_Dac_pure
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    n, bins, _ = ax.hist(xTDAC, np.arange(-0.5, 32.5))
    ax.set_title('T-DAC Distro')
    ax.set_xlabel('T-DAC')
    ax.set_ylabel('No. of Pixels')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid()

#     A = gauss_TDAC[0]
#     mu = gauss_TDAC[1]
#     sigma = gauss_TDAC[2]
    try:
        lnspc = np.linspace(min(xTDAC) - 0.5, max(xTDAC) + 0.5, len(bins) - 1)
#         print np.mean(xTDAC)
#         print np.std(xTDAC)
        popt, _ = optimize.curve_fit(analysis.gauss, lnspc, n, p0=(10, 16, 7), maxfev=1000)  # , bounds=(2, 30))
        y = analysis.gauss(lnspc, *popt)
        ax.plot(lnspc, y, 'r--')
        print "T-DAC fit: ", popt
    except (RuntimeError, TypeError):
        print "error in fitting of T-DAC gaussian"

    print 'passed t-dac plot'
    fig.tight_layout()
    return fig


def scan_pix_hist(h5_file_name, scurve_sel_pix=200):  # 200 is (3,8)
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
        scan_range_inx = np.arange(scan_range[0], scan_range[1], scan_range[2])  # scan range in volts
        chi2 = in_file_h5.root.Chisq_results.Chisq_scurve[:]
        chi2long = in_file_h5.root.Chisq_results.Chisq_scurve_unformatted[:]
        s_meas = in_file_h5.root.Scurves_Measurments.Scurve[:]
        thresh_hm_data = in_file_h5.root.Scurves_Measurments.threshold_hm[:]
        repeat_command = scan_args['repeat_command']
        np.set_printoptions(threshold=np.nan)
        param = np.unique(meta_data['scan_param_id'])
        ret = []
        for i in param:
            # this can be faster and multi threaded
            wh = np.where(hit_data['scan_param_id'] == i)
            hd = hit_data[wh[0]]
            hits = hd['col'].astype(np.uint16)
            hitsvalue = hits * 64
            hits = hits + hd['row']
            value = np.bincount(hits)
            value = np.pad(value, (0, 64 * 64 - value.shape[0]), 'constant')
            if len(ret):
                ret = np.vstack((ret, value))
            else:
                ret = value

        #repeat_command = max(ret[-3])
        shape = en_mask.shape
        ges = 1
        for i in range(2):
            ges = ges * shape[i]
        ret_pure = ()
        en_mask = en_mask.reshape(ges)
        for n in range(param[-1] + 1):
            ret_pure1 = ()
            for i in range(ges):
                if (str(en_mask[i]) == 'True'):
                    ret_pure1 = np.append(ret_pure1, ret[n][i])
            if n == 0:
                ret_pure = ret_pure1
                continue
            ret_pure = np.vstack((ret_pure, ret_pure1))

        ret_pure = ret_pure.astype(int)

        pix_scan_hist = np.empty((s_meas.shape[1], repeat_command + 40))
        for param in range(s_meas.shape[1]):
            h_count = np.bincount(s_meas[:, param])
            h_count = h_count[: repeat_command + 40]
            pix_scan_hist[param] = np.pad(h_count, (0, (repeat_command + 40) - h_count.shape[0]), 'constant')

        log_hist = np.log10(pix_scan_hist * 0.5)
        # print pix_scan_hist
        log_hist[~np.isfinite(log_hist)] = 0
        data = {
            'scan_param': np.ravel(np.indices(pix_scan_hist.shape)[0]),
            'count': np.ravel(np.indices(pix_scan_hist.shape)[1]),
            'value': np.ravel(log_hist)  # log_hist for log plot
        }

        # single pixel plot
        px = scurve_sel_pix  # 225
        x = np.arange(scan_range[0], scan_range[1], scan_range[2])
        pix_line = analysis.scurve(x, repeat_command, Threshold_pure[px], Noise_pure[px])
        # print Threshold_pure[px]

        fig1 = Figure()
        _ = FigureCanvas(fig1)
        ax_singlePix = fig1.add_subplot(111)
        ax_singlePix.plot(x, s_meas[px], 'bs')
        ax_singlePix.set_title('Single Pixle Scan ' + str(px))
        ax_singlePix.set_xlabel('Injection Voltage [V]')
        ax_singlePix.set_ylabel('Hits')
        ax_singlePix.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax_singlePix.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax_singlePix.plot(x, pix_line, 'r--')
        ax_singlePix.grid()

        fig1.tight_layout()
        print 'passed single pix plots'

        # scurve overlay
        # x is the same as for single pixel plot
        # print data[]

        fig2 = Figure()
        _ = FigureCanvas(fig2)
        fig2.clear()
        fig2.patch.set_facecolor("white")
        ax_thresHM = fig2.add_subplot(111)
        ax_thresHM_2 = ax_thresHM.twiny()
        ax_thresHM.set_title('Threshold curve overlay')
        ax_thresHM.set_xlabel('Electrons')
        ax_thresHM.set_ylabel('count')
        ax_thresHM_2.set_xlabel('Volts')
        ax_thresHM.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_thresHM.yaxis.set_minor_locator(AutoMinorLocator(4))
        cmap = plt.cm.viridis
        cmap.set_under(color='white')
        h = ax_thresHM.hist2d(x=(data['scan_param'] * scan_range[2]) + scan_range[0], y=data['count'], weights=data['value'], bins=(
            max(data['scan_param']), max(data['count'])), cmap=cmap, vmin=1e-10)
        ticks = ax_thresHM.get_xticks()
        bound = ax_thresHM.get_xbound()
        ax_thresHM_2.set_xticks(ticks)
        ax_thresHM_2.set_xbound(bound)
        ax_thresHM.set_xticklabels(analysis.cap_fac() * 1000 * ax_thresHM.get_xticks())
        fig2.colorbar(h[3], ax=ax_thresHM, pad=0.01)
        ax_thresHM.grid()

        print 'passed thresh HM plot'

        # threshold vs pix number
        y_ThVsPx = Threshold_pure
        x_ThVsPx = np.asarray(range(len(y_ThVsPx)))

        fig3 = Figure()
        _ = FigureCanvas(fig3)
        fig3.clear()
        ax_ThVsPx = fig3.add_subplot(111)
        ax_ThVsPx_2 = ax_ThVsPx.twinx()
        ax_ThVsPx.set_title('Threshold Scatter Plot')
        ax_ThVsPx.set_xlabel('Pixel Number')
        ax_ThVsPx.set_ylabel('Electrons')
        ax_ThVsPx.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax_ThVsPx.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax_ThVsPx.plot(x_ThVsPx, y_ThVsPx, '.')
        ax_ThVsPx_2.set_ylabel('Volts')
        ticks = ax_ThVsPx.get_yticks()
        bound = ax_ThVsPx.get_ybound()
        ax_ThVsPx_2.set_yticks(ticks)
        ax_ThVsPx_2.set_yticklabels(bound)
        ax_ThVsPx.set_yticklabels((analysis.cap_fac() * ticks * 1000).round())

        ax_ThVsPx.grid()
        print 'passed thesh vs pix'

        # Threshold distro

        x_th = Threshold_pure
        filtThres = [i for i in Threshold_pure if i != 0]
        # percent pixels w thresholds
        len_filt = len(filtThres)
        len_th = len(x_th)
        perc_pix = float(len_filt) / float(len_th) * 100
        fig4 = Figure()
        _ = FigureCanvas(fig4)
        fig4.clear()
        ax_ThresDist = fig4.add_subplot(111)
        ax_ThresDist_2 = ax_ThresDist.twiny()
#         n, bins, patchs = ax_ThresDist.hist(filtThres, bins=100)

        mu_th = np.mean(filtThres)
        sigma_th = np.sqrt(np.var(filtThres))
        # bincenters = 0.5*(bins[1:]+bins[:-1])

        # y_th = mlab.normpdf(lnspc, mu_th, sigma_th)
        ax_ThresDist.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_ThresDist.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax_ThresDist.set_title('Threshold Distribution, percent included: %s' % perc_pix, y=1.10)
        ax_ThresDist.set_xlabel('Electrons')
        ax_ThresDist.set_ylabel('No. of Pixels')

        ax_ThresDist.grid()

        print "mu_th: ", mu_th
        print "sigma_th: ", sigma_th
#         for a loop to fit each columsn see below:
        for flav in range(8):
            #  try:
            stop = flav * 512 + 512
            start = flav * 512
            thresh = Threshold_pure[start:stop]
            bar_data = 0
            bar_data, bins = np.histogram(thresh, 150, range=(min(filtThres), max(filtThres)))
            bin_left = bins[:-1]
            ax_ThresDist.bar(left=bin_left, height=bar_data, width=0.001, alpha=0.4, align="edge")

            lnspc_th = np.linspace(min(filtThres), max(filtThres), 150)
            popt_th, _ = optimize.curve_fit(analysis.gauss, lnspc_th, bar_data, p0=(20, thresh.mean(), thresh.std()), maxfev=1000)
            y_th = analysis.gauss(lnspc_th, *popt_th)
            ax_ThresDist.plot(lnspc_th, y_th, label=("Fl: %s Avg: %s \nSigma: %s" % (flav, (popt_th[1] * analysis.cap_fac() * 1000).round(2),
                                                                                     (popt_th[2] * analysis.cap_fac() * 1000).round(2))))
            ax_ThresDist.legend(prop={'size': 7})
            print "Threshold fit flavor ", flav, ": ", popt_th
            print "Threshold fit (electrons)flavor ", flav, ": ", popt_th * analysis.cap_fac() * 1000
#         try:
#             lnspc_th = np.linspace(min(filtThres), max(filtThres), 100)
#             popt_th, _ = optimize.curve_fit(analysis.gauss, lnspc_th, np.asarray(n), p0=(20, mu_th, sigma_th), maxfev=1000)
#             y_th = analysis.gauss(lnspc_th, *popt_th)
#             ax_ThresDist.plot(lnspc_th, y_th, 'r--', label=("Avg: %s \nSigma: %s" % ((popt_th[1] * analysis.cap_fac() * 1000).round(2),
#                                                                                      (popt_th[2] * analysis.cap_fac() * 1000).round(2))))
#             ax_ThresDist.legend(loc=0)
#             print "Threshold fit: ", popt_th
#             print "Threshold fit (electrons): ", popt_th * analysis.cap_fac() * 1000
#         except (RuntimeError, ValueError):
#             print("error in fitting of threshold gaussian")
        ticks = ax_ThresDist.get_xticks()
        bound = ax_ThresDist.get_xbound()
        ax_ThresDist.set_xticklabels((analysis.cap_fac() * ticks * 1000).round())
        ax_ThresDist_2.set_xticks(ticks)
        ax_ThresDist_2.set_xbound(bound)
#         ax_ThresDist_2.set_xticklabels((analysis.cap_fac() * ax_ThresDist.get_xticks() * 1000).round())

        ax_ThresDist_2.set_xlabel('Volts')
        ax_ThresDist_2.xaxis.set_minor_locator(AutoMinorLocator(5))
        fig4.tight_layout()

        print "total length threshold: ", len(x_th)
        print "reduced length threshold: ", len(filtThres)
        print "Percentage of Pixels with threshold: ", perc_pix
        print 'passed threshold distro'

        # noise v pixel
        y_noiseHM = Noise_pure
        x_noiseHM = np.asarray(range(len(y_noiseHM)))

        fig5 = Figure()
        _ = FigureCanvas(fig5)
        fig5.clear()
        ax_noiseHM = fig5.add_subplot(111)
        ax_noiseHM_2 = ax_noiseHM.twinx()
        ax_noiseHM.set_title('Noise by Pixel')
        ax_noiseHM.set_xlabel('pixel #')
        ax_noiseHM.set_ylabel('Electrons')
        h = ax_noiseHM.plot(x_noiseHM, y_noiseHM, '.')
        ax_noiseHM.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax_noiseHM.yaxis.set_minor_locator(AutoMinorLocator(5))
        ticks = ax_noiseHM.get_yticks()
        bound = ax_noiseHM.get_ybound()
        ax_noiseHM.set_yticklabels((analysis.cap_fac() * ticks * 1000).round())
        ax_noiseHM_2.set_yticks(ticks)
        ax_noiseHM_2.set_ybound(bound)
#         ax_noiseHM_2.set_yticklabels((analysis.cap_fac() * ax_noiseHM.get_yticks()).round())
        ax_noiseHM_2.set_ylabel('uV')
        ax_noiseHM.grid()
        fig5.tight_layout()

        print 'passed noise hm'

        # noise Hist
        noiseDist = Noise_pure
        filtNoiseDist = [x for x in noiseDist if (x != 0.0 and x != 0.02)]
        fig6 = Figure()
        _ = FigureCanvas(fig6)
        fig6.clear()
        ax_noiseDist = fig6.add_subplot(111)
        ax_noiseDist_2 = ax_noiseDist.twiny()
        ax_noiseDist.set_title('Noise Distribution without 0 and 0.02 V entries', y=1.10)
        ax_noiseDist.set_xlabel('Electrons')
        # ax_noiseDist.set_ylabel('pixel number')
        n_n, bins_n, patches_n = ax_noiseDist.hist(filtNoiseDist, bins=100)
        try:
            lnspc_n = np.linspace(np.min(filtNoiseDist),
                                  np.max(filtNoiseDist), 100)
            mu_n = np.mean(filtNoiseDist)
            sigma_n = np.sqrt(np.var(filtNoiseDist))
            popt_n, _ = optimize.curve_fit(analysis.gauss, lnspc_n, np.asarray(n_n), p0=(10, 0.005, 0.005), maxfev=1000)
            y_n = analysis.gauss(lnspc_n, *popt_n)
            print "Noise Gaussian: ", popt_n
            ax_noiseDist.plot(lnspc_n, y_n, "r--")
            ax_noiseDist.plot(lnspc_n, y_n, 'r--', label=("Avg: %s Sigma: %s" %
                                                          (popt_n[1] * analysis.cap_fac() * 1000, popt_n[2] * analysis.cap_fac() * 1000)))
            ax_noiseDist.legend(loc=0)

        except (RuntimeError, ValueError):
            print "error fitting noise gaussian, mean: ", np.mean(filtNoiseDist)
        ax_noiseDist.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_noiseDist.yaxis.set_minor_locator(AutoMinorLocator(2))
        ticks = ax_noiseDist.get_xticks()
        bound = ax_noiseDist.get_xbound()
        ax_noiseDist_2.set_xticks(ticks)
        ax_noiseDist_2.set_xbound(bound)
        ax_noiseDist.set_yticklabels((analysis.cap_fac() * ticks * 1000).round())
        ax_noiseDist_2.set_xlabel('Noise [uV]')
#         ax_noiseDist_2.set_xticklabels((analysis.cap_fac() * ax_noiseDist.get_xticks() * 1000).round())
        ax_noiseDist.grid()

        fig6.tight_layout()
        print "total length noise: ", len(noiseDist)
        print "reduced length noise: ", len(filtNoiseDist)

        chiX = [i for i in chi2long if i < 100 and i > 0.]

        # fig7 = Figure()
        # _ = FigureCanvas(fig7)
        # fig7.clear()
        # ax_chisq = fig7.add_subplot(111)
        # ax_chisq.set_title('chisq vs pixel')
        # ax_chisq.plot(np.asarray(range(len(chiX))), chiX, '.')
        print "filtered chisq: (0. < Chi2 < 100): ", len(chiX)
        print "average chisq: ", chi2long.mean()

        fig8 = Figure()
        _ = FigureCanvas(fig8)
        fig8.clear()
        ax_hist_chi = fig8.add_subplot(111)
        ax_hist_chi.set_title('filtered chi squared histogram')
        ax_hist_chi.hist(chiX, bins=200, range=(0, 100))
        ax_hist_chi.grid()

        # fig1 = singple pix plot
        # fig2 = thresh HM plot
        # fig3 = thresh vs pix
        # fig4 = threshold distro
        # fig5 = noise HM
        # fig6 = noise dist
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()
        fig5.tight_layout()
        fig6.tight_layout()
#         fig7.tight_layout()
        fig8.tight_layout()
        return fig1, fig2, fig3, fig4, fig5, fig6, fig8
