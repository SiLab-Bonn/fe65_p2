#------------------------------------------------------------------------------
# code to make the plots for test beam data,
# this file is focused on making comparision plots for different bias voltages, thresholds, etc.
# created by Daniel Coquelin on 8/5/18
#------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import tables as tb
import analysis as analysis
import yaml
import analysis as analysis
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import NullFormatter
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from scipy import optimize
from matplotlib.backends.backend_pdf import PdfPages


cmap = plt.cm.viridis
cmap.set_under(color='white')

eff_file_h5 = '/media/daniel/Maxtor/fe65p2_testbeam_april_2018/efficiency_over_all_runs_wo7.h5'
# fe_name_list2 = ['CPL000', 'CPL001', 'CPL011', 'CPL001RH', 'CPL100', 'CPL101', 'CPL101RH', 'CPL101']
# fe_name_list = ['fe0', 'fe1', 'fe2', 'fe3', 'fe4', 'fe5', 'fe6', 'fe7']
# pixel_flav_list = ['nw15', 'nw20', 'nw25', 'nw30', 'dnw15', 'dnw20', 'dnw25', 'dnw30']

fe_name_list2 = ['CPL000', 'CPL001', 'CPL011', 'CPL001RH', 'CPL100', 'CPL101', 'CPL101RH']
fe_name_list = ['fe0', 'fe1', 'fe2', 'fe3', 'fe4', 'fe5', 'fe6', 'fe7']
pixel_flav_list = ['nw15', 'nw20', 'nw25', 'nw30', 'dnw15', 'dnw20', 'dnw25', 'dnw30']


def sensor_and_FE_plots(bias_voltage=None, vth1=None):
    with tb.open_file(eff_file_h5, 'r+') as eff_file:
        eff_table = eff_file.root.eff_table[:]

    if bias_voltage:
        print 'bias', bias_voltage
        eff_table = eff_table[eff_table['bias'] == bias_voltage]

        # one plot with all overlayed
        fig = Figure()
        _ = FigureCanvas(fig)
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Global Threshold")
        ax.set_ylabel("Efficiency (%)")
        ax.set_title("Efficiency vs Global Threshold at a Bias Voltage of %s \nSensor Flavors" % str(bias_voltage))
        # total
        ax.errorbar(x=eff_table['vth1'], y=eff_table['cuts'], yerr=[eff_table['cuts_errm'] * -1,
                                                                    eff_table['cuts_errp'] * 1], fmt='o', label='Total', markersize=3, capsize=5)
        # nw15
        for name in pixel_flav_list:
            ax.errorbar(eff_table['vth1'], eff_table[name], yerr=[eff_table[name + '_errm'] * -1, eff_table[name + '_errp'] * 1],
                        fmt='o', label=name, markersize=3, capsize=2)
        ax.set_ylim([95, 100])
        ax.legend()
        ax.grid()

        fig1 = Figure()
        _ = FigureCanvas(fig1)
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel("Global Threshold")
        ax1.set_ylabel("Efficiency (%)")
        ax1.set_title("Efficiency vs Global Threshold at a Bias Voltage of %s \nFrontend Flavors" % str(bias_voltage))
        # total
        ax1.errorbar(x=eff_table['vth1'], y=eff_table['cuts'], yerr=[eff_table['cuts_errm'] * -1,
                                                                     eff_table['cuts_errp'] * 1], fmt='o', label='Total', markersize=3, capsize=5)
        for i in range(8):
            ax1.errorbar(eff_table['vth1'], eff_table['fe' + str(i)], yerr=[eff_table['fe' + str(i) + '_errm'] * -1, eff_table['fe' + str(i) + '_errp'] * 1],
                         fmt='o', label='fe' + str(i), markersize=3, capsize=2)

        ax1.set_ylim([95, 100])
        ax1.legend()
        ax1.grid()
    elif vth1:
        print 'vth1', vth1
        eff_table = eff_table[eff_table['vth1'] == vth1]
#         print eff_table[(eff_table['vth1'] == 203)]

        # one plot with all overlayed
        fig = Figure()
        _ = FigureCanvas(fig)
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_title("Efficiency vs Bias Voltage at a Global Threshold of %s\nSensor Flavors" % str(vth1))
        ax.set_xlabel("Bias (V)")
        ax.set_ylabel("Efficiency (%)")
        ax.grid()
        # total
        ax.errorbar(x=eff_table['bias'], y=eff_table['cuts'], yerr=[eff_table['cuts_errm'] * -1,
                                                                    eff_table['cuts_errp'] * 1], fmt='o', label='Total', markersize=3, capsize=5)
        for name in pixel_flav_list:
            ax.errorbar(eff_table['bias'], eff_table[name], yerr=[eff_table[name + '_errm'] * -1, eff_table[name + '_errp'] * 1],
                        fmt='o', label=name, markersize=3, capsize=2)

        ax.set_ylim([95, 100])
        ax.invert_xaxis()
        ax.legend()

        fig1 = Figure()
        _ = FigureCanvas(fig1)
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel("Bias (V)")
        ax1.set_ylabel("Efficiency (%)")
        ax1.set_title("Efficiency vs Bias Voltage at a Global Threshold of %s\nFrontend Flavors" % str(vth1))
        # total
        ax1.errorbar(x=eff_table['bias'], y=eff_table['cuts'], yerr=[eff_table['cuts_errm'] * -1,
                                                                     eff_table['cuts_errp'] * 1], fmt='o', label='Total', markersize=3, capsize=5)
        for i in range(8):
            ax1.errorbar(eff_table['bias'], eff_table['fe' + str(i)], yerr=[eff_table['fe' + str(i) + '_errm'] * -1, eff_table['fe' + str(i) + '_errp'] * 1],
                         fmt='o', label='fe' + str(i), markersize=3, capsize=2)

        ax1.set_ylim([95, 100])
        ax1.invert_xaxis()
        ax1.legend()
        ax1.grid()
    else:
        print "provide either bias or vth1"

    return fig, fig1


def combined_sensor_and_FE_plots(bias_voltage=None, vth1=None):
    with tb.open_file(eff_file_h5, 'r+') as eff_file:
        eff_table = eff_file.root.eff_table[:]

    combi_dict = {'nw15': ['fe0', 'fe1', 'fe2', 'fe3'],
                  'nw20': ['fe0', 'fe1', 'fe2', 'fe3'],
                  'nw25': ['fe0', 'fe1', 'fe2', 'fe3'],
                  'nw30': ['fe0', 'fe1', 'fe2', 'fe3'],
                  'dnw15': ['fe4', 'fe5', 'fe6', 'fe7'],
                  'dnw20': ['fe4', 'fe5', 'fe6', 'fe7'],
                  'dnw25': ['fe4', 'fe5', 'fe6', 'fe7'],
                  'dnw30': ['fe4', 'fe5', 'fe6', 'fe7'],
                  'fe0': ['nw15', 'nw20', 'nw25', 'nw30'],
                  'fe1': ['nw15', 'nw20', 'nw25', 'nw30'],
                  'fe2': ['nw15', 'nw20', 'nw25', 'nw30'],
                  'fe3': ['nw15', 'nw20', 'nw25', 'nw30'],
                  'fe4': ['dnw15', 'dnw20', 'dnw25', 'dnw30'],
                  'fe5': ['dnw15', 'dnw20', 'dnw25', 'dnw30'],
                  'fe6': ['dnw15', 'dnw20', 'dnw25', 'dnw30'],
                  'fe7': ['dnw15', 'dnw20', 'dnw25', 'dnw30']}
    if bias_voltage:
        fig_list = []
        print 'bias', bias_voltage
        eff_table = eff_table[eff_table['bias'] == bias_voltage]

        for sens in pixel_flav_list:
            fig = Figure()
            _ = FigureCanvas(fig)
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_title("Efficiency vs Global Threshold at a Bias Voltage of %s" % str(bias_voltage))
            ax.set_xlabel("Global Threshold")
            ax.set_ylabel("Efficiency (%)")
            for fe in combi_dict[sens]:
                name = sens + '_' + fe
                ax.errorbar(eff_table['vth1'], eff_table[name], yerr=[eff_table[name + '_errm'] * -1, eff_table[name + '_errp'] * 1],
                            fmt='o', label=name, markersize=3, capsize=2)
            ax.set_ylim([95, 100])
            ax.legend()
            ax.grid()
            fig_list.append(fig)

        for fe in fe_name_list:
            fig1 = Figure()
            _ = FigureCanvas(fig1)
            fig1.clear()
            ax1 = fig1.add_subplot(111)
            ax1.set_title("Efficiency vs Global Threshold at a Bias Voltage of %s" % str(bias_voltage))
            ax1.set_xlabel("Global Threshold")
            ax1.set_ylabel("Efficiency (%)")
            for sens in combi_dict[fe]:
                name = sens + '_' + fe
                ax1.errorbar(eff_table['vth1'], eff_table[name], yerr=[eff_table[name + '_errm'] * -1, eff_table[name + '_errp'] * 1],
                             fmt='o', label=name, markersize=3, capsize=2)
            ax1.set_ylim([95, 100])
            ax1.legend()
            ax1.grid()
            fig_list.append(fig1)

    elif vth1:
        print 'vth1', vth1
        eff_table = eff_table[eff_table['vth1'] == vth1]
        fig_list = []

        for sens in pixel_flav_list:
            fig = Figure()
            _ = FigureCanvas(fig)
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_title("Efficiency vs Bias Voltage at a Global Threshold of %s" % str(vth1))
            ax.set_xlabel("Bias (V)")
            ax.set_ylabel("Efficiency (%)")
            for fe in combi_dict[sens]:
                name = sens + '_' + fe
                ax.errorbar(eff_table['bias'], eff_table[name], yerr=[eff_table[name + '_errm'] * -1, eff_table[name + '_errp'] * 1],
                            fmt='o', label=name, markersize=3, capsize=2)
            ax.set_ylim([95, 100])
            ax.invert_xaxis()
            ax.legend()
            ax.grid()
            fig_list.append(fig)

        for fe in fe_name_list:
            fig1 = Figure()
            _ = FigureCanvas(fig1)
            fig1.clear()
            ax1 = fig1.add_subplot(111)
            ax1.set_title("Efficiency vs Bias Voltage at a Global Threshold of %s" % str(vth1))
            ax1.set_xlabel("Bias (V)")
            ax1.set_ylabel("Efficiency (%)")
            for sens in combi_dict[fe]:
                name = sens + '_' + fe
                # print eff_table[name], eff_table[name + '_errm'] * 1, eff_table[name + '_errm'] * 1
                ax1.errorbar(eff_table['bias'], eff_table[name], yerr=[eff_table[name + '_errm'] * -1, eff_table[name + '_errp'] * 1],
                             fmt='o', label=name, markersize=3, capsize=2)

            ax1.set_ylim([95, 100])
            ax1.legend()
            ax1.invert_xaxis()
            ax1.grid()
            fig_list.append(fig1)
    return fig_list


def col_comparison(feX, feY, bias_voltage=None, vth1=None):
    # function to create plots of only two different readout flavors
    with tb.open_file(eff_file_h5, 'r+') as eff_file:
        eff_table = eff_file.root.eff_table[:]

#     fe_name_list
    fe_list = [feX, feY]
    if feX not in fe_name_list or feY not in fe_name_list:
        print 'either feX or feY not in names, check fe_name_list', feX, feY
    if bias_voltage:
        print 'bias', bias_voltage
        eff_table = eff_table[eff_table['bias'] == bias_voltage]
        fig1 = Figure()
        _ = FigureCanvas(fig1)
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        ax1.set_title("Efficiency vs Global Threshold at a Bias Voltage of %s\n Comparison between FE columns %s and %s" %
                      (str(bias_voltage), feX, feY))
        ax1.set_xlabel("Global Threshold")
        ax1.set_ylabel("Efficiency (%)")
        for fe in fe_list:
            ax1.errorbar(eff_table['vth1'], eff_table[fe], yerr=[eff_table[fe + '_errm'] * -1., eff_table[fe + '_errp'] * 1.],
                         fmt='o', label=fe, markersize=3, capsize=2)
        ax1.set_ylim([95, 100])
        ax1.legend()
        ax1.grid()

    elif vth1:
        print 'vth1', vth1
        eff_table = eff_table[eff_table['vth1'] == vth1]

        fig1 = Figure()
        _ = FigureCanvas(fig1)
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        ax1.set_title("Efficiency vs Bias Voltage at a Global Threshold of %s\n Comparison between FE columns %s and %s" % (str(vth1), feX, feY))
        ax1.set_xlabel("Bias (V)")
        ax1.set_ylabel("Efficiency (%)")
        for fe in fe_list:
            ax1.errorbar(eff_table['bias'], eff_table[fe], yerr=[eff_table[fe + '_errm'] * -1., eff_table[fe + '_errp'] * 1.],
                         fmt='o', label=fe, markersize=3, capsize=2)

        ax1.set_ylim([95, 100])
        ax1.invert_xaxis()
        ax1.legend()
        ax1.grid()
    return fig1


def compare_nw_w_dnw(nw, dnw, bias_voltage=None, vth1=None):
    with tb.open_file(eff_file_h5, 'r+') as eff_file:
        eff_table = eff_file.root.eff_table[:]

#     fe_name_list
    sens_list = [nw, dnw]
    if nw not in pixel_flav_list or dnw not in pixel_flav_list:
        print 'either feX or feY not in names, check pixel_flav_list', nw, dnw
    if bias_voltage:
        print 'bias', bias_voltage
        eff_table = eff_table[eff_table['bias'] == bias_voltage]
        fig1 = Figure()
        _ = FigureCanvas(fig1)
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        ax1.set_title("Efficiency vs Global Threshold at a Bias Voltage of %s\n Comparison between Sensor Flavors %s and %s" %
                      (str(bias_voltage), nw, dnw))
        ax1.set_xlabel("Global Threshold")
        ax1.set_ylabel("Efficiency (%)")
        for s in sens_list:
            ax1.errorbar(eff_table['vth1'], eff_table[s], yerr=[eff_table[s + '_errm'] * -1., eff_table[s + '_errp'] * 1.],
                         fmt='o', label=s, markersize=3, capsize=2)
        ax1.set_ylim([95, 100])
        ax1.legend()
        ax1.grid()

    elif vth1:
        print 'vth1', vth1
        eff_table = eff_table[eff_table['vth1'] == vth1]

        fig1 = Figure()
        _ = FigureCanvas(fig1)
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        ax1.set_title("Efficiency vs Bias Voltage at a Global Threshold of %s\n Comparison between Sensor Flavors %s and %s" %
                      (str(vth1), nw, dnw))
        ax1.set_xlabel("Bias (V)")
        ax1.set_ylabel("Efficiency (%)")
        for s in sens_list:
            ax1.errorbar(eff_table['bias'], eff_table[s], yerr=[eff_table[s + '_errm'] * -1., eff_table[s + '_errp'] * 1.],
                         fmt='o', label=s, markersize=3, capsize=2)

        ax1.set_ylim([95, 100])
        ax1.invert_xaxis()
        ax1.legend()
        ax1.grid()
    return fig1


def eff_vs_nw_depth(bias_voltage=None, vth1=None):
    with tb.open_file(eff_file_h5, 'r+') as eff_file:
        eff_table = eff_file.root.eff_table[:]

    if bias_voltage and vth1:
        print 'bias', bias_voltage, 'vth1', vth1
        eff_table = eff_table[eff_table['bias'] == bias_voltage]
        eff_table = eff_table[eff_table['vth1'] == vth1]
        fig1 = Figure()
        _ = FigureCanvas(fig1)
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        ax1.set_title("Efficiency vs Pixel Type at a Bias Voltage of %s\n and Global Threshold of %s" % (str(bias_voltage), str(vth1)))
        ax1.set_xlabel("Sensor Flavors")
        ax1.set_ylabel("Efficiency (%)")
        for i, pix in enumerate(pixel_flav_list):
            #             print eff_table[pix].shape
            #             print eff_table[pix + '_errm'].shape
            #             print eff_table[pix + '_errp'].shape
            #             print [eff_table[pix + '_errm'] * -1., eff_table[pix + '_errp'] * 1.]

            ax1.errorbar(pix, eff_table[pix],  # yerr=[eff_table[pix + '_errm'] * -1., eff_table[pix + '_errp'] * 1.],
                         fmt='o', markersize=3, capsize=2)
        ax1.set_ylim([95, 100])
        ax1.grid()

        fig2 = Figure()
        _ = FigureCanvas(fig2)
        fig2.clear()
        ax2 = fig2.add_subplot(111)
        ax2.set_title("Efficiency vs Pixel Type at a Bias Voltage of %s\n and Global Threshold of %s" % (str(bias_voltage), str(vth1)))
        ax2.set_xlabel("Frontend Flavors")
        ax2.set_ylabel("Efficiency (%)")
        for i, fe in enumerate(fe_name_list):
            #             print eff_table[pix].shape
            #             print eff_table[pix + '_errm'].shape
            #             print eff_table[pix + '_errp'].shape
            #             print [eff_table[pix + '_errm'] * -1., eff_table[pix + '_errp'] * 1.]

            ax2.errorbar(fe, eff_table[fe],  # yerr=[eff_table[fe + '_errm'] * -1., eff_table[fe + '_errp'] * 1.],
                         fmt='o', markersize=3, capsize=2)
        ax2.set_ylim([95, 100])
        ax2.grid()

    return fig1, fig2


def poll_effs(bias_line=None, vth1_line=None, below_bias=None, above_vth1=None, below_vth1=None):
    with tb.open_file(eff_file_h5, 'r+') as eff_file:
        eff_table = eff_file.root.eff_table[:]

    poll_fe = {fe: [0, 0, 0, 0, 0, 0, 0, 0] for fe in fe_name_list}
    poll_nw = {nw: [0, 0, 0, 0, 0, 0, 0, 0] for nw in pixel_flav_list}

    fig_list = []
    if bias_line:
        print 'above bias'
        fig_list = poll_top_effs(run_list=eff_table[eff_table['bias'] * -1 >= bias_line]
                                 ['run_num'], title="Bias >= %s" % str(bias_line))
        print 'below bias'
        fig_list.extend(poll_top_effs(run_list=eff_table[eff_table['bias'] * -1 < bias_line]
                                      ['run_num'], title="Bias below %s" % str(bias_line)))
        print 'finished bias'

    if vth1_line:
        print "above vth1"
        fig_list = poll_top_effs(run_list=eff_table[eff_table['vth1'] >= vth1_line]
                                 ['run_num'], title="Vth1 >= %s" % str(vth1_line))
        print 'below vth1'
        fig_list.extend(poll_top_effs(run_list=eff_table[eff_table['vth1'] * -1 < vth1_line]
                                      ['run_num'], title="vth1 below %s" % str(vth1_line)))
        print 'finished vth1'
    return fig_list


def poll_top_effs(run_list=None, title=None):
    with tb.open_file(eff_file_h5, 'r') as eff_file:
        eff_table = eff_file.root.eff_table[:]

    poll_fe = {fe: [0, 0, 0, 0, 0, 0, 0, 0] for fe in fe_name_list}
    poll_nw = {nw: [0, 0, 0, 0, 0, 0, 0, 0] for nw in pixel_flav_list}
    nw_effs = {nw: [0, 0, 0] for nw in pixel_flav_list}
    fe_effs = {fe: [0, 0, 0] for fe in fe_name_list}

    if run_list is None:
        run_list = eff_table['run_num']

    nw_runs = {nw: [float(len(run_list)), float(len(run_list)), float(len(run_list))] for nw in pixel_flav_list}
    fe_runs = {fe: [float(len(run_list)), float(len(run_list)), float(len(run_list))] for fe in fe_name_list}
    for run in run_list:
        eff_table_holdf = eff_table[eff_table['run_num'] == run]
        eff_table_hold = [eff_table_holdf[x][0] for x in fe_name_list]
        for fe in fe_name_list:
            fe_effs[fe][0] += eff_table_holdf[fe][0]
            if eff_table_holdf[fe + '_errp'][0] > 0.:
                fe_effs[fe][1] += eff_table_holdf[fe + '_errp'][0]
            else:
                fe_runs[fe][1] -= 1.
            if eff_table_holdf[fe + '_errm'][0] < 0.:
                fe_effs[fe][2] += eff_table_holdf[fe + '_errm'][0]
            else:
                fe_runs[fe][2] -= 1.

        order = np.sort(eff_table_hold)
#         print order
#         print eff_table_hold
#         print np.where(eff_table_hold == order[-1])[0][0]
        poll_fe[fe_name_list[np.where(eff_table_hold == order[7])[0][0]]][0] += 1
        poll_fe[fe_name_list[np.where(eff_table_hold == order[6])[0][0]]][1] += 1
        poll_fe[fe_name_list[np.where(eff_table_hold == order[5])[0][0]]][2] += 1
        poll_fe[fe_name_list[np.where(eff_table_hold == order[4])[0][0]]][3] += 1
        poll_fe[fe_name_list[np.where(eff_table_hold == order[3])[0][0]]][4] += 1
        poll_fe[fe_name_list[np.where(eff_table_hold == order[2])[0][0]]][5] += 1
        poll_fe[fe_name_list[np.where(eff_table_hold == order[1])[0][0]]][6] += 1
        poll_fe[fe_name_list[np.where(eff_table_hold == order[0])[0][0]]][7] += 1

        # need to fix this tomorrow!

        eff_table_hold = [eff_table_holdf[x][0] for x in pixel_flav_list]
        for nw in pixel_flav_list:
            nw_effs[nw][0] += eff_table_holdf[nw][0]
            nw_effs[nw][1] += eff_table_holdf[nw + '_errp'][0]
            nw_effs[nw][2] += eff_table_holdf[nw + '_errm'][0]
            if eff_table_holdf[nw + '_errp'][0] > 0.:
                nw_effs[nw][1] += eff_table_holdf[nw + '_errp'][0]
            else:
                nw_runs[nw][1] -= 1.
            if eff_table_holdf[nw + '_errm'][0] < 0.:
                nw_effs[nw][2] += eff_table_holdf[nw + '_errm'][0]
            else:
                nw_runs[nw][2] -= 1.

        order = np.sort(eff_table_hold)

        poll_nw[pixel_flav_list[np.where(eff_table_hold == order[7])[0][0]]][0] += 1
        poll_nw[pixel_flav_list[np.where(eff_table_hold == order[6])[0][0]]][1] += 1
        poll_nw[pixel_flav_list[np.where(eff_table_hold == order[5])[0][0]]][2] += 1
        poll_nw[pixel_flav_list[np.where(eff_table_hold == order[4])[0][0]]][3] += 1
        poll_nw[pixel_flav_list[np.where(eff_table_hold == order[3])[0][0]]][4] += 1
        poll_nw[pixel_flav_list[np.where(eff_table_hold == order[2])[0][0]]][5] += 1
        poll_nw[pixel_flav_list[np.where(eff_table_hold == order[1])[0][0]]][6] += 1
        poll_nw[pixel_flav_list[np.where(eff_table_hold == order[0])[0][0]]][7] += 1

    # want the average eff w error for each fe and flavor

    fig_list = []
    print "\n\tFrontend Table"
    for x in poll_fe:
        # histogram the dictionaries
        # the values are already histogrammed
        #         avg_eff = np.mean(eff_table[eff_table['run_num'] == run_list][x])
        #         avg_eff_errp = np.mean(eff_table_2[x + '_errp'])
        #         avg_eff_errm = np.mean(eff_table_2[x + '_errm'])
        perc_1 = 100. * (float(poll_fe[x][0]) / float(len(run_list)))
        perc_2 = 100. * (float(poll_fe[x][1]) / float(len(run_list)))
        perc_3 = 100. * (float(poll_fe[x][2]) / float(len(run_list)))
        perc_2ndlast = 100. * (float(poll_fe[x][-2]) / float(len(run_list)))
        perc_last = 100. * (float(poll_fe[x][-1]) / float(len(run_list)))
        if 0 not in fe_runs[x]:
            print x, "avg eff", fe_effs[x][0] / fe_runs[x][0], "+", fe_effs[x][1] / fe_runs[x][1], "-", fe_effs[x][2] / fe_runs[x][2], "\tperc1:", perc_1, "perc2:", perc_2, "perc_3:", perc_3, "perc last:", perc_last

        # print x, "avg eff", avg_eff, "+", avg_eff_errp, "-", avg_eff_errm,
        # "\tperc1:", perc_1, "perc2:", perc_2, "perc_3:", perc_3, "perc 2nd
        # last:", perc_2ndlast, "perc last:", perc_last
        fig = Figure()
        _ = FigureCanvas(fig)
        fig.clear()
        ax = fig.add_subplot(111)

        ax.bar(x=range(8), height=poll_fe[x], align="center")
        if title:
            ax.set_title("%s\nOrder of %s" % (title, str(x)))
        else:
            ax.set_title("All Runs\nOrder of %s" % str(x))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid()
        fig_list.append(fig)
    print "\n\tSensor table"
    for x in poll_nw:
        #         avg_eff = np.mean(eff_table_2[x])
        #         avg_eff_errp = np.mean(eff_table_2[x + '_errp'])
        #         avg_eff_errm = np.mean(eff_table_2[x + '_errm'])
        perc_1 = 100. * (float(poll_nw[x][0]) / float(len(run_list)))
        perc_2 = 100. * (float(poll_nw[x][1]) / float(len(run_list)))
        perc_3 = 100. * (float(poll_nw[x][2]) / float(len(run_list)))
        perc_last = 100. * (float(poll_nw[x][-1]) / float(len(run_list)))

        print x, "avg eff", nw_effs[x][0] / nw_runs[x][0], "+", nw_effs[x][1] / nw_runs[x][1], "-", nw_effs[x][2] / nw_runs[x][2], "\tperc1:", perc_1, "perc2:", perc_2, "perc_3:", perc_3, "perc last:", perc_last

        fig = Figure()
        _ = FigureCanvas(fig)
        fig.clear()
        ax = fig.add_subplot(111)

        ax.bar(x=range(8), height=poll_nw[x], align="center")
        if title:
            ax.set_title("%s\nOrder of %s" % (title, str(x)))
        else:
            ax.set_title("All Runs\nOrder of %s" % str(x))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid()
        fig_list.append(fig)
    return fig_list


if __name__ == "__main__":
    bias_list = [-100, -175, -35]
    vth1_list = [43, 143, 203]

#     pdfName = "/media/daniel/Maxtor/fe65p2_testbeam_april_2018/eff_analysis_plots_wo7.pdf"
#     pp = PdfPages(pdfName)
#     for bias in bias_list:
#         fig, fig1 = sensor_and_FE_plots(bias_voltage=bias)
#         pp.savefig(fig, layout='tight')
#         plt.clf()
#         pp.savefig(fig1, layout='tight')
#         plt.clf()
#
#     for vth1 in vth1_list:
#         fig, fig1 = sensor_and_FE_plots(vth1=vth1)
#         pp.savefig(fig, layout='tight')
#         plt.clf()
#         pp.savefig(fig1, layout='tight')
#         plt.clf()
#     pp.close()
#
#     pdfName2 = "/media/daniel/Maxtor/fe65p2_testbeam_april_2018/eff_analysis_plots_bias_combi_wo7.pdf"
#     pp2 = PdfPages(pdfName2)
#     for bias in bias_list:
#         fig_list = combined_sensor_and_FE_plots(bias_voltage=bias)
#         for x in fig_list:
#             pp2.savefig(x, layout='tight')
#             plt.clf()
#     pp2.close()
#
#     pdfName3 = "/media/daniel/Maxtor/fe65p2_testbeam_april_2018/eff_analysis_plots_vth1_combi_wo7.pdf"
#     pp3 = PdfPages(pdfName3)
#     for vth1 in vth1_list:
#         fig_list = combined_sensor_and_FE_plots(vth1=vth1)
#         for x in fig_list:
#             pp3.savefig(x, layout='tight')
#             plt.clf()
#     pp3.close()
#
#     pdfName4 = "/media/daniel/Maxtor/fe65p2_testbeam_april_2018/eff_analysis_plots_fe_comp_wo7.pdf"
#     col_comp_list = [[3, 5, 0, 1, 0, 1, 5], [6, 7, 4, 5, 7, 3, 6]]
#
#     pp4 = PdfPages(pdfName4)
#     for i, _ in enumerate(col_comp_list[0]):
#         print '\nfe' + str(col_comp_list[0][i]) + ' vs fe' + str(col_comp_list[1][i])
#         for bias in bias_list:
#             fig = col_comparison(feX='fe' + str(col_comp_list[0][i]), feY='fe' + str(col_comp_list[1][i]), bias_voltage=bias)
#             pp4.savefig(fig, layout='tight')
#             plt.clf()
#         for vth1 in vth1_list:
#             fig = col_comparison(feX='fe' + str(col_comp_list[0][i]), feY='fe' + str(col_comp_list[1][i]), vth1=vth1)
#             pp4.savefig(fig, layout='tight')
#             plt.clf()
#
#     pp4.close()
#
#     pdfName5 = "/media/daniel/Maxtor/fe65p2_testbeam_april_2018/eff_analysis_plots_nw_v_dnw_wo7.pdf"
#     sens_comp_list = [15, 20, 25, 30]
#
#     pp5 = PdfPages(pdfName5)
#     for i in sens_comp_list:
#         print '\nnw' + str(i) + ' vs dnw' + str(i)
#         for bias in bias_list:
#             fig = compare_nw_w_dnw(nw='nw' + str(i), dnw='dnw' + str(i), bias_voltage=bias)
#             pp5.savefig(fig, layout='tight')
#             plt.clf()
#         for vth1 in vth1_list:
#             fig = compare_nw_w_dnw(nw='nw' + str(i), dnw='dnw' + str(i), vth1=vth1)
#             pp5.savefig(fig, layout='tight')
#             plt.clf()
#     bias_vth1_list = [[-100, -75, -149, -76, -13], [103, 53, 43, 43, 43]]
#     for v, b in enumerate(bias_vth1_list[0]):
#         print bias_vth1_list[1][v], b
#         fig, fig2 = eff_vs_nw_depth(bias_voltage=b, vth1=bias_vth1_list[1][v])
#         pp5.savefig(fig, layout="tight")
#         plt.clf()
#         pp5.savefig(fig2, layout="tight")
#         plt.clf()
#     pp5.close()

    print "\tbias line = 100"
    poll_figs = poll_effs(bias_line=100)
    print "\n------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
    print "\tvth1 line = 105"
    poll_figs.extend(poll_effs(vth1_line=105))
    print "\n------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
    print "\tAll scans"
    poll_figs.extend(poll_top_effs())

#     pdfName6 = "/media/daniel/Maxtor/fe65p2_testbeam_april_2018/eff_analysis_polls_w7.pdf"
#     pp6 = PdfPages(pdfName6)
#     for x in poll_figs:
#         pp6.savefig(x, layout='tight')
#         plt.clf()
#     pp6.close()
