#------------------------------------------------------------------------------
# script to analyze test beam data
# first function is to get hit data for fe65 (should be take care of for fei4s)
#
# plots to make:
# correlation
# ????
#------------------------------------------------------------------------------


import DGC_plotting
import numpy as np
import tables as tb
import tables as tb
import analysis as analysis
import yaml
import scans.noise_tuning_columns as noise_cols
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import NullFormatter
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
from scipy import optimize
import matplotlib.pyplot as plt
from os import getcwd, chdir
from multiprocessing import Pool
import glob
import re
from matplotlib.backends.backend_pdf import PdfPages
from scan_base import ScanBase
import fe65p2
import matplotlib.pyplot as plt
from testbeam_analysis.tools import analysis_utils
#------------------------------------------------------------------------------
#
#
#
#------------------------------------------------------------------------------


def overlay_eff_hm_w_disabled_pix(fe65p2_h5_file, eff_h5_file,  res_file, noisy_h5, eff_binning=[64, 64]):
    # print 'for this function the efficiency must be binned with each pixel (64x64)'
    # overlay of the efficiency of the pixel with the disabled pixels circled in red
    # circle pixels in purple (or something like that) which have 0 occupancy
    with tb.open_file(res_file, 'r+') as res_in_file:
        resX_fit_results = res_in_file.root.ResidualsX_DUT1.attrs.fit_coeff
        resY_fit_results = res_in_file.root.ResidualsY_DUT1.attrs.fit_coeff

    sigma_x = resX_fit_results[2]
    sigma_y = resY_fit_results[2]

    # have the error on the track fits in um -> will cut the pixels within 2 sigma :/

    one_side_pix_to_ignore_x = round((sigma_x * 2 / 50. - 1.), 0)  # cuts 2 * sigma and then converts to a number of pixels

    one_side_pix_to_ignore_y = round((sigma_y * 2 / 50. - 1.), 0)  # cuts 2 * sigma and then converts to a number of pixels

    # print one_side_pix_to_ignore_x, one_side_pix_to_ignore_y

    with tb.open_file(eff_h5_file, 'r+') as eff_in_file:
        eff_data = eff_in_file.root.DUT_1.Efficiency[:]
        total_tracks = eff_in_file.root.DUT_1.Total_tracks[:]
        passing_tracks = eff_in_file.root.DUT_1.Passing_tracks[:]
    with tb.open_file(noisy_h5, 'r+') as noisy:
        noise_mask = noisy.root.NoisyPixelMask[:]
    with tb.open_file(fe65p2_h5_file, 'r+') as in_file_h5:
        raw_data = in_file_h5.root.raw_data[:]
        meta_data = in_file_h5.root.meta_data[:]
        scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
        hit_data = in_file_h5.root.hit_data[:]

        file0 = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/chip3_tuning/20180327_172429_noise_tuning.h5'
        file1 = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/chip3_tuning/20180327_183923_noise_tuning.h5'
        file2 = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/chip3_tuning/20180327_195210_noise_tuning.h5'
        file3 = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/chip3_tuning/20180327_205655_noise_tuning.h5'
        file4 = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/chip3_tuning/20180327_220635_noise_tuning.h5'
        file5 = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/chip3_tuning/20180327_232100_noise_tuning.h5'
        file6 = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/chip3_tuning/20180328_003045_noise_tuning.h5'
        file7 = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/chip3_tuning/20180328_013525_noise_tuning.h5'
        mask_en_from_file, mask_tdac, _ = noise_cols.combine_prev_scans(
            file0=file0, file1=file1, file2=file2, file3=file3, file4=file4, file5=file5, file6=file6, file7=file7)
        mask_en_from_file = mask_en_from_file.reshape(4096)

        occ = np.histogram2d(x=hit_data['col'], y=hit_data['row'], bins=(64, 64), range=((0, 64), (0, 64)))[0]

        # these lists are all indexes
        man_disabled = scan_args['ex_pix_disable_list']
        tune_disabled = np.where(mask_en_from_file == False)
        occ_0 = np.where((occ.reshape(4096) < 10) & (mask_en_from_file == True))
#         occ_0 = [x for x in occ_0 if x not in man_disabled]

        man_dis_mask = np.full(4096, False)  # true means DISABLED
        tune_dis_mask = np.full(4096, False)  # true means DISABLED
        occ_0_mask = np.full(4096, False)  # true means DISABLED

        tune_dis_mask[tune_disabled] = True
        tune_dis_mask[noise_mask.reshape(4096) == True] = True
        man_dis_mask[man_disabled] = True
        occ_0_mask[occ_0] = True

        tune_dis_mask = tune_dis_mask.reshape((64, 64))
        man_dis_mask = man_dis_mask.reshape((64, 64))
        occ_0_mask = occ_0_mask.reshape((64, 64))

    eff_list = []
    perc_inc_list = []
    print "total passing tracks: ", np.sum(passing_tracks), "total tracks:", np.sum(total_tracks)
    eff_before_cuts = analysis_utils.get_mean_efficiency(passing_tracks, total_tracks)
    eff_list.append(eff_before_cuts)
    perc_inc_list.append(100.)
    print 'mean eff before cuts: ', eff_before_cuts[0]
    # need to plot the eff and then the cicled pixels on top of it
    fig1 = Figure()
    _ = FigureCanvas(fig1)
    fig1.clear()
    fig1.patch.set_facecolor("white")
    ax1 = fig1.add_subplot(111)
#     ax1.set_title('Thresholds for each pixel')
#     ax1.set_xlabel('column')
#     ax1.set_ylabel('row')
    # ax1_main.xaxis.set_label_position('bottom')
    # ax1_main.xaxis.tick_bottom()
    cmap = plt.cm.viridis
    cmap.set_under(color='white')
    h1 = ax1.imshow(eff_data, origin='lower', interpolation='nearest', cmap=cmap, zorder=1, vmin=85.)
    cbar = fig1.colorbar(h1)
    cbar.ax.minorticks_on()
    tune_cords = np.where(tune_dis_mask == True)
    man_cords = np.where(man_dis_mask == True)
    occ_cords = np.where(occ_0_mask == True)
    # plot for the disabled pixels on the
    ax1.plot(tune_cords[0], tune_cords[1], "o", markeredgewidth=1, markeredgecolor='r',
             markerfacecolor='None', zorder=2, label="Disabled in Tuning")
#     ax1.plot(man_cords[0], man_cords[1], "p", markeredgewidth=2, markeredgecolor='k',
#              markerfacecolor='None', zorder=4, label="Manually Disabled after tuning")
    ax1.plot(occ_cords[0], occ_cords[1], "D", markeredgewidth=1, markeredgecolor='m',
             markerfacecolor='None', zorder=3, label="Un-tunable Pixels")
    ax1.legend()
    eff_data[(tune_cords[1], tune_cords[0])] = 0
    eff_data[(occ_cords[1], occ_cords[0])] = 0
    fig4 = Figure()
    _ = FigureCanvas(fig4)
    fig4.clear()
    fig4.patch.set_facecolor("white")
    ax4 = fig4.add_subplot(111)
    ax4.grid()

    eff_cut_hist, bins = np.histogram(eff_data[eff_data != 0], 100, range=(85, 100))
    bin_left = bins[:-1]
    ax4.set_title('Mean Efficiency: %s' % str(eff_before_cuts[0]))
    ax4.set_xlabel('Efficiency')
    ax4.set_ylabel('Number of pixels')
    ax4.bar(left=bin_left, height=eff_cut_hist, width=np.diff(bin_left)[0], align="edge")
    # h1 = ax3.imshow(eff_data, origin='lower', interpolation='none', cmap=cmap, zorder=1)
    ax4.set_yscale('log')

    fig2 = Figure()
    _ = FigureCanvas(fig2)
    fig2.clear()
    fig2.patch.set_facecolor("white")
    ax2 = fig2.add_subplot(111)

    # y is consecutive x is side to side in columns

    eff_low_w_surround = []
    eff_low_w_surround.append(np.append(occ_cords[0], tune_cords[0]))
    eff_low_w_surround.append(np.append(occ_cords[1], tune_cords[1]))

    # get 1st position
    # loop over x axis for one_side_pix_to_ignore_x
    # add pixels to disable on x axis
    # add the disabled pixels on the y axis

    for pix in range(eff_low_w_surround[0].shape[0]):
        x_plus_err_list = []
        x_plus_err_list.append(eff_low_w_surround[0][pix])
        y_plus_err_list = []
        y_plus_err_list.append(eff_low_w_surround[1][pix])
        # x loop
        for x_err in range(0, int(one_side_pix_to_ignore_x) + 1, 1):
            xplus_hold = eff_low_w_surround[0][pix] + x_err
            x_plus = False
            x_minus = False
            if xplus_hold < 64:
                np.append(eff_low_w_surround[0], xplus_hold)
                x_plus_err_list.append(xplus_hold)
                x_plus = True
            xminus_hold = eff_low_w_surround[0][pix] - x_err
            if xminus_hold >= 0:
                np.append(eff_low_w_surround[0], xminus_hold)
                x_plus_err_list.append(xminus_hold)
                x_minus = True

            # y loop
            for y_err in range(0, int(one_side_pix_to_ignore_y) + 1, 1):
                # need each y coord for -, 0, +
                yplus_hold = eff_low_w_surround[1][pix] + y_err
                if yplus_hold < 64:
                    y_plus_err_list.append(yplus_hold)
                    if x_plus:
                        eff_low_w_surround[0] = np.append(eff_low_w_surround[0], [xplus_hold])
                        eff_low_w_surround[1] = np.append(eff_low_w_surround[1], [yplus_hold])
                    if x_minus:
                        eff_low_w_surround[0] = np.append(eff_low_w_surround[0], [xminus_hold])
                        eff_low_w_surround[1] = np.append(eff_low_w_surround[1], [yplus_hold])
                    eff_low_w_surround[0] = np.append(eff_low_w_surround[0], [eff_low_w_surround[0][pix]])
                    eff_low_w_surround[1] = np.append(eff_low_w_surround[1], [yplus_hold])
                yminus_hold = eff_low_w_surround[1][pix] - y_err
                if yminus_hold >= 0:
                    y_plus_err_list.append(yminus_hold)
                    if x_plus:
                        eff_low_w_surround[0] = np.append(eff_low_w_surround[0], [xplus_hold])
                        eff_low_w_surround[1] = np.append(eff_low_w_surround[1], [yminus_hold])
                    if x_minus:
                        eff_low_w_surround[0] = np.append(eff_low_w_surround[0], [xminus_hold])
                        eff_low_w_surround[1] = np.append(eff_low_w_surround[1], [yminus_hold])
                    eff_low_w_surround[0] = np.append(eff_low_w_surround[0], [eff_low_w_surround[0][pix]])
                    eff_low_w_surround[1] = np.append(eff_low_w_surround[1], [yminus_hold])

    h1 = ax2.imshow(eff_data, origin='lower', interpolation='none', cmap=cmap, zorder=1, vmin=85)
    ax2.plot(eff_low_w_surround[0], eff_low_w_surround[1], "o", markeredgewidth=1, markeredgecolor='r', markerfacecolor='None', zorder=2)
    cbar = fig2.colorbar(h1)
    cbar.ax.minorticks_on()

    edges_list = []
    edges_list.append(range(0, 64, 1))
    edges_list.append([0] * 63)

    edges_list[1].extend(x for x in range(0, 64, 1))
    edges_list[0].extend([0] * 63)

    edges_list[0].extend(x for x in range(0, 64, 1))
    edges_list[1].extend([63] * 63)

    edges_list[1].extend(x for x in range(0, 64, 1))
    edges_list[0].extend([63] * 63)

    edges_list[0].extend(range(0, 64, 1))
    edges_list[1].extend([1] * 63)

    edges_list[1].extend(x for x in range(0, 64, 1))
    edges_list[0].extend([1] * 63)

    edges_list[0].extend(x for x in range(0, 64, 1))
    edges_list[1].extend([62] * 63)

    edges_list[1].extend(x for x in range(0, 64, 1))
    edges_list[0].extend([62] * 63)

    eff_mask = np.full((64, 64), True, np.bool)
    eff_mask[(eff_low_w_surround[0], eff_low_w_surround[1])] = False

    eff_mask[(edges_list[0], edges_list[1])] = False

#     fe7_mask = np.full(4096, True, np.bool)
#     fe7_mask[4095 - (4096 / 8):] = False
#     fe7_mask = np.reshape(fe7_mask, (64, 64))
#     eff_data[fe7_mask == False] = 0
    eff_data[(edges_list[0], edges_list[1])] = 0
    eff_data[(eff_low_w_surround[0], eff_low_w_surround[1])] = 0
    eff_avg = analysis_utils.get_mean_efficiency(passing_tracks[eff_mask == True], total_tracks[eff_mask == True])
    eff_list.append(eff_avg)
    # create eff for pixels not included in the cuts
    fig3 = Figure()
    _ = FigureCanvas(fig3)
    fig3.clear()
    fig3.patch.set_facecolor("white")
    ax3 = fig3.add_subplot(111)
    ax3.grid()

    # print type(eff_data)
    # print eff_data.shape
    #eff_data[(eff_low_w_surround[1], eff_low_w_surround[0])] = 0
    print "cuts passing tracks: ", sum(passing_tracks[eff_data != 0]), "total tracks:", sum(total_tracks[eff_data != 0])
    print 'mask passing', sum(passing_tracks[eff_mask == True]), "total tracks:", sum(total_tracks[eff_mask == True])
#     print analysis_utils.get_mean_efficiency(passing_tracks[eff_data != 0], total_tracks[eff_data != 0])
    print "eff_after cuts", eff_avg[0]
    # int((100 - round(min(eff_data[eff_data != 0]) - 1, 0)) * 3)
    eff_cut_hist, bins = np.histogram(eff_data, 100, range=(85, 100))
    bin_left = bins[:-1]
    perc = len(np.nonzero(eff_data)[0]) / 4096. * 100
    perc_inc_list.append(perc)
    ax3.set_title('Percent Included: %s Mean Efficiency after cuts: %s' %
                  (round(perc, 5), str(round(eff_avg[0] * 100, 5))))
    ax3.set_xlabel('Efficiency')
    ax3.set_ylabel('Number of pixels')
    ax3.bar(left=bin_left, height=eff_cut_hist, width=np.diff(bin_left)[0], align="edge")
    ax3.set_yscale('log')
    # eff_data_cut = eff_data[][]

    return fig1, fig4, fig2, fig3, eff_data, eff_list, perc_inc_list


def eff_pixel_flavor_per_type(eff_data, type_cords_list, flav_name, eff_h5_file, type_cords_list_2=None, flav_name_2=None):
    # print "for combi, type_cords_list_2 is for the readout flavor, name should be assined accordingly"
    with tb.open_file(eff_h5_file, 'r+') as in_file:
        total_tracks = in_file.root.DUT_1.Total_tracks[:]
        passing_tracks = in_file.root.DUT_1.Passing_tracks[:]

    eff_mask = np.full((64, 64), False, np.bool)

    out_list = []
    out_list_x = []
    out_list_y = []
    for x in range(type_cords_list[0][0], type_cords_list[1][0] + 1):
        for y in range(type_cords_list[0][1], type_cords_list[1][1] + 1):
            out_list_x.append(x)
            out_list_y.append(y)
    out_list.append(out_list_x)
    out_list.append(out_list_y)

    eff_mask[(out_list[0], out_list[1])] = True

    if type_cords_list_2:
        eff_mask2 = np.full((64, 64), False, np.bool)
        out_list2 = []
        out_list2_x = []
        out_list2_y = []
        for x in range(type_cords_list_2[0][0], type_cords_list_2[1][0] + 1):
            for y in range(type_cords_list_2[0][1], type_cords_list_2[1][1] + 1):
                out_list2_x.append(x)
                out_list2_y.append(y)
        out_list2.append(out_list2_x)
        out_list2.append(out_list2_y)
        eff_mask2[(out_list2[0], out_list2[1])] = True

        true_coords = np.where((eff_mask == True) & (eff_mask2 == True))

        out_eff = eff_data[true_coords[0], true_coords[1]]
        passing_tracks_in_sel = passing_tracks[true_coords[0], true_coords[1]]
        total_tracks_in_sel = total_tracks[true_coords[0], true_coords[1]]
        print flav_name, flav_name_2
    else:
        out_eff = eff_data[out_list[0], out_list[1]]
        passing_tracks_in_sel = passing_tracks[out_list[0], out_list[1]]
        total_tracks_in_sel = total_tracks[out_list[0], out_list[1]]
        print flav_name

    num_pix = len(np.nonzero(out_eff)[0])
    print "passing tracks: ", sum(passing_tracks_in_sel[out_eff != 0]), "total tracks:",  sum(total_tracks_in_sel[out_eff != 0])

    eff_avg = analysis_utils.get_mean_efficiency(passing_tracks_in_sel[out_eff != 0], total_tracks_in_sel[out_eff != 0])

    fig = Figure()
    _ = FigureCanvas(fig)
    fig.clear()
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)

    eff_cut_hist, bins = np.histogram(out_eff, 105, range=(90, 100.5))
    bin_left = bins[:-1]
    print eff_avg[0]
    perc = (num_pix / float(out_eff.shape[0])) * 100
    if flav_name_2:
        ax.set_title("Pixel Flavor: %s, Column Flavor %s\nPercent Included: %s Efficiency: %s" %
                     (flav_name, flav_name_2, str(round(perc, 5)), str(round(eff_avg[0] * 100, 5))))
    else:
        ax.set_title("Flavor: %s Percent Included: %s Efficiency: %s" %
                     (flav_name, str(round(perc, 5)), str(round(eff_avg[0] * 100, 5))))
    ax.set_xlabel('Efficiency')
    ax.set_ylabel('Number of pixels')
    ax.bar(left=bin_left, height=eff_cut_hist, width=np.diff(bin_left)[0], align="edge")
    ax.set_yscale('log')
    ax.grid()

    return fig, eff_avg, perc


def get_hit_data(fe65p2_h5_file, fe65p2_h5_file_anal=None):
    with tb.open_file(fe65p2_h5_file, 'r+') as in_file_h5:
        meta_data = in_file_h5.root.meta_data[:]
        raw_data = in_file_h5.root.raw_data[:]

        fe65 = fe65p2.fe65p2()

        try:
            hit_data = in_file_h5.root.hit_data[:]
        except:
            hit_data = fe65.interpret_raw_data_tlu(raw_data, meta_data)
            in_file_h5.create_table(in_file_h5.root, 'hit_data', hit_data)
            if fe65p2_h5_file_anal == None:
                return
        try:
            hit_table = in_file_h5.root.Hits[:]
        except:
            hit_table = create_hit_table_for_testbeam_analysis(fe65p2_h5_file)
    with tb.open_file(fe65p2_h5_file_anal, 'w') as out_file_h5:

        out_file_h5.create_table(out_file_h5.root, 'Hits', hit_table)


def create_hit_table_for_testbeam_analysis(h5_file):
        # make a version of the hit data for test_beam_analysis
        # go from bcid, col, row, tot, lv1id, scan_param, trig_id
        # to event_number, frame, column, row, charge
        # col -> column
        # row -> row
        # tot -> charge
        # trig_id -> event_number
        # lv1id -> frame
    with tb.open_file(h5_file, 'r+') as in_file_h5:
        hit_data = in_file_h5.root.hit_data[:]

        data_type = {'names': ['event_number', 'frame', 'column', 'row', 'charge'],
                     'formats': ['int64', 'uint8', 'uint16', 'uint16', 'uint16']}
        hit_table = np.recarray(hit_data.shape, dtype=data_type)
        hit_table['event_number'] = hit_data['trig_id'].astype(np.int64)
        hit_table['frame'] = hit_data['lv1id']
        hit_table['column'] = hit_data['col'] + 1
        hit_table['row'] = hit_data['row'] + 1
        hit_table['charge'] = hit_data['tot']
        in_file_h5.create_table(in_file_h5.root, 'Hits', hit_table)
    return hit_table


def create_hit_table_for_testbeam_analysis_fei4(h5_file, out_h5_file):
    with tb.open_file(h5_file, 'r+') as in_file_h5:
        Hits = in_file_h5.root.Hits[:]

        data_type = {'names': ['event_number', 'frame', 'column', 'row', 'charge'],
                     'formats': ['uint32', 'uint8', 'uint8', 'uint8', 'uint8']}
        hit_table = np.recarray(Hits.shape, dtype=data_type)
        hit_table['event_number'] = Hits['event_number']
        hit_table['frame'] = Hits['relative_BCID']
        hit_table['column'] = Hits['column']
        hit_table['row'] = Hits['row']
        hit_table['charge'] = Hits['tot']

    with tb.open_file(out_h5_file, 'r+') as out_file_h5:
        out_file_h5.create_table(out_file_h5.root, 'Hits', hit_table)

#------------------------------------------------------------------------------


def flavor_controller(folder_name=None):
    chdir("/media/daniel/Maxtor/fe65p2_testbeam_april_2018/" + folder_name)
    fe65p2_h5_file = glob.glob('*_tlu_test_scan.h5')
    noisy_h5 = glob.glob('*_tlu_test_scan_anal_noisy_pixel_mask.h5')
    eff_h5_file = "/media/daniel/Maxtor/fe65p2_testbeam_april_2018/" + folder_name + "/analyzed/Efficiency.h5"
    res_file = "/media/daniel/Maxtor/fe65p2_testbeam_april_2018/" + folder_name + "/analyzed/Residuals_prealigned.h5"

#     edges = [[0, 0],
#              [0, 63],
#              [63, 0],
#              [63, 63],
#              [0, 0],
#              [63, 0],
#              [0, 63],
#              [63, 63]]

    nw15 = [[1, 1], [31, 8]]
    nw20 = [[1, 9], [31, 16]]
    nw25 = [[1, 17], [31, 24]]
    nw30 = [[1, 25], [31, 62]]

    dnw15 = [[32, 1], [62, 8]]
    dnw20 = [[32, 9], [62, 16]]
    dnw25 = [[32, 17], [62, 24]]
    dnw30 = [[32, 25], [62, 62]]

    fe0 = [[0, 0], [7, 63]]
    fe1 = [[8, 0], [15, 63]]
    fe2 = [[16, 0], [23, 63]]
    fe3 = [[24, 0], [31, 63]]
    fe4 = [[32, 0], [39, 63]]
    fe5 = [[40, 0], [47, 63]]
    fe6 = [[48, 0], [55, 63]]
    fe7 = [[56, 0], [63, 63]]

    type_name_list = ['nw15', 'nw20', 'nw25', 'nw30', 'dnw15', 'dnw20', 'dnw25', 'dnw30',
                      'CPL000', 'CPL001', 'CPL011', 'CPL001RH', 'CPL100', 'CPL101', 'CPL101RH', 'CPL101']
    type_list = [nw15, nw20, nw25, nw30, dnw15, dnw20, dnw25, dnw30, fe0, fe1, fe2, fe3, fe4, fe5, fe6, fe7]

    fe_name_list = ['CPL000', 'CPL001', 'CPL011', 'CPL001RH', 'CPL100', 'CPL101', 'CPL101RH', 'CPL101']
    fe_list = [fe0, fe1, fe2, fe3, fe4, fe5, fe6, fe7]
    pixel_flav_list = ['nw15', 'nw20', 'nw25', 'nw30', 'dnw15', 'dnw20', 'dnw25', 'dnw30']
    pixel_list = [nw15, nw20, nw25, nw30, dnw15, dnw20, dnw25, dnw30]

#     try:
    overlay_hm, first_cuts_hist, fig2, fig3, eff_data, eff_list, perc_inc_list = overlay_eff_hm_w_disabled_pix(fe65p2_h5_file=fe65p2_h5_file[0],
                                                                                                               eff_h5_file=eff_h5_file,
                                                                                                               res_file=res_file,
                                                                                                               noisy_h5=noisy_h5[0])
#     except Exception as e:
#         print e  # if e == 'group ``/`` does not have a child named ``hit_data``':
#         get_hit_data(fe65p2_h5_file[0])
#         overlay_hm, first_cuts_hist, fig2, fig3, eff_data, eff_list, perc_inc_list = overlay_eff_hm_w_disabled_pix(fe65p2_h5_file=fe65p2_h5_file[0],
#                                                                                                                    eff_h5_file=eff_h5_file,
#                                                                                                                    res_file=res_file,
#                                                                                                                    noisy_h5=noisy_h5[0])
    pdfName = "/media/daniel/Maxtor/fe65p2_testbeam_april_2018/" + folder_name + '/eff_w_cuts_by_flavor.pdf'
    pp = PdfPages(pdfName)
    pp.savefig(overlay_hm, layout='tight')
    plt.clf()
    pp.savefig(first_cuts_hist, layout='tight')
    plt.clf()
    pp.savefig(fig2, layout='tight')
    plt.clf()
    pp.savefig(fig3, layout='tight')

    flav_eff_list = []
    flav_perc_inc_list = []
    for num, flav in enumerate(type_name_list):
        try:
            fig, eff, perc = eff_pixel_flavor_per_type(eff_data, type_list[num], flav, eff_h5_file=eff_h5_file)
            flav_eff_list.append(eff)
            flav_perc_inc_list.append(perc)
            plt.clf()
            pp.savefig(fig, layout='tight')
        except:
            print 'failed to make efficiency plots for ', flav

    for num_pix in range(4):
        for num_col in range(4):
            try:
                fig, eff, perc = eff_pixel_flavor_per_type(eff_data, type_cords_list=pixel_list[num_pix], flav_name=pixel_flav_list[num_pix], eff_h5_file=eff_h5_file,
                                                           type_cords_list_2=fe_list[num_col], flav_name_2=fe_name_list[num_col])
                flav_eff_list.append(eff)
                flav_perc_inc_list.append(perc)
                plt.clf()
                pp.savefig(fig, layout='tight')
            except:
                print "failed to make efficiency plots for ", num_col, num_pix

    for num_pix in range(4, 8):
        for num_col in range(4, 8):
            try:
                fig, eff, perc = eff_pixel_flavor_per_type(eff_data, type_cords_list=pixel_list[num_pix], flav_name=pixel_flav_list[num_pix], eff_h5_file=eff_h5_file,
                                                           type_cords_list_2=fe_list[num_col], flav_name_2=fe_name_list[num_col])
                flav_eff_list.append(eff)
                flav_perc_inc_list.append(perc)
                plt.clf()
                pp.savefig(fig, layout='tight')
            except:
                print "failed to make efficiency plots for ", num_col, num_pix

    pp.close()
    return eff_list, perc_inc_list, flav_eff_list, flav_perc_inc_list


if __name__ == "__main__":

    names = ['total', 'cuts', 'nw15', 'nw20', 'nw25', 'nw30',
             'dnw15', 'dnw20', 'dnw25', 'dnw30', 'fe0', 'fe1', 'fe2', 'fe3', 'fe4', 'fe5', 'fe6', 'fe7',
             'nw15_fe0', 'nw15_fe1', 'nw15_fe2', 'nw15_fe3',
             'nw20_fe0', 'nw20_fe1', 'nw20_fe2', 'nw20_fe3',
             'nw25_fe0', 'nw25_fe1', 'nw25_fe2', 'nw25_fe3',
             'nw30_fe0', 'nw30_fe1', 'nw30_fe2', 'nw30_fe3',
             'dnw15_fe4', 'dnw15_fe5', 'dnw15_fe6', 'dnw15_fe7',
             'dnw20_fe4', 'dnw20_fe5', 'dnw20_fe6', 'dnw20_fe7',
             'dnw25_fe4', 'dnw25_fe5', 'dnw25_fe6', 'dnw25_fe7',
             'dnw30_fe4', 'dnw30_fe5', 'dnw30_fe6', 'dnw30_fe7']

    names2 = ['total', 'cuts', 'nw15', 'nw20', 'nw25', 'nw30',
              'dnw15', 'dnw20', 'dnw25', 'dnw30', 'fe0', 'fe1', 'fe2', 'fe3', 'fe4', 'fe5', 'fe6',
              'nw15_fe0', 'nw15_fe1', 'nw15_fe2', 'nw15_fe3',
              'nw20_fe0', 'nw20_fe1', 'nw20_fe2', 'nw20_fe3',
              'nw25_fe0', 'nw25_fe1', 'nw25_fe2', 'nw25_fe3',
              'nw30_fe0', 'nw30_fe1', 'nw30_fe2', 'nw30_fe3',
              'dnw15_fe4', 'dnw15_fe5', 'dnw15_fe6',
              'dnw20_fe4', 'dnw20_fe5', 'dnw20_fe6',
              'dnw25_fe4', 'dnw25_fe5', 'dnw25_fe6',
              'dnw30_fe4', 'dnw30_fe5', 'dnw30_fe6', ]
    dict = {}
    dict['run_num'] = tb.UInt8Col(pos=0)
    dict['bias'] = tb.Int16Col(pos=1)
    dict['vth1'] = tb.UInt8Col(pos=2)
    i = 3
    for name in names:
        dict[name] = tb.Float64Col(pos=i)
        dict[name + '_errp'] = tb.Float64Col(pos=i + 1)
        dict[name + '_errm'] = tb.Float64Col(pos=i + 2)
        dict['perc_inc_' + name] = tb.Float64Col(pos=i + 3)
        i += 4
    #fe65p2_h5_file = '/home/daniel/MasterThesis/test_beam_data/run23/20180424_222201_tlu_test_scan.h5'
    #fe65p2_h5_file_anal = '/home/daniel/MasterThesis/test_beam_data/run23/20180424_222201_tlu_test_scan_anal.h5'
    #get_hit_data(fe65p2_h5_file=fe65p2_h5_file, fe65p2_h5_file_anal=fe65p2_h5_file_anal)

    run_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]

    bias_list = [149, 149, 150, 100, 76, 76, 25, 1, 13, 13, 50, 100, 100, 100, 100, 100, 100,
                 100, 35, 35, 35, 35, 35, 65, 65, 65, 65, 65, 75, 75, 175, 175, 175, 175, 140, 140, 15]

    vth1_list = [43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 63, 83, 103, 103, 123, 143, 173,
                 173, 203, 143, 93, 33, 33, 93, 143, 203, 103, 53, 53, 153, 103, 203, 53, 153, 103, 103]

#     chdir("/media/daniel/Maxtor/fe65p2_testbeam_april_2018/")
    with tb.open_file('/media/daniel/Maxtor/fe65p2_testbeam_april_2018/efficiency_over_all_runs.h5', 'w') as out_file_h5:
        eff_table = out_file_h5.create_table(out_file_h5.root, name='eff_table', description=dict, title='eff_table')
        for x, num in enumerate(run_list):
            #     for num in [5]:
            folder_name = str("run") + str(num)
            print folder_name, run_list[x], bias_list[x], vth1_list[x]
            eff_list, perc_inc_list, flav_eff_list, flav_perc_inc_list = flavor_controller(folder_name)

            eff_table.row['run_num'] = run_list[x]
            eff_table.row['bias'] = bias_list[x] * -1
            eff_table.row['vth1'] = vth1_list[x]

            eff_table.row['total'] = eff_list[0][0] * 100
            eff_table.row['total_errm'] = eff_list[0][1] * 100
            eff_table.row['total_errp'] = eff_list[0][2] * 100
            eff_table.row['perc_inc_total'] = perc_inc_list[0]
            eff_table.row['cuts'] = eff_list[1][0] * 100
            eff_table.row['cuts_errm'] = eff_list[1][1] * 100
            eff_table.row['cuts_errp'] = eff_list[1][2] * 100
            eff_table.row['perc_inc_cuts'] = perc_inc_list[1]

            j = 0
            for i in names[2:]:
                eff_table.row[i] = flav_eff_list[j][0] * 100
                eff_table.row[i + '_errm'] = flav_eff_list[j][1] * 100
                eff_table.row[i + '_errp'] = flav_eff_list[j][2] * 100
                eff_table.row['perc_inc_' + i] = flav_perc_inc_list[j]
                j += 1

            eff_table.row.append()
            eff_table.flush()

    print "finished without errors"
