import os
import inspect
import logging
import glob
import re
import numpy as np
import tables as tb
# import fe65p2.test_beam_analysis as fe65p2_converter

from os import getcwd, chdir

run_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
# for num in [26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 43, 44]:  # range(38, 45):
total_events = 0
multi_clut_events = 0
out_list = []
noise_list = []
with tb.open_file('/media/daniel/Maxtor/fe65p2_testbeam_april_2018/efficiency_over_all_runs_wo7_new_sections_500.h5', 'r') as eff_h5:
    eff_table = eff_h5.root.eff_table[:]
for num in run_list:
    folder_name = str("run") + str(num)
#     print folder_name

    chdir("/media/daniel/Maxtor/fe65p2_testbeam_april_2018/" + folder_name)
    fe65_file = glob.glob('*_tlu_test_scan.h5')

    run_eff = eff_table[eff_table['run_num'] == num]['cuts']
#     print 100 - run_eff

    with tb.open_file(fe65_file[0], 'r+') as io_file_h5:
        hit_data = io_file_h5.root.hit_data[:]

    mask_file = glob.glob('*tlu_test_scan_anal_noisy_pixel_mask.h5')

    with tb.open_file(mask_file[0], 'r+') as io_file_h5:
        mask = io_file_h5.root.NoisyPixelMask[:]
        where = np.where(mask == True)
        for x in where[0]:
            hit_data[hit_data['col'] == x] = -2
        for x in where[1]:
            hit_data[hit_data['row'] == x] = -2
    noise_prob = float(hit_data[hit_data['lv1id'] < 8].shape[0]) * 2 / float(hit_data.shape[0])
    noise_list.append(noise_prob)

#     with tb.open_file(fe65_file[0], 'r') as h5:
#         clusters = h5.root.Cluster[:]
#
#         run_events = clusters[clusters['n_hits'] <= 4].shape[0]
#         total_events += run_events
#         run_mult_events = run_events - np.unique(clusters[clusters['n_hits'] <= 4]['event_number']).shape[0]
#         multi_clut_events += run_mult_events

    p_out = (1 - (run_eff[0] / 100.)) * (noise_prob)
    out_list.append(p_out)
    print 'run', num, noise_prob, 'P_out (%):', p_out * 100


print
print min(noise_list) * 100, max(noise_list) * 100, np.mean(noise_list) * 100
print 'P(Ebar n Noise):', np.mean(out_list), np.std(out_list), min(out_list), max(out_list)
