import yaml
import os
import inspect
import glob
import numpy as np
import tables as tb
# import fe65p2.test_beam_analysis as fe65p2_converter

from os import getcwd, chdir

run_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
# for num in [26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 43, 44]:  # range(38, 45):


row_res_list = []
col_res_list = []
with tb.open_file('/media/daniel/Maxtor/fe65p2_testbeam_april_2018/efficiency_over_all_runs_wo7_new_sections_500.h5', 'r') as eff_h5:
    eff_table = eff_h5.root.eff_table[:]
for num in run_list:
    folder_name = str("run") + str(num)
#     print folder_name

#     chdir("/media/daniel/Maxtor/fe65p2_testbe am_april_2018/" + folder_name + "/analyzed")
#     fe65_file = glob.glob('*_tlu_test_scan.h5')

#     run_eff = eff_table[eff_table['run_num'] == num]['cuts']
# #     print 100 - run_eff
#
#     with tb.open_file(fe65_file[0], 'r+') as io_file_h5:
#         hit_data = io_file_h5.root.hit_data[:]

    res_file = '/media/daniel/Maxtor/fe65p2_testbeam_april_2018/' + folder_name + '/analyzed/Residuals_prealigned.h5'
    with tb.open_file(res_file, 'r+') as in_file_h5:
        row_fits = in_file_h5.root.ResidualsRow_DUT1.attrs.fit_coeff
        col_fits = in_file_h5.root.ResidualsCol_DUT1.attrs.fit_coeff

        row_res_list.append(row_fits[2])
        col_res_list.append(col_fits[2])
        if col_fits[2] > 100:
            print num, col_fits[2]

print "rows:", np.mean(row_res_list), np.std(row_res_list), min(row_res_list), max(row_res_list)
print "cols:", np.mean(col_res_list), np.std(col_res_list), min(col_res_list), max(col_res_list)