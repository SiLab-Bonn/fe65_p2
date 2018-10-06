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


eff0_list = []
eff0_err_list = []
eff2_list = []
eff2_err_list = []
eff3_list = []
eff3_err_list = []

for num in run_list:
    folder_name = str("run") + str(num)
#     print folder_name
#
#     chdir("/media/daniel/Maxtor/fe65p2_testbe am_april_2018/" + folder_name + "/analyzed")
    eff_file = '/media/daniel/Maxtor/fe65p2_testbeam_april_2018/' + folder_name + '/analyzed/Efficiency.h5'

    with tb.open_file(eff_file, 'r+') as in_file_h5:
        eff = in_file_h5.root.DUT_0.Efficiency[:]
        if np.mean(eff[eff != 0]) > 98:
            print num, np.mean(eff[eff != 0])
            eff0_list.append(np.mean(eff[eff != 0]))
            eff0_err_list.append(np.std(eff[eff != 0]))

#             with tb.open_file(eff_file, 'r+') as in_file_h5:
            eff = in_file_h5.root.DUT_2.Efficiency[:]
            eff2_list.append(np.mean(eff[eff != 0]))
            eff2_err_list.append(np.std(eff[eff != 0]))

#             with tb.open_file(eff_file) as in_file_h5:
            eff = in_file_h5.root.DUT_3.Efficiency[:]
            eff3_list.append(np.mean(eff[eff != 0]))
            eff3_err_list.append(np.std(eff[eff != 0]))

print np.mean(eff0_list), np.std(eff0_list), np.mean(eff0_err_list)
print np.mean(eff2_list), np.std(eff2_list), np.mean(eff2_err_list)
print np.mean(eff3_list), np.std(eff3_list), np.mean(eff3_err_list)
