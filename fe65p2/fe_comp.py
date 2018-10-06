#------------------------------------------------------------------------------
# want to get the average frontend readout for each fun for the readouts
#
#
#------------------------------------------------------------------------------

import numpy as np
import tables as tb
import matplotlib.pyplot as plt


h5_file_w7 = '/media/daniel/Maxtor/fe65p2_testbeam_april_2018/efficiency_over_all_runs_w7_new_sections_500.h5'
h5_file_wo7 = '/media/daniel/Maxtor/fe65p2_testbeam_april_2018/efficiency_over_all_runs_wo7_new_sections_500.h5'

nw_fes_w7 = []
dnw_fes_w7 = []
nw_fes_wo7 = []
dnw_fes_wo7 = []

fe_eff_w7 = []
fe_eff_wo7 = []


with tb.open_file(h5_file_w7, 'r+') as eff_file:
    eff_table_w7 = eff_file.root.eff_table[:]
    bias_tab = eff_table_w7[(eff_table_w7['vth1'] == 43)]  # (eff_table_w7['bias'] <= -100) &

    fe2 = eff_table_w7['fe2']
    fe5 = eff_table_w7['fe6']
    print np.average(fe2 - fe5), np.std(fe2 - fe5)

#     for fl in [15, 20, 25, 30]:
#         plt.plot(bias_tab['bias'], bias_tab['nw' + str(fl) + '_fe2'], 'o')
#         plt.plot(bias_tab['bias'], bias_tab['dnw' + str(fl) + '_fe5'], 'o')
#         plt.show()

# for run in eff_table_w7['run_num']:
#     nw_hold = []
#     eff_list = []
#     for j in range(4):
#         nw_hold.append(eff_table_w7[eff_table_w7['run_num'] == run]["fe" + str(j)][0])
#     nw_fes_w7.append(nw_hold)
#
#     dnw_hold = []
#     for k in range(4, 7):
#         dnw_hold.append(eff_table_w7[eff_table_w7['run_num'] == run]["fe" + str(j)][0])
# #     print dnw_hold
#     dnw_fes_w7.append(dnw_hold)
#
#     for x in range(8):
#         eff_list.append(eff_table_w7[eff_table_w7['run_num'] == run]["fe" + str(x)][0])
#     fe_eff_w7.append([np.mean(eff_list), np.std(eff_list)])
#
# with tb.open_file(h5_file_wo7, 'r+') as eff_file2:
#     eff_table_wo7 = eff_file2.root.eff_table[:]
#
# for run in eff_table_wo7['run_num']:
#     nw_hold = []
#     eff_list = []
#     for j in range(4):
#         nw_hold.append(eff_table_wo7[eff_table_wo7['run_num'] == run]["fe" + str(j)][0])
#     nw_fes_wo7.append(nw_hold)
#
#     dnw_hold = []
#     for k in range(4, 7):
#         dnw_hold.append(eff_table_wo7[eff_table_wo7['run_num'] == run]["fe" + str(j)][0])
#     dnw_fes_wo7.append(dnw_hold)
#
#     for x in range(7):
#         eff_list.append(eff_table_wo7[eff_table_wo7['run_num'] == run]["fe" + str(x)][0])
#     fe_eff_wo7.append([np.mean(eff_list), np.std(eff_list)])
#
# mu = []
# sig = []
# for l in range(len(fe_eff_w7)):
#     mu.append(fe_eff_wo7[l][0] - fe_eff_w7[l][0])
#     sig.append(fe_eff_wo7[l][1] - fe_eff_w7[l][1])
#     print "wo7-w7 mu:", fe_eff_wo7[l][0]
#     print "wo7-w7 sig:", fe_eff_wo7[l][1] - fe_eff_w7[l][1]
#
# print "average mu diff:", np.mean(mu)
# print "average sig diff:", np.mean(sig)
#
# nw_avg_w7 = []
# dnw_avg_w7 = []
# nw_avg_wo7 = []
# dnw_avg_wo7 = []
# nw_std_wo7 = []
# dnw_std_wo7 = []
#
# for inx in range(len(dnw_fes_wo7)):
#     nw_avg_w7.append(np.mean(nw_fes_w7[inx]))
#     dnw_avg_w7.append(np.mean(dnw_fes_w7[inx]))
#     nw_avg_wo7.append(np.mean(nw_fes_wo7[inx]))
#     dnw_avg_wo7.append(np.mean(dnw_fes_wo7[inx]))
#     nw_std_wo7.append(np.std(nw_fes_wo7[inx]))
#     dnw_std_wo7.append(np.std(dnw_fes_wo7[inx]))
#
# #     print "w7\t", np.mean(nw_fes_w7[inx]), np.mean(dnw_fes_w7[inx]), "\t", np.mean(nw_fes_w7[inx]) - np.mean(dnw_fes_w7[inx])
#     print "wo7\t", np.std(nw_fes_wo7[inx]), np.std(dnw_fes_wo7[inx])  # , "\t", np.mean(nw_fes_wo7[inx]) - np.mean(dnw_fes_wo7[inx])
#     print
#
# print "nw-dnw avgs", np.mean(np.array(nw_avg_wo7) - np.array(dnw_avg_wo7)), np.std(np.array(nw_avg_wo7) - np.array(dnw_avg_wo7))
