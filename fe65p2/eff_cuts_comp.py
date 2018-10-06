import numpy as np
import tables as tb


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
    cuts_perc = eff_table_w7['perc_inc_cuts']

    print "%after cuts mean, std, max, min", np.mean(cuts_perc), np.std(cuts_perc), max(cuts_perc), min(cuts_perc)

    cuts = eff_table_w7['cuts']

    print "eff after cuts[%] mean, std, max, min", np.mean(cuts), np.std(cuts), max(cuts), min(cuts)

    eff = eff_table_w7['total']

    print '\n', 'diff between before and after cuts'
    print np.mean(cuts - eff), np.std(cuts - eff), max(cuts - eff), min(cuts - eff)
