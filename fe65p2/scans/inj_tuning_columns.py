'''
this script will run a vth1_scan then a tdac_tuning scan to find the global threshold then the tdac threshold for each flavor
this is very similar to the noise_tuning_columns scan

Created by Daniel Coquelin on 30/1/2018
'''
from fe65p2.scans.vth1_scan import Vth1Scan
import fe65p2.DGC_plotting as DGC_plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fe65p2.scans.tdac_tuning import TDACScan
import numpy as np
import tables as tb
import logging
import time
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip3.yaml'

local_configuration = {
    #   DAC parameters
    # chip 3
    #     "PrmpVbpDac": 160,
    #     "vthin1Dac": 60,
    #     "vthin2Dac": 0,
    #     "vffDac": 80,
    #     "PrmpVbnFolDac": 87,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 50,
    #     "preCompVbnDac": 86,

    # chip 4
    #     "PrmpVbpDac": 170,
    #     "vthin1Dac": 100,
    #     "vthin2Dac": 0,
    #     "vffDac": 86,
    #     "PrmpVbnFolDac": 91,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 42,
    #     "preCompVbnDac": 90,

    "mask_steps": 6,
    "repeat_command": 50,
    "inj_electrons": 1000,
    # bare chip mask: '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180115_174703_noise_tuning.h5',
    "mask_filename": '',  # '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180116_091958_noise_tuning.h5',
    "TDAC": 16
}


def vth1_sc(qcols):
    logging.info("Starting Vth1 Scan")
    global_thresh_scan = Vth1Scan()
    out_file = str(global_thresh_scan.output_filename) + '.h5'

    custom_conf = {
        "quad_columns": qcols,
        "scan_range": [25, 120, 2],  # this is the vth1 scan range
    }

    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan_conf = dict(local_configuration, **custom_conf)
    global_thresh_scan.start(**scan_conf)
    final_vth1 = global_thresh_scan.analyze()
    print final_vth1
    global_thresh_scan.dut.close()
    return out_file, final_vth1


def tdac_sc(qcols, vth1):
    logging.info("starting TDAC scan")
    local_tdac_scan = TDACScan()
    out_file = str(local_tdac_scan.output_filename) + '.h5'

    custom_conf = {
        "quad_columns": qcols,
        "vth1_from_scan": vth1 + 7,
        "scan_range": [0, 32, 1],
    }
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan_conf = dict(local_configuration, **custom_conf)
    local_tdac_scan.start(**scan_conf)
    local_tdac_scan.analyze()
#     print noise_sc.vth1Dac
    local_tdac_scan.dut.close()
    return out_file


def combine_prev_scans(file0, file1, file2, file3, file4, file5, file6, file7):
    # loop over the files like "file"+str(i)
    file_list = [file0, file1, file2, file3, file4, file5, file6, file7]
    vth1_list = []
    for j, filename in enumerate(file_list):

        with tb.open_file(filename, 'r+') as in_file:
            mask_en_hold = in_file.root.analysis_results.en_mask[:]
            mask_tdac_hold = in_file.root.analysis_results.tdac_mask[:]
            dac_status = yaml.load(in_file.root.meta_data.attrs.dac_status)
            vth1 = dac_status['vthin1Dac']
            vth1_list.append(vth1)
        if filename == file0:
            mask_tdac = np.delete(mask_tdac_hold, np.s_[8:], axis=0)
            mask_en = np.delete(mask_en_hold, np.s_[8:], axis=0)

        else:
            mask_en_hold1 = np.delete(mask_en_hold, np.s_[(j + 1) * 8:], axis=0)
            mask_en_hold2 = np.delete(mask_en_hold1, np.s_[0:(j * 8)], axis=0)
            mask_tdac_hold1 = np.delete(mask_tdac_hold, np.s_[(j + 1) * 8:], axis=0)
            mask_tdac_hold2 = np.delete(mask_tdac_hold1, np.s_[0:(j * 8)], axis=0)

            mask_tdac = np.concatenate((mask_tdac, mask_tdac_hold2), axis=0)
            mask_en = np.concatenate((mask_en, mask_en_hold2), axis=0)
#     print np.mean(vth1_list)
    avg_vth1 = np.mean(vth1_list) + 30
    return mask_en, mask_tdac.astype(int), max(vth1_list)


if __name__ == "__main__":
    # need to cycle over all of the columns
    # output_file = "/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/" + str(time.clock()) + "inj_tuning_by_flavor" + ".h5"
    #mask_en = np.full([64, 64], False, dtype=np.bool)
    #mask_tdac = np.full([64, 64], 0, dtype=np.uint8)

    tdac_cols = {}
    vth1_dict = {}

    qcolumns = [False] * 16
    for i in range(0, 8, 1):
        qcolumns = [False] * (i * 2) + [True] * 2 + [False] * ((7 - i) * 2)
        print qcolumns
        in_file_vth1, vth1 = vth1_sc(qcolumns)
        in_file_tdac = tdac_sc(qcolumns, vth1)
        with tb.open_file(str(in_file_tdac), 'r') as in_file_h5:
            # saves all in a dictionary numbered as col_0 -> col_14 in intervals of 2
            mask_tdac_hold = in_file_h5.root.analysis_results.tdac_mask[:]
            mask_en_hold = in_file_h5.root.analysis_results.en_mask[:]

        # delete the last columns of the en and tdac masks, then delete the first columns
        mask_en_hold = np.delete(mask_en_hold, np.s_[(i + 1) * 8::], axis=1)
        mask_tdac_hold = np.delete(mask_tdac_hold, np.s_[(i + 1) * 8::], axis=1)
        if i == 0:
            mask_tdac = mask_tdac_hold
            mask_en = mask_en_hold

        if i != 0:
            mask_en_hold = np.delete(mask_en_hold, np.s_[0:(i * 8) + 7], axis=1)
            mask_tdac_hold = np.delete(mask_tdac_hold, np.s_[0:(i * 8) + 7], axis=1)

            mask_tdac = np.concatenate((mask_tdac, mask_tdac_hold), axis=1)
            mask_en = np.concatenate((mask_en, mask_en_hold), axis=1)
        vth1_dict["global col_%s" % str(i * 2)] = vth1
    print vth1_dict
    pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/test1.pdf'
    pp = PdfPages(pdfName)
    tdac_hm, tdac_hist = DGC_plotting.tdac_heatmap(h5_file_name=None, en_mask_in=mask_en, tdac_mask_in=mask_tdac)
    pp.savefig(tdac_hm)
    plt.clf()
    pp.savefig(tdac_hist)
#     plt.show()
    pp.close()
    # at end need to save masks to file, need to save vth1_dict also
#     scan_results = output_file.create_group("/", 'scan_results', 'Scan Results')
#     with tb.open_file(output_file, mode='w'):
#         output_file.create_carray(scan_results, 'tdac_mask', obj=mask_tdac)
#         output_file.create_carray(scan_results, 'en_mask', obj=mask_en)

#     for j in enumerate(range(0, 16, 2)):
#         mask_tdac += tdac_cols["col_%s" % str(j * 2)]
    # must check to make sure that no parts of the other columns come into this part!
    # TODO: make output file for all masks and vth1s with which columns it goes with
