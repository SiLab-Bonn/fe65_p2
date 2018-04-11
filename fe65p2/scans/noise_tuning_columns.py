'''
code to do a noise tuning test on each column or chip flavor seperately.
tdac and en masks are created by combining the files with a function defined here
vth1 values will be printed at the end of each run and compared at the end

Created by Daniel Coquelin on 1/17/2018
'''

from fe65p2.scans.noise_tuning import NoiseTuning
import fe65p2.DGC_plotting as DGC_plotting
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import numpy as np
import tables as tb
import logging
import time
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip4.yaml'

local_configuration = {
    #   DAC parameters
    #     "PrmpVbpDac": 140,
    #     "vthin1Dac": 200,
    #     "vthin2Dac": 0,
    #     "vffDac": 84,
    #     "PrmpVbnFolDac": 87,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 50,
    #     "preCompVbnDac": 86,

}


def noise_sc(qcols):
    logging.info("Starting Noise Scan")
    noise_sc = NoiseTuning()
    noise_mask_file = str(noise_sc.output_filename) + '.h5'

    custom_conf = {
        "columns": qcols,
        "stop_pixel_percent": 2,
        "pixel_disable_switch": 6,
        "repeats": 100000,
    }
    yaml_kwargs = yaml.load(open(yaml_file))
    scan_conf = dict(dict(yaml_kwargs), **custom_conf)
    noise_sc.start(**scan_conf)
    _, dis_count, tdac_hist, _ = noise_sc.analyze()
#     print noise_sc.vth1Dac
    noise_sc.dut.close()
    return noise_mask_file, noise_sc.final_vth1, dis_count, tdac_hist

# def save_flavor_to_mask(mask,):


def combine_prev_scans(file0, file1, file2, file3, file4, file5, file6, file7):
    # loop over the files like "file"+str(i)
    file_list = [file0, file1, file2, file3, file4, file5, file6, file7]
    vth1_list = []
    for j, filename in enumerate(file_list):

        with tb.open_file(filename, 'r+') as in_file:
            mask_en_hold = in_file.root.scan_results.en_mask[:]
            mask_tdac_hold = in_file.root.scan_results.tdac_mask[:]
#             dac_status = yaml.load(in_file.root.meta_data.attrs.dac_status)
            vth1 = yaml.load(in_file.root.meta_data.attrs.vth1)
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
    max_vth1 = max(vth1_list)
    return mask_en, mask_tdac, max_vth1


if __name__ == "__main__":
    vth1_dict = {}

    qcolumns = [False] * 16
    bin0 = []
    disabled_count = []
    for i in range(0, 8, 1):
        en_mask_qcol = np.full((64, 8), False, dtype=np.bool)
        qcolumns = [False] * (i * 2) + [True] * 2 + [False] * ((7 - i) * 2)
        print qcolumns
        noise_filename, noise_vth1, dis_count, tdac_hist = noise_sc(qcolumns)
        np.append(bin0, tdac_hist[0])
        np.append(disabled_count, dis_count)
        with tb.open_file(str(noise_filename), 'r') as in_file_h5:
            # saves all in a dictionary numbered as col_0 -> col_14 in intervals of 2
            mask_tdac_hold = in_file_h5.root.scan_results.tdac_mask[:]
            mask_en_hold = in_file_h5.root.scan_results.en_mask[:]
        # reshape to 4096 then delete the other stuff
        np.reshape(mask_en_hold, 4096)
        np.reshape(mask_tdac_hold, 4096)
        if i == 0:
            mask_en = mask_en_hold[0:512]
            mask_tdac = mask_tdac_hold[0:512]
        if i != 0:
            mask_en = np.concatenate((mask_en, mask_en_hold[i * 512:(i + 1) * 512]))
            mask_tdac = np.concatenate((mask_tdac, mask_tdac_hold[i * 512:(i + 1) * 512]))

        vth1_dict["col_%s" % str(i * 2)] = noise_vth1
    np.reshape(mask_en, (64, 64))
    np.reshape(mask_tdac, (64, 64))
    print vth1_dict
    print "total disabled:", np.sum(disabled_count)
    print disabled_count
    print "total bin0:", np.sum(bin0)
    # at end need to save masks to file, should save vth1_dict also
    output_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/noise_tuning_cols' + str(time.clock()) + '.h5'
    h5_file = tb.open_file(output_file, mode='w')
    scan_results = h5_file.create_group("/", 'scan_results', 'Scan Results')
    tdac_table = h5_file.create_carray(scan_results, 'tdac_mask', obj=mask_tdac)
    h5_file.create_carray(scan_results, 'en_mask', obj=mask_en)
    tdac_table.attrs.vth1s = yaml.dump(vth1_dict)

    pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/noise_cols_combi_test.pdf'
    pp = PdfPages(pdfName)
    tdac_hm, tdac_hist = DGC_plotting.tdac_heatmap(h5_file_name=None, en_mask_in=mask_en, tdac_mask_in=mask_tdac)
    pp.savefig(tdac_hm, layout='tight')
    plt.clf()
    pp.savefig(tdac_hist, layout='tight')
    pp.close()
