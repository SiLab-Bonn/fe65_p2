'''
code to do a noise tuning test on each column or chip flavor seperately.
tdac masks will be combined at the end,
en masks will be combined as well but this is a bit tricky, may just save the pixel numbers and pass those
vth1 values will be printed at the end of each run and  compared at the end

Created by Daniel Coquelin on 1/17/2018
'''

from fe65p2.scans.noise_tuning import NoiseTuning
import numpy as np
import tables as tb
import logging
import time
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

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
    # chip4
    "PrmpVbpDac": 125,
    "vthin1Dac": 185,
    "vthin2Dac": 0,
    "vffDac": 73,
    "PrmpVbnFolDac": 61,
    "vbnLccDac": 1,
    "compVbnDac": 45,
    "preCompVbnDac": 180,

}


def noise_sc(qcols):
    logging.info("Starting Noise Scan")
    noise_sc = NoiseTuning()
    noise_mask_file = str(noise_sc.output_filename) + '.h5'

    custom_conf = {
        "columns": qcols,
        "stop_pixel_percent": 3,
        "pixel_disable_switch": 6,
        "repeats": 10000,
    }

    scan_conf = dict(local_configuration, **custom_conf)
    noise_sc.start(**scan_conf)
    noise_sc.analyze()
#     print noise_sc.vth1Dac
    noise_sc.dut.close()
    return noise_mask_file, noise_sc.final_vth1

# def save_flavor_to_mask(mask,):


def combine_prev_scans(file0, file1, file2, file3, file4, file5, file6, file7):
    # loop over the files like "file"+str(i)
    file_list = [file0, file1, file2, file3, file4, file5, file6, file7]
    vth1_list = []
    for j, filename in enumerate(file_list):

        with tb.open_file(filename, 'r+') as in_file:
            mask_en_hold = in_file.root.scan_results.en_mask[:]
            mask_tdac_hold = in_file.root.scan_results.tdac_mask[:]
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
    max_vth1 = max(vth1_list)
    return mask_en, mask_tdac, max_vth1


if __name__ == "__main__":
    # need to cycle over all of the columns
    output_file = str(time.clock()) + "noise_tuning_by_flavor" + ".h5"
    #mask_en = np.full([64, 64], False, dtype=np.bool)
    #mask_tdac = np.full([64, 64], 0, dtype=np.uint8)

    tdac_cols = {}
    vth1_dict = {}

    qcolumns = [False] * 16
    for i in range(0, 8, 1):
        en_mask_qcol = np.full((64, 8), False, dtype=np.bool)
        qcolumns = [False] * (i * 2) + [True] * 2 + [False] * ((7 - i) * 2)
        print qcolumns
        noise_filename, noise_vth1 = noise_sc(qcolumns)
        with tb.open_file(str(noise_filename), 'r') as in_file_h5:
            # saves all in a dictionary numbered as col_0 -> col_14 in intervals of 2
            mask_tdac_hold = in_file_h5.root.scan_results.tdac_mask[:]
            mask_en_hold = in_file_h5.root.scan_results.en_mask[:]
        #mask_tdac += tdac_cols["col_%s" % str(i * 2)][mask_en_hold == True]

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
        vth1_dict["col_%s" % str(i * 2)] = noise_vth1
    print vth1_dict

    # at end need to save masks to file, need to save vth1_dict also

    with tb.open_file(output_file, mode='w'):
        scan_results = output_file.create_group("/", 'scan_results', 'Scan Results')
        output_file.create_carray(scan_results, 'tdac_mask', obj=mask_tdac)
        output_file.create_carray(scan_results, 'en_mask', obj=mask_en)

#     for j in enumerate(range(0, 16, 2)):
#         mask_tdac += tdac_cols["col_%s" % str(j * 2)]
    # must check to make sure that no parts of the other columns come into this part!
    # TODO: make output file for all masks and vth1s with which columns it goes with
