#===============================================================================
# objective; find the correct timing numbers
#
# can be done with or without tdc
#===============================================================================

from fe65p2.scan_base import ScanBase
import fe65p2.plotting as plotting
import fe65p2.DGC_plotting as DGC_plotting
import time
import fe65p2.analysis as analysis
import yaml
import logging
import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, save
import fe65p2.scans.inj_tuning_columns as inj_cols
import fe65p2.scans.noise_tuning_columns as noise_cols
from bokeh.models.layouts import Column, Row
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from basil.dut import Dut
from pytlu.tlu import Tlu
import pytlu


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip4.yaml'

local_configuration = {
    "quad_columns": [True] * 16 + [False] * 0,
    "repeat_command": 100,
    "mask_steps": 4,
    "veto_width": 100,  # used with sending triggers every 100000 units of 25ns
    "trigger_delay": 1,
    "trigger_width": 16,
    "veto_delay": 0,
    "veto_width": 16,
}

TRIGGER_ID = 0x80000000
TRG_MASK = 0x7FFFFFFF
BCID_ID = 0x800000
BCID_MASK = 0x7FFFFF


class TLU_Timing_Scan(ScanBase):
    scan_id = "tlu_timing_scan"

    def scan(self, repeat_command=1000, mask_filename='', **kwargs):

        #=======================================================================
        # 1. operate the pulser in continuous mode (sync will send the trigger pulse to the TLU)
        # 2. tlu talks to chip to start readout
        # 3. tlu makes busy signal to hold chip in readout state
        # 4. chip finishes readout, next available pulse starts process again
        # 5. data is printed to show structure.....TBD about analysis code
        #=======================================================================

        self.dut.set_for_configuration()

        self.dut['control']['RESET'] = 0b01
        self.dut['control']['DISABLE_LD'] = 0
        self.dut['control']['PIX_D_CONF'] = 0
        self.dut['control'].write()

        self.dut['control']['CLK_OUT_GATE'] = 1
        self.dut['control']['CLK_BX_GATE'] = 1
        self.dut['control'].write()
        time.sleep(0.1)

        self.dut['control']['RESET'] = 0b11
        self.dut['control'].write()
        self.dut['control']['EN_HITOR_TRIGGER'] = 0
        self.dut['control'].write()
        self.dut.start_up()

        self.dut['global_conf']['OneSr'] = 1

        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        columns = kwargs['quad_columns']
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        mask_en = np.full([64, 64], False, dtype=np.bool)
        mask_tdac = np.full([64, 64], 15, dtype=np.uint8)
        mask_inj = np.full([64, 64], False, dtype=np.bool)
        mask_hitor = np.full([64, 64], False, dtype=np.bool)

        for inx, col in enumerate(kwargs['quad_columns']):
            if col:
                mask_en[inx * 4:(inx + 1) * 4, :] = True

        file0 = kwargs.get("noise_col0")
        file1 = kwargs.get("noise_col1")
        file2 = kwargs.get("noise_col2")
        file3 = kwargs.get("noise_col3")
        file4 = kwargs.get("noise_col4")
        file5 = kwargs.get("noise_col5")
        file6 = kwargs.get("noise_col6")
        file7 = kwargs.get("noise_col7")
        mask_en_from_file, mask_tdac, vth1 = noise_cols.combine_prev_scans(
            file0=file0, file1=file1, file2=file2, file3=file3, file4=file4, file5=file5, file6=file6, file7=file7)
        vth1 += 20
        print vth1

        ex_pix_disable_list = kwargs.get("ex_pix_disable_list")
        mask_en_from_file = mask_en_from_file.reshape(4096)
        mask_en_from_file[ex_pix_disable_list] = False
        mask_en_from_file = mask_en_from_file.reshape(64, 64)

#         mask_steps = kwargs.get("mask_steps", 4)
#         wait_for_read = (16 + columns.count(True) * (4 * 64 / mask_steps) * 2) * (20 / 2) + 10000
#         self.dut['inj'].set_delay(wait_for_read * 100)
#         self.dut['inj'].set_width(1000)
#         self.dut['inj'].set_repeat(repeat_command)
#         self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(15)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        self.dut['TLU_veto_pulse'].set_delay(100)
        self.dut['TLU_veto_pulse'].set_width(16)
        self.dut['TLU_veto_pulse'].set_repeat(1)
        self.dut['TLU_veto_pulse'].set_en(True)

        self.dut['TLU'].TRIGGER_COUNTER = 0
        self.dut['TLU'].TRIGGER_MODE = 3
        self.dut['TLU'].TRIGGER_SELECT = 0  # 0-> disabled, 1-> hitOr, 2->RX0, 3-> RX1, 4->RX2
        self.dut['TLU'].TRIGGER_VETO_SELECT = 1
        self.dut['TLU'].EN_TLU_VETO = 0
        self.dut['TLU'].DATA_FORMAT = 0  # only trigger id
        self.dut['TLU'].TRIGGER_DATA_DELAY = 6
        self.dut['TLU'].TRIGGER_HANDSHAKE_ACCEPT_WAIT_CYCLES = 20


#===============================================================================
# scan loop is completed for a variety of different delays for both trigger and veto delays
#===============================================================================
        param_num = 0
        timing_test_range = np.arange(200, 1000, 14)
        veto_test_range = np.arange(100, 10000, 200)
        for t_veto in veto_test_range:
            self.dut.set_for_configuration()
            self.dut['TLU_veto_pulse'].set_delay(int(t_veto))
            self.dut['TLU_veto_pulse'].set_width(16)
            self.dut['TLU_veto_pulse'].set_repeat(1)
            self.dut['TLU_veto_pulse'].set_en(True)

            for t_trig in timing_test_range:
                words = 0
                logging.info("trig: %s veto: %s", str(t_trig), str(t_veto))
                self.dut.set_for_configuration()
                self.dut['trigger'].set_delay(int(t_trig))
                self.dut['trigger'].set_width(16)
                self.dut['trigger'].set_repeat(1)
                self.dut['trigger'].set_en(True)
                self.dut.write_global()
                self.set_local_config(vth1=vth1)

                with self.readout(scan_param_id=param_num, fill_buffer=True, clear_buffer=True):
                    #                     self.dut['TLU'].TRIGGER_ENABLE = 1
                    param_num += 1
                    time.sleep(0.5)
                    self.stop_tlu_triggers()
#                     self.dut['TLU'].TRIGGER_ENABLE = 0
                    words = self.fifo_readout.get_record_count()
                    print words, t_trig, t_veto
                if words > 0:
                    logging.info("\n\tTRIGGER ORDER STUFF\n")
                    trig_counter = self.dut['TLU'].TRIGGER_COUNTER
                    dqdata = self.fifo_readout.data
                    data = np.concatenate([item[0] for item in dqdata])
                    trig_id_pos = np.where(data & TRIGGER_ID)
                    trig_id_list = data[trig_id_pos] & TRG_MASK

                    trig_id_full_less_data = data[np.where((data & TRIGGER_ID) | (data & BCID_ID))]
                    trig_id_num_less_data = np.where(trig_id_full_less_data & TRIGGER_ID)
                    num_between = np.diff(trig_id_num_less_data)

                    trig_id_diffs = np.diff(trig_id_list).astype(int)
                    try:
                        trig_diff_hist = np.bincount(trig_id_diffs)
                        bin_positions_trig = np.arange(trig_diff_hist.shape[0])
                        trig_mean = np.average(trig_diff_hist, weights=bin_positions_trig) * \
                            bin_positions_trig.sum() / np.nansum(trig_diff_hist)

                        perc_w_1_bw_triggs = (float(trig_diff_hist[1]) / float(trig_counter)) * 100.
                        logging.info("trig mean:%s trig std: %s \ntrig max: %s trig max position: %s", str(trig_mean), str(np.std(trig_id_diffs)),
                                     str(max(trig_id_diffs)), str(np.argmax(trig_id_diffs)))
                        logging.info(trig_diff_hist)
                        logging.info(perc_w_1_bw_triggs)
                    except:
                        logging.info("negative values in the trigger differences!!!")
                        trig_id_diffs[trig_id_diffs < 0] = 0
                        try:
                            trig_diff_hist = np.bincount(trig_id_diffs)
                            bin_positions_trig = np.arange(trig_diff_hist.shape[0])
                            trig_mean = np.average(trig_diff_hist, weights=bin_positions_trig) * \
                                bin_positions_trig.sum() / np.nansum(trig_diff_hist)
                            perc_w_1_bw_triggs = (float(trig_diff_hist[1]) / float(trig_counter)) * 100.
                            logging.info("trig mean:%s trig std: %s \ntrig max: %s trig max position: %s", str(trig_mean), str(np.std(trig_id_diffs)),
                                         str(max(trig_id_diffs)), str(np.argmax(trig_id_diffs)))
                            logging.info(trig_diff_hist)
                            logging.info(perc_w_1_bw_triggs)
                        except:
                            logging.info("failed twice to make bincount for trigs, exiting")

                    logging.info("\n\tBCID STUFF \n")
                    bcid_diff_hist = np.bincount(num_between[0])
                    bin_positions_bcid = np.arange(bcid_diff_hist.shape[0])
                    bcid_mean = np.average(bcid_diff_hist, weights=bin_positions_bcid) * \
                        bin_positions_bcid.sum() / np.nansum(bcid_diff_hist)
                    logging.info("bcid mean:%s bcid std: %s \nbcid max: %s bcid max position: %s", str(bcid_mean), str(np.std(bcid_diff_hist)),
                                 str(max(bcid_diff_hist)), str(np.argmax(bcid_diff_hist)))
                    logging.info("diff between trig numbers hist \n %s", str(bcid_diff_hist))

                    logging.info("\n\tLv1id Stuff\n")

                    hit_data = self.dut.interpret_raw_data_tlu(data)
                    hit_data_lv1id = hit_data['lv1id']
                    print "sum:", np.sum(hit_data_lv1id), "std:", np.std(hit_data_lv1id)
                    if np.sum(hit_data_lv1id) > 100:
                        bar_data = np.bincount(hit_data_lv1id)
                        bins = np.arange(bar_data.shape[0])
                        print 'trig:', t_trig, 'sum:', np.sum(bar_data), 'veto:', t_veto
                        print bar_data
    #                     logging.info("std:", str(np.std(hit_data_lv1id)))
                        logging.info(str(bar_data))
                        max_bin = bins[np.argmax(bar_data)]
                        if np.sum(bar_data) > 100 and np.std(hit_data_lv1id) < 4:
                            logging.info("t_trig %s t_veto %s sum %s max %s", str(t_trig), str(t_veto), str(np.sum(bar_data)), str(max_bin))
                            logging.info("\n")
                        else:
                            logging.info("Not enough data/not correct data shape\nt_trig %s t_veto %s", str(t_trig), str(t_veto))
                else:
                    logging.info("No words for t_trig %s t_veto %s", str(t_trig), str(t_veto))
            print "\nend veto loop\n"
        logging.info("\n end scan loop \n\n")


if __name__ == "__main__":
    scan = TLU_Timing_Scan()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan.start(**local_configuration)
#     scan.analyze()
