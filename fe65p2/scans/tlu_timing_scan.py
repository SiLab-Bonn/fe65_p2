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
import fe65p2.scans.inj_tuning_columns as inj_cols
import fe65p2.scans.noise_tuning_columns as noise_cols
from bokeh.models.layouts import Column, Row
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from basil.dut import Dut
import fe65p2.scans.tlu_tuning as tlu_tuning
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

        self.dut['trigger'].set_delay(356)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        self.dut['TLU_veto_pulse'].set_delay(100)
        self.dut['TLU_veto_pulse'].set_width(16)
        self.dut['TLU_veto_pulse'].set_repeat(1)
        self.dut['TLU_veto_pulse'].set_en(True)

        self.dut['TLU'].TRIGGER_COUNTER = 0
        self.dut['TLU'].TRIGGER_ENABLE = 0
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
        veto_test_range = np.arange(1, 3000, 50)
        for t_veto in veto_test_range:
            self.dut.set_for_configuration()
            self.dut['TLU_veto_pulse'].set_delay(int(t_veto))
            self.dut['TLU_veto_pulse'].set_width(16)
            self.dut['TLU_veto_pulse'].set_repeat(1)
            self.dut['TLU_veto_pulse'].set_en(True)

#             for t_trig in timing_test_range:
#                 words = 0
#                 logging.info("trig: %s veto: %s", str(t_trig), str(t_veto))
#                 self.dut.set_for_configuration()
#                 self.dut['trigger'].set_delay(int(t_trig))
#                 self.dut['trigger'].set_width(16)
#                 self.dut['trigger'].set_repeat(1)
#                 self.dut['trigger'].set_en(True)
#                 self.dut.write_global()
            self.set_local_config(vth1=vth1)

            old_trigs = self.dut['TLU'].TRIGGER_COUNTER
            with self.readout(scan_param_id=param_num, fill_buffer=True, clear_buffer=True):
                self.dut['TLU'].TRIGGER_ENABLE = 1
                param_num += 1
                time.sleep(60)
                self.dut['TLU'].TRIGGER_ENABLE = 0
                words = self.fifo_readout.get_record_count()
                print words, t_veto
            new_trigs = self.dut['TLU'].TRIGGER_COUNTER - old_trigs
            print "triggers: ", new_trigs
            if words > 0:
                #                     logging.info("\n\tTRIGGER ORDER STUFF\n")
                #                     trig_counter = self.dut['TLU'].TRIGGER_COUNTER
                dqdata = self.fifo_readout.data
                data = np.concatenate([item[0] for item in dqdata])
                data_words = np.where(((data & TRIGGER_ID) == False) & ((data & BCID_ID) == False))
                print 'data_words: ', len(data_words[0])
                logging.info("data words: %s", str(len(data_words[0])))
#                 if data_words[0].shape[0] > 50:
                #                         tlu_tuning.trig_id_inc_rate(data, new_trigs)
                #                         tlu_tuning.dist_between_tlu_words(data)

                hit_data = self.dut.interpret_raw_data_tlu(data)
                hit_data_lv1id = hit_data['lv1id']
                print "num:", hit_data_lv1id.shape[0], "std:", np.std(hit_data_lv1id)
                if hit_data_lv1id.shape[0] > 25:
                    bar_data = np.bincount(hit_data_lv1id)
                    bins = np.arange(bar_data.shape[0])
                    print 'sum:', np.sum(bar_data), 'veto:', t_veto
                    print bar_data
#                     logging.info("std:", str(np.std(hit_data_lv1id)))
                    logging.info(str(bar_data))
                    max_bin = bins[np.argmax(bar_data)]
                    if np.sum(bar_data) > 10:
                        logging.info("t_veto %s sum %s max %s",
                                     str(t_veto), str(np.sum(bar_data)), str(max_bin))
                        logging.info("\n")
                    else:
                        logging.info("Not enough data/not correct data shape\nt_veto %s", str(t_veto))
            else:
                logging.info("No words for t_veto %s", str(t_veto))
            print "\nend veto loop\n"
        logging.info("\n end scan loop \n\n")


if __name__ == "__main__":
    scan = TLU_Timing_Scan()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan.start(**local_configuration)
#     scan.analyze()
