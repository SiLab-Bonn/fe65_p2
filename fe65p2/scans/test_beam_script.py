#===============================================================================
# Script for taking test beam data
# must have the proper delays (get from the timing scans)
#
# TODO before:
#     -> must run a TLU tuning for the proper cable length
#     -> if the cable length is different, then use the timing scan as well
#     ->
#
# this scan will run until X data words are received
# TODO: create a controller to start the next scan after this one finishes
#     exit condition? 1hr?
#     where is tlu? my pc or elsewhere?
# TODO: switch between tdc and regular data readout
# TODO:
#
#===============================================================================

from fe65p2.scan_base import ScanBase
import fe65p2.DGC_plotting as DGC_plotting
import time
import fe65p2.analysis as analysis
import yaml
import logging
import numpy as np
import bitarray
import tables as tb
#import fe65p2.scans.inj_tuning_columns as inj_cols
import fe65p2.scans.noise_tuning_columns as noise_cols
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from basil.dut import Dut
from pytlu.tlu import Tlu
from threading import Thread
from queue import Queue
import sys
import traceback
from pybar import *

TRIGGER_ID = 0x80000000
TRG_MASK = 0x7FFFFFFF
BCID_ID = 0x800000
BCID_MASK = 0x7FFFFF

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

yaml_file = '/home/xraytube/fe65_p2/fe65p2/chip4.yaml'

local_configuration = {
    "quad_columns": [True] * 16 + [False] * 0,
    "repeat_command": 100,
    "mask_steps": 4,
}

scan_finished_flag = False
scan_working = False
scan_anal_done = False


def scan_func(self):
    scan = Test_Beam_Script()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    try:
        scan_finished_flag = scan.start(**local_configuration)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logging.warning(lines)
        # if overflow -> power cycle
        try:
            dut = Dut(ScanBase.get_basil_dir(self) + '/examples/lab_devices/tti_ql355tp_pyserial.yaml')
            dut.init()
            print dut['THURLBY-THANDAR,QL355TP,D,1.6'].get_info()
            dut['THURLBY-THANDAR,QL355TP,D,1.6'].power_cycle(channel=3)
            time.sleep(0.2)
            scan.start(**local_configuration)
        except:
            logging.warning("Failed power cycle of chip")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            logging.warning(lines)
    if scan_finished_flag == True:
        scan.analyze()
        scan_finished_flag = False


def scan_controller(self):
    # TODO: from the analysis want to see the lv1id, tot hist, occ, and correlation?
    t1 = Thread(target=scan_func())
    t2 = Thread(target=scan_func())
#     t3 = Thread(target=scan_func())
    t1_started = False
    t2_started = False
#     t3_started = False
    # TODO: need a new start condition, total words???
    while scan_working == False:
        if not t1_started:
            t1.start()
            t1_started = True
        elif t1_started and not t1.is_alive():
            t1.run()
        else:
            t1.join()

        while scan_working == True:
            time.sleep(1.0)  # must poll before starting the next scan

        if not t2.is_alive():
            if not t2_started:
                t2.start()
            else:
                t2.run()
        else:
            t2.join()

        while scan_working:
            time.sleep(1.0)
            if scan_anal_done:
                t1.join()
                scan_anal_done = False
        t2.join()


class Test_Beam_Script(ScanBase):
    scan_id = "test_beam_script"

    def scan(self, repeat_command=1000, mask_filename='', **kwargs):
        scan_working = True
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
        mask_inj = np.full([64, 64], True, dtype=np.bool)
        mask_hitor = np.full([64, 64], False, dtype=np.bool)
#         np.reshape(mask_inj, 4096)
#         mask_inj[1::4] = True
#         np.reshape(mask_inj, (64, 64))
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

        self.dut.write_en_mask(mask_en_from_file)
        self.dut.write_tune_mask(mask_tdac)
        self.dut.write_inj_mask(mask_inj)
        self.dut.write_hitor_mask(mask_hitor)

        self.dut['trigger'].set_delay(469)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        self.dut['TLU_veto_pulse'].set_delay(624)
        self.dut['TLU_veto_pulse'].set_width(16)
        self.dut['TLU_veto_pulse'].set_repeat(1)
        self.dut['TLU_veto_pulse'].set_en(True)

        self.dut['TLU'].RESET = 1
        self.dut['TLU'].TRIGGER_COUNTER = 0
        self.dut['TLU'].TRIGGER_MODE = 3
        self.dut['TLU'].TRIGGER_SELECT = 0  # 0-> disabled, 1-> hitOr, 2->RX0, 3-> RX1, 4->RX2
        self.dut['TLU'].TRIGGER_VETO_SELECT = 0
        self.dut['TLU'].DATA_FORMAT = 0  # only trigger id
        # Depends on the cable length and should be adjusted (run scan/tune_tlu.py in pybar repository)
        self.dut['TLU'].TRIGGER_DATA_DELAY = 5
        self.dut['TLU'].TRIGGER_HANDSHAKE_ACCEPT_WAIT_CYCLES = 5


#===============================================================================
# scan loop is until there is 50000 words
#===============================================================================

        word_stop = 50000000
        words = 0
        completed = False
        param_num = 0
        self.dut.set_for_configuration()
        self.set_local_config(vth1=vth1)
        self.dut['TLU'].TRIGGER_ENABLE = 1
        with self.readout(scan_param_id=param_num):
            while not completed:
                time.sleep(300.)
                words = self.fifo_readout.get_record_count()
                if words >= word_stop:
                    print words
                    completed = True
                param_num += 1
        self.dut['TLU'].TRIGGER_ENABLE = 0

        scan_working = False
        scan_finished_flag = True
        return words

    def analyze(self):
        scan_anal_done = False
        h5_filename = self.output_filename + '.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]

            hit_data = self.dut.interpret_raw_data_tlu(raw_data, meta_data)
            in_file_h5.create_table(in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)
        scan_anal_done = True
        # TODO: log needed things here


if __name__ == "__main__":
    scan = Test_Beam_Script()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan.start(**local_configuration)
    scan.analyze()
