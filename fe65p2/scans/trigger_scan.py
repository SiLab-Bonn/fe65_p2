
from fe65p2.scan_base import ScanBase
import fe65p2.plotting as plotting
import time

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, show, vplot, hplot, save
from progressbar import ProgressBar
import os

local_configuration = {
    "mask_steps": 4,
    "repeat_command": 100,

    # DAC parameters
    "PrmpVbpDac": 36,
    "vthin1Dac": 70,
    "vthin2Dac": 0,
    "vffDac": 24,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 110,
    "mask_filename": '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180105_131328_noise_tuning.h5'
}


class TriggerScan(ScanBase):
    scan_id = "trigger_scan"

    def scan(self, mask_steps=4, repeat_command=10, columns=[True] * 16, mask_filename=None, **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat_command : int
            Number of injections.
        '''
        repeat_command = kwargs.get('repeat_command', repeat_command)

        #columns = [True] + [False] * 15

        self.set_local_config()
        self.dut.set_for_configuration()

        mask_trig = np.full([64, 64], False, dtype=np.bool)
        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], 16, dtype=np.uint8)

        if mask_filename:
            logging.info('***** Using pixel mask from file: %s', mask_filename)

            with tb.open_file(str(mask_filename), 'r') as in_file_h5:
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                mask_en = in_file_h5.root.scan_results.en_mask[:]

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        self.dut.start_up()

        # enable inj pulse and trigger
        # wait_for_read = (16 + columns.count(True) *
        #                  (4 * 64 / mask_steps) * 2) * (20 / 2) + 10000
        # self.dut['inj'].set_delay(wait_for_read)
        # self.dut['inj'].set_width(100)
        # self.dut['inj'].set_repeat(repeat_command)
        # self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(400 - 4)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        lmask = [1] + ([0] * (mask_steps - 1))
        lmask = lmask * ((64 * 64) / mask_steps + 1)
        lmask = lmask[:64 * 64]

        bv_mask = bitarray.bitarray(lmask)
        num = 0

        # mask_en = np.full([64, 64], False, dtype=np.bool)
        # mask_en[5, 4] = True
        # mask_en[45, 13] = True
        # mask_en[1, 2] = True
        # self.dut.write_en_mask(mask_en)

        with self.readout():
            while num < repeat_command:
                for i in range(mask_steps):
                    self.dut.set_for_configuration()

                    self.dut.write_global()
                    time.sleep(0.1)

                    self.dut['pixel_conf'][:] = bv_mask
                    self.dut.write_pixel_col()
                    self.dut['global_conf']['InjEnLd'] = 1
                    #self.dut['global_conf']['PixConfLd'] = 0b11
                    self.dut.write_global()

                    bv_mask[1:] = bv_mask[0:-1]
                    bv_mask[0] = 0

                    self.set_local_config()
                    time.sleep(0.1)

                    self.dut['trigger'].start()
                    while not self.dut['trigger'].is_done():
                        time.sleep(0.01)

                self.dut['trigger'].set_en(False)
                # self.get_record_count()
                if num % 10 == 0:
                    print num
                num += 1

    def analyze(self):
        h5_filename = self.output_filename + '.h5'

        with tb.open_file(h5_filename, 'r+') as out_file_h5:
            raw_data = out_file_h5.root.raw_data[:]
            meta_data = out_file_h5.root.meta_data[:]

            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            out_file_h5.create_table(
                out_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)

            occ = np.histogram2d(x=hit_data['col'], y=hit_data['row'],
                                 bins=(64, 64), range=((0, 64), (0, 64)))[0]

            out_file_h5.create_carray(out_file_h5.root, name='HistOcc', title='Occupancy Histogram',
                                      obj=occ)

        occ_plot, H = plotting.plot_occupancy(h5_filename)
        tot_plot, _ = plotting.plot_tot_dist(h5_filename)
        lv1id_plot, _ = plotting.plot_lv1id_dist(h5_filename)

        output_file(self.output_filename + '.html', title=self.run_name)
        save(vplot(occ_plot, tot_plot, lv1id_plot))


if __name__ == "__main__":

    scan = TriggerScan()
    scan.start(**local_configuration)
    scan.analyze()
