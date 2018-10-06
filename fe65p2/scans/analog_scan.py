
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
    "repeat_command": 10,

    # DAC parameters
    "PrmpVbpDac": 36,
    "vthin1Dac": 110,
    "vthin2Dac": 0,
    "vffDac": 42,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 50,
}


class AnalogScan(ScanBase):
    scan_id = "analog_scan"

    def scan(self, mask_steps=4, repeat_command=100, columns=[True] * 16, **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''
        
        #columns = [True] + [False] * 15

        self.dut['INJ_LO'].set_voltage(0.1, unit='V')
        self.dut['INJ_HI'].set_voltage(1.2, unit='V')
        

        self.set_local_config()
        self.dut.set_for_configuration()

        mask_inj = np.full([64, 64], False, dtype=np.bool)
        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], 16, dtype=np.uint8)

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        self.dut.start_up()

        # enable inj pulse and trigger
        wait_for_read = (16 + columns.count(True) *
                         (4 * 64 / mask_steps) * 2) * (20 / 2) + 10000
        self.dut['inj'].set_delay(wait_for_read)
        self.dut['inj'].set_width(100)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(400 - 4)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        # for the set_en command
        # If true: The pulse comes with a fixed delay with respect to the external trigger (EXT_START).
        # If false: The pulse comes only at software start.

        with self.readout():
            pbar = ProgressBar(maxval=mask_steps).start()
            for i in range(mask_steps):
                for qcol in range(16):
                    self.dut.set_for_configuration()

                    mask_inj[:, :] = False
                    mask_inj[qcol * 4:(qcol + 1) * 4 + 1, i::mask_steps] = True
                    self.dut.write_inj_mask(mask_inj)

                    self.set_local_config()

                    self.dut['inj'].start()

                    if os.environ.get('TRAVIS'):
                        logging.debug('.')

                    pbar.update(i)

                    while not self.dut['inj'].is_done():
                        time.sleep(0.01)

                    while not self.dut['trigger'].is_done():
                        time.sleep(0.01)

                    time.sleep(0.01)

            # just some time for last read
            self.dut['trigger'].set_en(False)
            self.dut['inj'].start()

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

        return H


if __name__ == "__main__":

    scan = AnalogScan()
    scan.start(**local_configuration)
    scan.analyze()
