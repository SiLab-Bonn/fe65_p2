
from fe65p2.scan_base import ScanBase

import fe65p2.DGC_plotting as DGC_plotting
import time

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from progressbar import ProgressBar
import fe65p2.scans.noise_tuning_columns as noise_cols

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from basil.dut import Dut
import os
import yaml

local_configuration = {
    "mask_steps": 4,
    "repeat_command": 100,

    # DAC parameters
    #     "PrmpVbpDac": 36,
    #     "vthin1Dac": 60,
    #     "vthin2Dac": 0,
    #     "vffDac": 42,
    #     "PrmpVbnFolDac": 51,
    #     "vbnLccDac": 0,
    #     "compVbnDac": 25,
    #     "preCompVbnDac": 50

    "PrmpVbpDac": 120,
    "vthin1Dac": 120,
    "vthin2Dac": 3,
    "vffDac": 86,
    "PrmpVbnFolDac": 61,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 120,

    "mask_filename": '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180301_070254_noise_tuning.h5',

}


class AnalogScan(ScanBase):
    scan_id = "analog_scan"

    def scan(self, mask_steps=4, repeat_command=500, columns=[True] * 16, mask_filename='', **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''

        #columns = [True] + [False] * 15
        try:  # pulser
            pulser = Dut(ScanBase.get_basil_dir(self) + '/examples/lab_devices/agilent33250a_pyserial.yaml')
            pulser.init()
            logging.info('Connected to Pulser: ' + str(pulser['Pulser'].get_info()))
#             pulser['Pulser'].set_usr_func("FEI4_PULSE")
            pulse_width = 30000
            pulser['Pulser'].set_pulse_period(pulse_width * 10**-9)
            pulser['Pulser'].set_voltage(0.0001, 1.2, unit='V')
        except:
            INJ_LO = 0.2
            self.dut['INJ_LO'].set_voltage(float(INJ_LO), unit='V')
            logging.info('External injector not connected. Switch to internal one')
            self.dut['INJ_LO'].set_voltage(0.1, unit='V')
            self.dut['INJ_HI'].set_voltage(1.2, unit='V')

#         self.set_local_config()
        self.dut.set_for_configuration()

        mask_inj = np.full([64, 64], False, dtype=np.bool)
        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], 15, dtype=np.uint8)

#         mask_en, mask_tdac, _ = noise_cols.combine_prev_scans(file0='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180228_184403_noise_tuning.h5',
#                                                               file1='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180228_185238_noise_tuning.h5',
#                                                               file2='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180228_190128_noise_tuning.h5',
#                                                               file3='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180228_191006_noise_tuning.h5',
#                                                               file4='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180228_191831_noise_tuning.h5',
#                                                               file5='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180228_192721_noise_tuning.h5',
#                                                               file6='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180228_193620_noise_tuning.h5',
#                                                               file7='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180228_194600_noise_tuning.h5')
#         print vth1

        if mask_filename:
            logging.info('***** Using pixel mask from file: %s', mask_filename)

            with tb.open_file(str(mask_filename), 'r') as in_file_h5:
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                mask_en = in_file_h5.root.scan_results.en_mask[:]
                dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
                vth1 = dac_status['vthin1Dac']
                self.final_vth1 = vth1 + 10
                print vth1

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)
#         print mask_tdac.astype(np.uint8)

        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        self.dut.start_up()
        repeat_command = kwargs.get("repeat_command", repeat_command)

        # enable inj pulse and trigger
        wait_for_read = (16 + columns.count(True) * (4 * 64 / mask_steps) * 2) * (20 / 2) + 10000
#         print wait_for_read

        self.dut['inj'].set_delay(wait_for_read * 10)
        self.dut['inj'].set_width(100)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(401)
        self.dut['trigger'].set_width(8)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        # for the set_en command
        # If true: The pulse comes with a fixed delay with respect to the external trigger (EXT_START).
        # If false: The pulse comes only at software start.

        # mask_en = np.full([64, 64], False, dtype=np.bool)
        # mask_en[5, 4] = True
        # mask_en[45, 13] = True
        # mask_en[1, 2] = True
        # self.dut.write_en_mask(mask_en)

        with self.readout():
            pbar = ProgressBar(maxval=mask_steps).start()
            for i in range(mask_steps):

                self.dut.set_for_configuration()

                mask_inj[:, :] = False
                for qcol in range(16):
                    mask_inj[qcol * 4:(qcol + 1) * 4 + 1, i::mask_steps] = True
                self.dut.write_inj_mask(mask_inj)
                self.set_local_config()
#                 self.set_local_config(vth1=vth1 + 7)
                self.dut['inj'].start()
                if os.environ.get('TRAVIS'):
                    logging.debug('.')

                pbar.update(i)
                time.sleep(0.3)
                while not self.dut['inj'].is_done():
                    time.sleep(0.01)

                while not self.dut['trigger'].is_done():
                    time.sleep(0.01)

            # just some time for last read
            # self.dut['trigger'].set_en(False)
            # self.dut['inj'].start()

    def analyze(self):
        h5_filename = self.output_filename + '.h5'

        with tb.open_file(h5_filename, 'r+') as out_file_h5:
            raw_data = out_file_h5.root.raw_data[:]
            meta_data = out_file_h5.root.meta_data[:]
            scan_args = yaml.load(out_file_h5.root.meta_data.attrs.kwargs)

            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            out_file_h5.create_table(
                out_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)

            occ = np.histogram2d(x=hit_data['col'], y=hit_data['row'], bins=(64, 64), range=((0, 64), (0, 64)))[0]

            out_file_h5.create_carray(out_file_h5.root, name='HistOcc', title='Occupancy Histogram', obj=occ)

            print "occ average: ", occ.mean(), " occ std: ", occ.std(), " Num == repeats %: ", (float(occ[occ == scan_args['repeat_command']].shape[0]) / 4096) * 100
        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/analog_scan_testing3.pdf'
        pp = PdfPages(pdfName)
        occ_plot = DGC_plotting.plot_occupancy(h5_filename)
        pp.savefig(occ_plot)
        plt.clf()
        tot_plot = DGC_plotting.plot_tot_dist(h5_filename)
        pp.savefig(tot_plot)
        plt.clf()
        lv1id_plot = DGC_plotting.plot_lv1id_dist(h5_filename)
        pp.savefig(lv1id_plot)
        pp.close()

#         occ_plot, H = plotting.plot_occupancy(h5_filename)
#         tot_plot, _ = plotting.plot_tot_dist(h5_filename)
#         lv1id_plot, _ = plotting.plot_lv1id_dist(h5_filename)
#
#         output_file(self.output_filename + '.html', title=self.run_name)
#         save(vplot(occ_plot, tot_plot, lv1id_plot))
#
#         return H


if __name__ == "__main__":
    scan = AnalogScan()
    scan.start(**local_configuration)
    scan.analyze()
