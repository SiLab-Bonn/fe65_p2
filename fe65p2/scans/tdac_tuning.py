'''
scan to be used after the global threshold is found using vth1_scan
this scan will inject the same amount of charge each time and vary the TDAC from 0 to 31 and then fit s_curves for each pixel
after the fits, the averages of the pixel readouts are the tdac values. 
these averages are then made into a mask then saved to an h5 file to be used later

created by Daniel Coquelin on 26/01/2018
'''

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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from bokeh.charts import output_file, save
from bokeh.models.layouts import Column, Row

from basil.dut import Dut


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

local_configuration = {
    "quad_columns": [True] * 16 + [False] * 0,
    #   DAC parameters
    # default params
    #     "PrmpVbpDac": 36,
    #     "vthin1Dac": 255,
    #     "vthin2Dac": 0,
    #     "vffDac": 42,
    #     "PrmpVbnFolDac": 51,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 25,
    #     "preCompVbnDac": 50,

    # chip 3
    "PrmpVbpDac": 36,
    "vthin1Dac": 60,
    "vthin2Dac": 0,
    "vffDac": 42,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 50,

    # chip 4
    #     "PrmpVbpDac": 100,
    #     "vthin1Dac": 60,
    #     "vthin2Dac": 0,
    #     "vffDac": 110,
    #     "PrmpVbnFolDac": 51,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 25,
    #     "preCompVbnDac": 150,

    #   thrs scan
    "mask_steps": 4,
    "repeat_command": 100,
    "scan_range": [0, 32, 1],
    # bare chip mask: '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180115_174703_noise_tuning.h5',
    "mask_filename": '',  # '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180116_091958_noise_tuning.h5',
    "inj_electrons": 2000,
    "vth1_from_scan": 100,
}


class TDACScan(ScanBase):
    scan_id = "tdac_scan"

    def scan(self, mask_steps=4, repeat_command=100, columns=[True] * 16, **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''
        inj_electrons = kwargs.get("inj_electrons", 2000)
        #columns = [True] + [False] * 15
        try:  # pulser
            pulser = Dut(ScanBase.get_basil_dir(self) + '/examples/lab_devices/agilent33250a_pyserial.yaml')
            pulser.init()
            logging.info('Connected to Pulser: ' + str(pulser['Pulser'].get_info()))
            pulser['Pulser'].set_usr_func("FEI4_PULSE")
            pulse_width = 30000
            pulser['Pulser'].set_pulse_period(pulse_width * 10**-9)
            pulser['Pulser'].set_voltage(0.0, inj_electrons / (1000 * analysis.cap_fac()), unit='V')
        except:
            INJ_LO = 0.2
            self.dut['INJ_LO'].set_voltage(float(INJ_LO), unit='V')
            logging.info('External injector not connected. Switch to internal one')
            self.dut['INJ_LO'].set_voltage(0.1, unit='V')
            self.dut['INJ_HI'].set_voltage(1.2, unit='V')

        vth1_from_scan = kwargs.get("vth1_from_scan", 100)
        self.set_local_config(vth1=vth1_from_scan)
        self.dut.set_for_configuration()

        quad_columns = kwargs.get("quad_columns", columns)
        mask_inj = np.full([64, 64], False, dtype=np.bool)
        mask_en = np.full([64, 64], False, dtype=np.bool)
        for inx, col in enumerate(quad_columns):
            if col:
                mask_en[inx * 4:(inx + 1) * 4, :] = True
        mask_tdac = np.full([64, 64], 0, dtype=np.uint8)

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)

        columns = kwargs.get("columns", columns)
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        self.dut.start_up()
        repeat_command = kwargs.get("repeat_command", repeat_command)

        # enable inj pulse and trigger
        wait_for_read = (16 + columns.count(True) * (4 * 64 / mask_steps) * 2) * (20 / 2) + 10000

        self.dut['inj'].set_delay(wait_for_read * 20)
        self.dut['inj'].set_width(1000)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(400 - 1)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        mask_steps = kwargs.get("mask_steps", mask_steps)
        tdac_range = np.arange(32)

        for idx, tdac in enumerate(tdac_range):
            self.dut.set_for_configuration()
            mask_tdac = np.full([64, 64], tdac, dtype=np.uint8)
            self.dut.write_tune_mask(mask_tdac)
            with self.readout(scan_param_id=idx):
                logging.info('Scan Parameter: %f (%d of %d)', tdac, idx + 1, len(tdac_range))

                for i in range(mask_steps):
                    self.dut.set_for_configuration()

                    mask_inj[:, :] = False

                    for qcol in range(16):
                        mask_inj[qcol * 4:(qcol + 1) * 4 + 1, i::mask_steps] = True
                    self.dut.write_inj_mask(mask_inj)

                    self.set_local_config(vth1=vth1_from_scan)

                    self.dut['inj'].start()
                    time.sleep(0.3)

                    while not self.dut['inj'].is_done():
                        time.sleep(0.05)

                    while not self.dut['trigger'].is_done():
                        time.sleep(0.05)
                    time.sleep(.5)

                    print "finished mask_step: ", i, " words recieved: ", self.fifo_readout.get_record_count()
        scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Masks')
        self.h5_file.create_carray(scan_results, 'tdac_mask', obj=np.full([64, 64], 0, dtype=np.uint8))
        self.h5_file.create_carray(scan_results, 'en_mask', obj=mask_en)

    def analyze(self):
        pp = PdfPages(self.output_filename + '.pdf')
#         pp = PdfPages('/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/tdac_tuning_testing.pdf')
        print pp
        h5_filename = self.output_filename + '.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]

            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            in_file_h5.create_table(
                in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)

            occ = np.histogram2d(x=hit_data['col'], y=hit_data['row'], bins=(64, 64), range=((0, 64), (0, 64)))[0]

            in_file_h5.create_carray(in_file_h5.root, name='HistOcc', title='Occupancy Histogram', obj=occ)

            # self.meta_data_table.attrs.dac_status
        analysis.analyze_tdac_scan(h5_filename)

        occ_plot = DGC_plotting.plot_occupancy(h5_filename)
        pp.savefig(occ_plot)
        plt.clf()
        tot_plot = DGC_plotting.plot_tot_dist(h5_filename)
        pp.savefig(tot_plot)
        plt.clf()
        lv1id_plot = DGC_plotting.plot_lv1id_dist(h5_filename)
        pp.savefig(lv1id_plot)
        plt.clf()
        singlePixPolt, thresHM, thresVsPix, thresDist, noiseHM, noiseDist, chi2plot = DGC_plotting.scan_pix_hist(h5_filename)
        pp.savefig(singlePixPolt)
        plt.clf()
        pp.savefig(thresHM)
        plt.clf()
        pp.savefig(thresVsPix)
        plt.clf()
        pp.savefig(thresDist)
        plt.clf()
        pp.savefig(noiseHM)
        plt.clf()
        pp.savefig(noiseDist)
        plt.clf()
        pp.savefig(chi2plot)
        plt.clf()
        t_dac_plot = DGC_plotting.tdac_plot_for_tdac_scan(h5_filename)
        pp.savefig(t_dac_plot)

        pp.close()

#         status_plot1 = plotting.plot_status(h5_filename)
#         occ_plot1, _ = plotting.plot_occupancy(h5_filename)
#         tot_plot1, _ = plotting.plot_tot_dist(h5_filename)
#         lv1id_plot1, _ = plotting.plot_lv1id_dist(h5_filename)
#         scan_pix_hist1, _ = plotting.scan_pix_hist(h5_filename)
#
#         output_file(self.output_filename + '.html', title=self.run_name)
#
#         save(Column(Row(occ_plot1, tot_plot1, lv1id_plot1), scan_pix_hist1, status_plot1))


if __name__ == "__main__":

    scan = TDACScan()
    scan.start(**local_configuration)
    scan.analyze()
