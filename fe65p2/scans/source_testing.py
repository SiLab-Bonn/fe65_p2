'''
code to use with a radioactive source. self triggering, 
connect LEMO_TX[0] (tdc_out) to LEMO_RX[0] (trig_in) to send data when measured (avoid empty triggers)

Created by Daniel Coquelin on 21/12/2017
'''
from fe65p2.scan_base import ScanBase
import fe65p2.plotting as plotting
import fe65p2.DGC_plotting as DGC_plotting

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


from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfFile
import matplotlib.pyplot as plt


local_configuration = {
    "max_data_count": 10000,
    "columns": [True] * 16 + [False] * 0 + [True] * 0 + [False] * 0,

    # DAC parameters
    #     "PrmpVbpDac": 100,
    #     "vthin1Dac": 20,
    #     "vthin2Dac": 0,
    #     "vffDac": 110,
    #     "PrmpVbnFolDac": 51,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 25,
    #     "preCompVbnDac": 150,
    "PrmpVbpDac": 80,
    "vthin1Dac": 100,
    "vthin2Dac": 0,
    "vffDac": 42,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 100,
    "mask_filename": '',  # /home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20171222_140517_noise_tuning.h5',
    "thresh_mask": ''
}


class SourceTesting(ScanBase):
    scan_id = "source_testing"

    def scan(self, max_data_count=100, columns=[True] * 16, mask_filename=None, **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat_command : int
            Number of injections.
        '''
        max_data_count = kwargs.get("max_data_count", max_data_count)

        #columns = [True] + [False] * 15

        self.dut.set_for_configuration()

        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], 16, dtype=np.uint8)
        mask_hitor = np.full([64, 64], True, dtype=np.bool)

        if mask_filename:
            logging.info('***** Using pixel mask from file: %s', mask_filename)

            with tb.open_file(str(mask_filename), 'r') as in_file_h5:
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                mask_en = in_file_h5.root.scan_results.en_mask[:]

        self.dut.start_up()

        self.dut['global_conf']['OneSr'] = 1
        # TODO: do i need to set the value for the DUT_HIT_OR here?? dont think so but unsure
        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)
        self.dut.write_hitor_mask(mask_hitor)

        columns = kwargs.get("columns", columns)
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)

        # trigger delay needs to be tuned for here. the hit_or takes more time to go through everything
        # best delay here was ~395 (for chip1) make sure to tune before data taking.
        # once tuned reduce the number of triggers sent (width)

        self.dut['trigger'].set_delay(395)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)

        with self.readout():

            count_old = 0
            total_old = 0
            self.dut.set_for_configuration()
            self.set_local_config()

            self.dut['trigger'].set_en(True)

            for loop in range(100):
                sleep_time = 6
                time.sleep(sleep_time)
                count_loop = self.fifo_readout.get_record_count() - count_old
                print "words received in loop ", loop, ": ", count_loop, " \tcount rate per second: ", count_loop / sleep_time
                count_old = self.fifo_readout.get_record_count()

            self.dut['trigger'].set_en(False)
            time.sleep(1)

            # for vth1 in xrange(30, 100, 5):
            #     self.dut['global_conf']['vthin1Dac'] = vth1
            # for delay in range(0, 500, 20):
            self.dut.set_for_configuration()

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
            print "occ sum", occ.sum()

            out_file_h5.create_carray(out_file_h5.root, name='HistOcc', title='Occupancy Histogram',
                                      obj=occ)

            # lv1id_hist = np.histogram(hit_data['lv1id'])
            # out_file_h5.create_carray(
            #     out_file_h5.root, name='lv1id_hist', title="lv1id hist", obj=lv1id_hist)

        pdfName = self.output_filename + '.pdf'
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

        occ_plot, H = plotting.plot_occupancy(h5_filename)
        tot_plot, _ = plotting.plot_tot_dist(h5_filename)
        lv1id_plot, _ = plotting.plot_lv1id_dist(h5_filename)
        output_file(self.output_filename + '.html', title=self.run_name)
        save(vplot(occ_plot, tot_plot, lv1id_plot))


if __name__ == "__main__":

    scan = SourceTesting()
    scan.start(**local_configuration)
    scan.analyze()
