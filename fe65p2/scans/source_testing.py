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
    "PrmpVbpDac": 36,
    "vthin1Dac": 60,
    "vthin2Dac": 0,
    "vffDac": 255,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 0,
    "compVbnDac": 25,
    "preCompVbnDac": 50,
    "mask_filename": '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20171222_140517_noise_tuning.h5',
    "thresh_mask": ''
}


class SourceTesting(ScanBase):
    scan_id = "source_testing"

    def scan(self, mask_steps=4, max_data_count=100, columns=[True] * 16, mask_filename=None, **kwargs):
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

        lmask = [1] + ([0] * (mask_steps - 1))
        lmask = lmask * ((64 * 64) / mask_steps + 1)
        lmask = lmask[:64 * 64]

        bv_mask = bitarray.bitarray(lmask)
        columns = kwargs.get("columns", columns)
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)

        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        count_old = 0
        total_old = 0
        self.dut.set_for_configuration()
        self.dut.write_global()
        self.set_local_config()

        with self.readout():
            # for vth1 in xrange(30, 100, 5):
            #     self.dut['global_conf']['vthin1Dac'] = vth1
            # for delay in range(0, 500, 20):
            self.dut.set_for_configuration()
            self.dut.write_global()
            delay = 395
            self.dut['trigger'].set_delay(delay)
            self.set_local_config()
            self.dut['trigger'].start()

            time.sleep(60)

            count_new = self.fifo_readout.get_record_count()
            count_diff = count_new - count_old
            count_old = count_new
            print ("delay %s \twords: %s" % (delay, count_diff))

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
