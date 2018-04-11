'''
Scan to find the vth1 for which there is a noise occupancy of <1 per 1 000 000 triggers
Created by Daniel Coquelin on 26/2/18
'''
from fe65p2.scan_base import ScanBase
import fe65p2.DGC_plotting as DGC_plotting
import fe65p2.scans.inj_tuning_columns as inj_cols
import fe65p2.scans.noise_tuning_columns as noise_cols
import time
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import yaml

local_configuration = {
    "triggers": 1000000,
    "columns": [True] * 16,

    # DAC parameters
    #     "PrmpVbpDac": 100,
    #     "vthin1Dac": 20,
    #     "vthin2Dac": 0,
    #     "vffDac": 110,
    #     "PrmpVbnFolDac": 51,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 25,
    #     "preCompVbnDac": 150,
    "PrmpVbpDac": 165,
    "vthin1Dac": 45,
    "vthin2Dac": 0,
    "vffDac": 86,
    "PrmpVbnFolDac": 61,
    "vbnLccDac": 1,
    "compVbnDac": 45,
    "preCompVbnDac": 185,

    "mask_filename": '/home/daniel/Documents/InterestingPlots/chip3/noise_tuning_14.05_26_0.h5',
}


class NoiseOccVth1(ScanBase):
    scan_id = "noise_occ_vth1_scan"

    def scan(self, max_data_count=100, columns=[True] * 16, mask_filename='', **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat_command : int
            Number of injections.
        '''
        triggers = kwargs.get("triggers", 1000000)

        #columns = [True] + [False] * 15

        self.dut.set_for_configuration()

        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], 15, dtype=np.uint8)
        mask_hitor = np.full([64, 64], True, dtype=np.bool)

#         try:
#             mask_en, mask_tdac, vth1 = noise_cols.combine_prev_scans(file0='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180302_113648_noise_tuning.h5',
#                                                                      file1='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180302_114850_noise_tuning.h5',
#                                                                      file2='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180302_120114_noise_tuning.h5',
#                                                                      file3='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180302_121346_noise_tuning.h5',
#                                                                      file4='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180302_122503_noise_tuning.h5',
#                                                                      file5='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180302_123728_noise_tuning.h5',
#                                                                      file6='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180302_125000_noise_tuning.h5',
#                                                                      file7='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180302_130204_noise_tuning.h5')
#         except:
#             vth1 = kwargs.get("vthin1Dac", 15)
#
#         self.vth1Dac = vth1
#         mask_filename = kwargs.get(
#             "mask_filename", '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180227_101242_noise_tuning.h5')

        if mask_filename:
            logging.info('***** Using pixel mask from file: %s', mask_filename)

            with tb.open_file(str(mask_filename), 'r') as in_file_h5:
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                mask_en = in_file_h5.root.scan_results.en_mask[:]
                mask_tdac[mask_tdac == 32] = 31

                vth1 = yaml.load(in_file_h5.root.meta_data.attrs.vth1) + 20
#                 vth1 = dac_status['vthin1Dac']
                print vth1

        self.vth1Dac = vth1
        print vth1

        self.dut.start_up()

        self.dut['global_conf']['OneSr'] = 1
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

        self.dut['trigger'].set_delay(1500)
        self.dut['trigger'].set_width(8)
        self.dut['trigger'].set_repeat(triggers)

        vthin1DacInc = 1
        finished = False
        iteration = 0
        acceptable_noise = 1 / 1000000
        noise_pix_acc = acceptable_noise * triggers
        while not finished:
            with self.readout(scan_param_id=self.vth1Dac, fill_buffer=True, clear_buffer=True):
                self.dut.set_for_configuration()
                self.set_local_config(vth1=self.vth1Dac)
                logging.info('Scan iteration: %d (vthin1Dac = %d)', iteration,  self.vth1Dac)

                self.dut['trigger'].start()

                while not self.dut['trigger'].is_done():
                    time.sleep(0.01)

            dqdata = self.fifo_readout.data
            data = np.concatenate([item[0] for item in dqdata])
            hit_data = self.dut.interpret_raw_data(data)
            hits = hit_data['col'].astype(np.uint16)
            hits = hits * 64
            hits = hits + hit_data['row']
            count = np.bincount(hits, minlength=64 * 64)
            # count bins and see how many hits, if cnz > noise_pix_acc up the vth1

            cnz = np.count_nonzero(count)
            print cnz, noise_pix_acc
            if cnz > noise_pix_acc:
                self.dut.set_for_configuration()
                self.vth1Dac += vthin1DacInc
                self.set_local_config(vth1=self.vth1Dac)
            else:
                finished = True

            iteration += 1
        logging.info('Final vthin1Dac value: %s', str(self.vth1Dac))
        self.final_vth1 = self.vth1Dac
        return self.vth1Dac

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

#         pdfName = self.output_filename + '.pdf'
        pdfName = "/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/noise_occ_testing.pdf"
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


if __name__ == "__main__":

    scan = NoiseOccVth1()
    _ = scan.start(**local_configuration)
    scan.analyze()
