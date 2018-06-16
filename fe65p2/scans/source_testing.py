'''
code to use with a radioactive source. self triggering, 
connect LEMO_TX[0] (tdc_out) to LEMO_RX[0] (trig_in) to send data when measured (avoid empty triggers)

Created by Daniel Coquelin on 21/12/2017
'''
from fe65p2.scan_base import ScanBase
import fe65p2.DGC_plotting as DGC_plotting
import fe65p2.scans.noise_tuning_columns as noise_cols
import fe65p2.analysis as analysis
import yaml
import time

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip4.yaml'


local_configuration = {
    "max_data_count": 10000,
    "quad_columns": [True] * 16 + [False] * 0 + [True] * 0 + [False] * 0,

    # DAC parameters
    #     "PrmpVbpDac": 100,
    #     "vthin1Dac": 20,
    #     "vthin2Dac": 0,
    #     "vffDac": 110,
    #     "PrmpVbnFolDac": 51,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 25,
    #     "preCompVbnDac": 150,

    # /home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20171222_140517_noise_tuning.h5',
    #     "mask_filename": '/home/daniel/Documents/InterestingPlots/chip3/20180321_112749_noise_tuning.h5',
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

        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], 16, dtype=np.uint8)
        mask_hitor = np.full([64, 64], False, dtype=np.bool)

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
        vth1 += 125
        print vth1

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

        self.dut.start_up()

        self.dut['global_conf']['OneSr'] = 1

        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        columns = kwargs['quad_columns']
        # self.dut['global_conf']['OneSr'] = 0  #all multi columns in parallel
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac.astype(np.uint8))
        self.dut.write_hitor_mask(mask_hitor)

        # trigger delay needs to be tuned for here. the hit_or takes more time to go through everything
        # best delay here was ~395 (for chip1) make sure to tune before data taking.
        # once tuned reduce the number of triggers sent (width)

        self.dut['trigger'].set_delay(00)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(10)

#         # enable TDC
#         logging.debug('Enable TDC')
#         self.dut['tdc']['RESET'] = True
#         self.dut['tdc']['EN_TRIGGER_DIST'] = True
#         self.dut['tdc']['ENABLE_EXTERN'] = False
#         self.dut['tdc']['EN_ARMING'] = False
#         self.dut['tdc']['EN_INVERT_TRIGGER'] = False
#         self.dut['tdc']['EN_INVERT_TDC'] = False
#
#         # if using tdc need only one pixel at a time!
#         pixel_range = [0, 4092]
#         mask_en_test = mask_en_from_file
#         for pix in range(pixel_range[0], pixel_range[1]):
#             #             if mask_en_test[pix + 100] == True:
#             self.dut.set_for_configuration()
#             mask_en[:, :] = False
#             mask_hitor = mask_hitor.reshape(4096)
#             mask_hitor[:] = False
#             mask_hitor[1861] = True
#             mask_hitor = mask_hitor.reshape(64, 64)
#             mask_en = mask_hitor
#             self.dut.write_en_mask(mask_en)
#             self.dut.write_hitor_mask(mask_hitor)
#
#             self.set_local_config(vth1=vth1)
#             logging.info('Starting Scan on Pixel %s' % pix)
#
#             with self.readout(scan_param_id=pix):
#
#                 time.sleep(300)
#                 while not self.dut['trigger'].is_done():
#                     time.sleep(0.05)
#
#                 self.dut['tdc'].ENABLE = False
#                 logging.info('Words Received: %s' % str(self.fifo_readout.get_record_count()))
#
#             break
        with self.readout():

            count_old = 0
            total_old = 0
            self.dut.set_for_configuration()
            self.set_local_config(vth1=vth1)

            self.dut['trigger'].set_en(True)
            time.sleep(5.0)

#             repeat_loop = 100
#             sleep_time = 6
#             for loop in range(repeat_loop):
#                 time.sleep(sleep_time)
#                 count_loop = self.fifo_readout.get_record_count() - count_old
#                 print "words received in loop ", loop, ": ", count_loop, "\tcount rate per second: ", count_loop / sleep_time
#                 count_old = self.fifo_readout.get_record_count()

            self.dut['trigger'].set_en(False)
#             time.sleep(1)
            print "total_words:", self.fifo_readout.get_record_count(), "counts/s:", self.fifo_readout.get_record_count() / (repeat_loop * sleep_time)

            # for vth1 in xrange(30, 100, 5):
            #     self.dut['global_conf']['vthin1Dac'] = vth1
            # for delay in range(0, 500, 20):
            self.dut.set_for_configuration()

    def analyze(self):
        h5_filename = self.output_filename + '.h5'

        with tb.open_file(h5_filename, 'r+') as out_file_h5:
            #             raw = out_file_h5.root.raw_data[:]
            #             tdc_data = raw & 0xFFF  # only want last 12 bit
            #             # cut out data from the second part of the pulse
            #             tdc_delay = (raw & 0x0FF00000) >> 20
            # #             tdc_data = tdc_data[tdc_delay < 255][1:]
            #             print tdc_data
            #             print tdc_delay
            #
            #             plt.hist(tdc_data, bins=max(tdc_data))
            #             print "tdc mean: ", tdc_data.mean(), " sigma: ", tdc_data.std(), " length: ", tdc_data.shape[0]
            #
            # #             print "tdc delay mean: ", tdc_delay[tdc_delay < 253].mean(), " sigma: ", tdc_delay[tdc_delay < 253].std()
            #             plt.title("tot num delay: %d" % tdc_delay.shape[0])
            #             plt.show()

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
#
#         pdfName = self.output_filename + '.pdf'
        pdfName = "/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/source_testing.pdf"
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

    scan = SourceTesting()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan.start(**local_configuration)
    scan.analyze()
