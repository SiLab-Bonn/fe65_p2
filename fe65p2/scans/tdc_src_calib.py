#===============================================================================
#
# Scan to do charge calibation with sources (not specifically sources but ...)
#
# must enable the pixels around the target pixel to get clustering information
# must change analysis code to make this work
#
#===============================================================================

from fe65p2.scan_base import ScanBase
import time

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from progressbar import ProgressBar
import fe65p2.DGC_plotting as DGC_plotting
from matplotlib.backends.backend_pdf import PdfPages
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from basil.dut import Dut
import os
import fe65p2.scans.noise_tuning_columns as noise_cols
import yaml
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure


yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip3.yaml'


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


class TDCtest(ScanBase):
    scan_id = "tdc_test"
    # sample pixels to test with sources for charge calibration
    # (7,15), (13,11), (24,15), (42, 12), (8, 45), (14, 46), (23, 45), (46, 46)

    # non responsive pixels
    # (26, 25), (58, 36), (62, 55), (32, 48), (25, 30), (18, 14), (13, 61), (9, 59), (19, 50)
    def scan(self, repeat_command=10, columns=[True] * 16, pix_list=[(7, 15)], **kwargs):
        '''Scan loop

        Parameters/important shit
        ----------
            to sync the pulse generator and the boards need to have A: 'TX1 -> to Ext Input' and B: 'RXO -> to Sync/Trigger out'
        '''
        #pix_list = kwargs['pix_list']

        try:
            dut = Dut(ScanBase.get_basil_dir(self) + '/examples/lab_devices/agilent33250a_pyserial.yaml')
            dut.init()
            logging.info('Connected to ' + str(dut['Pulser'].get_info()))
        except RuntimeError:
            raise Warning("Pulser not connected, please connect it to use this scan!")

        self.set_local_config()
        self.dut.set_for_configuration()

        # enable all pixels but will only inject to selected ones see mask_inj
        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], 15., dtype=np.uint8)
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
#         vth1 = 1
        print vth1
        logging.info("vth1: %s" % str(vth1))
        self.dut.write_en_mask(mask_en_from_file)
        self.dut.write_tune_mask(mask_tdac.astype(np.uint8))
        self.dut.write_hitor_mask(mask_hitor)

        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        dut['Pulser'].set_on_off("ON")
        self.dut.start_up()

        self.dut['trigger'].set_delay(395)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        # enable TDC
        logging.debug('Enable TDC')
        self.dut['tdc']['RESET'] = True
        self.dut['tdc']['EN_TRIGGER_DIST'] = True
        self.dut['tdc']['ENABLE_EXTERN'] = False
        self.dut['tdc']['EN_ARMING'] = False
        self.dut['tdc']['EN_INVERT_TRIGGER'] = False
        self.dut['tdc']['EN_INVERT_TDC'] = False
        self.dut['tdc']['EN_WRITE_TIMESTAMP'] = True

        for pix in pix_list:
            # for each pixel need to enable the pixels around it
            #             dut['Pulser'].set_on_off("ON")
            count_old = 0
            total_old = 0
            mask_tdc = np.full([64, 64], False, dtype=np.bool)
            mask_en = np.full([64, 64], False, dtype=np.bool)
            mask_tdc[pix[0], pix[1]] = True
            mask_en[pix[0] - 2:pix[0] + 2, pix[1] - 2:pix[1] + 2] = True
            logging.info("pixel number: %s" % str(np.where(np.reshape(mask_tdc, 4096) == True)[0]))
            self.dut.write_hitor_mask(mask_tdc)
            self.dut.write_en_mask(mask_en)

            self.set_local_config(vth1=vth1)
#             time.sleep(3.0)

#             dut['Pulser'].set_on_off("OFF")

            with self.readout(scan_param_id=pix[0] * 64 + pix[1]):
                self.dut['tdc']['ENABLE'] = True
                repeat_loop = 10
                sleep_time = 1.
                self.dut['trigger'].set_width(int(sleep_time / 25e-9))
                pbar = tqdm(range(repeat_loop))
                for loop in pbar:
                    self.dut['trigger'].start()
                    # because of firmware bugs, need to set the number of repeats for the trigger by how long the loop is
                    # 25 ns clock on triggers, sleep_time in s
                    # sleep_time/25e-9
                    time.sleep(sleep_time)
                    count_loop = self.fifo_readout.get_record_count() - count_old
                    # print "words received in loop", loop, ":", count_loop, "\tcount rate per second: ", count_loop / sleep_time
                    pbar.set_description("Counts/s %s " % str(np.round(count_loop / sleep_time, 5)))
                    count_old = self.fifo_readout.get_record_count()
                    while not self.dut['trigger'].is_done():
                        time.sleep(0.05)

                self.dut['trigger'].set_en(False)
                self.dut['tdc'].ENABLE = 0
#         print "end for loop"

    def analyze(self):
        h5_filename = self.output_filename + '.h5'
        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/tdc_src_calib.pdf'
        pp = PdfPages(pdfName)
        fig = DGC_plotting.tdc_src_spectrum(h5_file=h5_filename)
        pp.savefig(fig, layout='tight')
        plt.clf()

        with tb.open_file(h5_filename, 'r+') as io_file_h5:
            meta_data = io_file_h5.root.meta_data[:]
            raw_data = io_file_h5.root.raw_data[:]
            for i in np.unique(meta_data['scan_param_id']):
                start = meta_data[meta_data['scan_param_id'] == i]['index_start'][0]
                stop = meta_data[meta_data['scan_param_id'] == i]['index_stop'][-1]
                data = raw_data[start:stop]
                tdc_data = data & 0xFFF  # only want last 12 bit

                fig1 = Figure()
                _ = FigureCanvas(fig1)
                ax1 = fig1.add_subplot(111)
                bar_data, bins = np.histogram(tdc_data, (max(tdc_data) - min(tdc_data)),
                                              range=(min(tdc_data), max(tdc_data)))
                bin_left = bins[:-1]
                ax1.bar(x=bin_left, height=bar_data, width=np.diff(bin_left)[0], align="edge")
                ax1.set_title("Spectrum of Source, Counts %s\nPixel: %s" % (str(tdc_data.shape[0]), str(i)))
                ax1.set_xlabel("TDC channel")
                ax1.set_ylabel("Counts")
                ax1.grid()
                fig1.tight_layout()

                pp.savefig(fig, layout='tight')
                plt.clf()
        pp.close()

        # cut out data from the second part of the pulse
#                 tdc_delay = (raw & 0x0FF00000) >> 20


# #             tdc_data = tdc_data[tdc_delay < 255][1:]
#             print tdc_data
#             print tdc_delay
#             plt.hist(tdc_data, bins=max(tdc_data))
# #             plt.hist(tdc_data[tdc_delay < 253], bins=max(tdc_data[tdc_delay < 253]))
#             print "tdc mean: ", tdc_data.mean(), " sigma: ", tdc_data.std(), " length: ", tdc_data.shape[0]
#
#             print "tdc delay mean: ", tdc_delay.mean()  # , " sigma: ", tdc_delay[tdc_delay != 255].std()
#             plt.title("tot num delay: %d" % tdc_delay.shape[0])
#             plt.show()


if __name__ == "__main__":
    test = TDCtest()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    test.start(**local_configuration)
    test.analyze()
