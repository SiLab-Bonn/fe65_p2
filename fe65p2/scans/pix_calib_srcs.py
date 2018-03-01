'''
Pixel tdc calibration with sources

This scan will take a set time on each pixel reading out hits automatically,
can either switch to the next pixel after a set number of hits or a set time
will use the results of the pixel_calib_inj to characterize the tdc for a larger range

Created by Daniel Coquelin 19.02.2018
'''

from fe65p2.scan_base import ScanBase
import fe65p2.DGC_plotting as DGC_plotting
import time
import fe65p2.analysis as analysis
import yaml
import logging
import numpy as np
import bitarray
import tables as tb
import fe65p2.scans.inj_tuning_columns as inj_cols
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from basil.dut import Dut
import os
from fe65p2 import DGC_plotting


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
    "PrmpVbpDac": 160,
    "vthin1Dac": 130,  # this will be overwritten with the average of all of the columns with the combine function from inj_cols
    "vthin2Dac": 0,
    "vffDac": 80,
    "PrmpVbnFolDac": 81,
    "vbnLccDac": 1,
    "compVbnDac": 50,
    "preCompVbnDac": 80,

    # chip 4
    #     "PrmpVbpDac": 100,
    #     "vthin1Dac": 60,
    #     "vthin2Dac": 0,
    #     "vffDac": 110,
    #     "PrmpVbnFolDac": 51,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 25,
    #     "preCompVbnDac": 150,

    # BEFORE RUNNING ENSURE THAT RX0 IS CONNECTED TO TX0!!!

    # tdc calib scan w src
    "stop_time": 120,  # unit: s -> stop after 2 mins of no data
    "stop_count": 100,  # number of words to read out before stopping
    "TDAC": 16,
    "pix_range": [0, 4096],
    "source_elecs": 0,  # MUST MANUALLY ENTER THIS!!!
    # list of radioactive sources in lab:
    # 90 Sr  -> 2283
    # 55 Fe  ->
    # 14 C   ->
    # 109 Cd ->
    # 241 Am -> 16000
    # 57 Co  ->
    # 22 Na  ->
    # 88 Y   ->
    # 137 Cs ->
    # 60 Co  ->
}


class PixelCalibSrc(ScanBase):
    scan_id = "pix_calib_src"

    def scan(self, TDAC=16, scan_range=[0.0, 1.0, 0.02], repeat_command=1000, mask_filename='', **kwargs):
        '''Scan loop
        Parameters
        ----------
        mask_filename : int
            Number of mask steps.
        repeat_command : int
            Number of injections.
        TDAC : int
            initial pixel threshold value
        '''

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

#         self.dut['pixel_conf'].setall(False)
#         self.dut.write_pixel()
#         self.dut['global_conf']['InjEnLd'] = 1
#         self.dut.write_global()
#         self.dut['global_conf']['InjEnLd'] = 0

        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], TDAC, dtype=np.uint8)
        mask_inj = np.full([64, 64], False, dtype=np.bool)
        mask_hitor = np.full([64, 64], False, dtype=np.bool)

        for inx, col in enumerate(kwargs['quad_columns']):
            if col:
                mask_en[inx * 4:(inx + 1) * 4, :] = True

#         if mask_filename:
#             logging.info('***** Using pixel mask from file: %s', mask_filename)
#
#             with tb.open_file(str(mask_filename), 'r') as in_file_h5:
#                 mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
#                 mask_en = in_file_h5.root.scan_results.en_mask[:]

        # run function from noise_cols to read all of the data from the noise scans for the columns
        try:
            mask_en, mask_tdac, tune_vth1 = inj_cols.combine_prev_scans(file0='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180206_210702_tdac_scan.h5',
                                                                        file1='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180206_211958_tdac_scan.h5',
                                                                        file2='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180206_213254_tdac_scan.h5',
                                                                        file3='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180206_214547_tdac_scan.h5',
                                                                        file4='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180207_024150_tdac_scan.h5',
                                                                        file5='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180207_025446_tdac_scan.h5',
                                                                        file6='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180207_030738_tdac_scan.h5',
                                                                        file7='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180207_032133_tdac_scan.h5')

        except:
            print 'fial'
        if tune_vth1:
            vth1 = tune_vth1
            print "set vth1 from tuning scans"
        else:
            vth1 = kwargs.get("vthin1Dac", 132)
        print vth1
        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac.astype('B'))
        self.dut.write_hitor_mask(mask_hitor)

        self.dut['trigger'].set_delay(395)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)

        # enable TDC
        logging.debug('Enable TDC')
        self.dut['tdc']['RESET'] = True
        self.dut['tdc']['EN_TRIGGER_DIST'] = True
        self.dut['tdc']['ENABLE_EXTERN'] = False
        self.dut['tdc']['EN_ARMING'] = False
        self.dut['tdc']['EN_INVERT_TRIGGER'] = False
        self.dut['tdc']['EN_INVERT_TDC'] = False

        # loop over all pixels, pix number = scan_parameter
        pix_range = kwargs.get("pix_range", [0, 4096])
        stop_time = kwargs.get("stop_time", 100)  # unit: s
        stop_count = kwargs.get("stop_count", 1000)
        for pix in range(pix_range[0], pix_range[1]):

            #             scan_range = list(np.linspace(1820, 17000, 30))
            #             flavor_scan_params.append(scan_range)
            #             if pix == 512:
            #                 scan_range = list(np.linspace(1900, 17000, 30))
            #                 flavor_scan_params.append(scan_range)
            #             if pix == 1024:
            #                 scan_range = list(np.linspace(1890, 17000, 30))
            #                 flavor_scan_params.append(scan_range)
            #             if pix == 1536:
            #                 scan_range = list(np.linspace(2230, 17000, 30))
            #                 flavor_scan_params.append(scan_range)
            #             if pix == 2048:
            #                 scan_range = list(np.linspace(1840, 17000, 30))
            #                 flavor_scan_params.append(scan_range)
            #             if pix == 2560:
            #                 scan_range = list(np.linspace(1940, 17000, 30))
            #                 flavor_scan_params.append(scan_range)
            #             if pix == 3072:
            #                 scan_range = list(np.linspace(2230, 17000, 30))
            #                 flavor_scan_params.append(scan_range)
            #             if pix == 3584:
            #                 scan_range = list(np.linspace(1950, 17000, 30))
            #                 flavor_scan_params.append(scan_range)

            self.dut.set_for_configuration()
            mask_hitor = mask_hitor.reshape(4096)
            mask_hitor[:] = False
            mask_hitor[pix] = True
            mask_hitor = mask_hitor.reshape(64, 64)
#             mask_en = mask_hitor
#             self.dut.write_en_mask(mask_en)
            self.dut.write_hitor_mask(mask_hitor)

            self.set_local_config(vth1=vth1)
            logging.info('Starting Scan on Pixel %s' % pix)
            with self.readout(scan_param_id=pix):
                finished = False
                start = time.time()
                while not finished:
                    self.dut['tdc']['ENABLE'] = True
                    self.dut['trigger'].set_en(True)
                    time.sleep(0.5)

                    while not self.dut['trigger'].is_done():
                        time.sleep(0.05)
                    # need to get the stop time working properly
                    if self.fifo_readout.get_record_count() > stop_count:
                        finished = True
                        self.dut['trigger'].set_en(False)
                        self.dut['tdc'].ENABLE = False
                        logging.info('Count Break, Words Received: %s' % str(self.fifo_readout.get_record_count()))

                    if time.time() > (start + stop_time):
                        finished = True
                        self.dut['trigger'].set_en(False)
                        self.dut['tdc'].ENABLE = False
                        logging.info('Time Break, Words Received: %s' % str(self.fifo_readout.get_record_count()))
#                     break
#             break

    def analyze(self):
        h5_filename = self.output_filename + '.h5'
        combi_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/tdc_calib.h5'
        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/pix_calib_src_testing.pdf'
        pp = PdfPages(pdfName)
        spectrum = DGC_plotting.tdc_src_spectrum(h5_filenam)
        pp.savefig(spectrum)
        plt.clf()
        analysis.tdc_table_w_srcs(h5_file_in=h5_filename, h5_file_old=???, out_file_name=combi_file)
        
        pp.close()
        
if __name__ == "__main__":

    scan = PixelCalibSrc()
    scan.start(**local_configuration)
    scan.analyze()
