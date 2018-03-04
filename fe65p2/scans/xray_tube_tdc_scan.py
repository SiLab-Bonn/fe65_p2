'''
Scan to characterize the sensor using the xray tube
things needed:
    TDC
    which source for the fitting
    add the characterisation to the other stuff for a good fit. 
    in the scan only need to get the data and test if its good
    
for this scan the setup is only the MIO board, GPAC, and the PCB with the fe65p2+sensor
    TDC is interally triggered by the hitbus

Created by Daniel Coquelin on 1/3/18
'''
from fe65p2.scan_base import ScanBase
import fe65p2.plotting as plotting
import fe65p2.DGC_plotting as DGC_plotting
import time
import logging
import numpy as np
import bitarray
import tables as tb
import os
import fe65p2.scans.noise_tuning_columns as noise_cols
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

local_configuration = {
    "max_data_count": 5000,
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
    # chip 3
    "PrmpVbpDac": 165,
    "vthin1Dac": 40,
    "vthin2Dac": 0,
    "vffDac": 86,
    "PrmpVbnFolDac": 61,
    "vbnLccDac": 1,
    "compVbnDac": 45,
    "preCompVbnDac": 185,

    # chip 4
    #     "PrmpVbpDac": 125,
    #     "vthin1Dac": 40,
    #     "vthin2Dac": 0,
    #     "vffDac": 73,
    #     "PrmpVbnFolDac": 61,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 45,
    #     "preCompVbnDac": 180,

    "mask_filename": '',  # /home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20171222_140517_noise_tuning.h5',

    "element_num": '',  # 42Mo, 50Sn, 73Ta, or 74W
}


class XRayTubeScan(ScanBase):
    scan_id = "xray_tube_scan"

    def scan(self, TDAC=16, scan_range=[0.0, 1.0, 0.02], repeat_command=1000, mask_filename='', **kwargs):
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

        try:
            mask_en, mask_tdac, _ = noise_cols.combine_prev_scans(file0='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180301_105301_noise_tuning.h5',
                                                                  file1='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180301_110423_noise_tuning.h5',
                                                                  file2='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180301_111543_noise_tuning.h5',
                                                                  file3='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180301_112658_noise_tuning.h5',
                                                                  file4='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180301_113821_noise_tuning.h5',
                                                                  file5='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180301_114959_noise_tuning.h5',
                                                                  file6='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180301_120120_noise_tuning.h5',
                                                                  file7='/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180301_121238_noise_tuning.h5')
            scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Masks')
            self.h5_file.create_carray(scan_results, 'tdac_mask', obj=mask_tdac)
            self.h5_file.create_carray(scan_results, 'en_mask', obj=mask_en)
        except:
            pass
        vth1 = kwargs.get("vthin1Dac", 50)
#         vth1 = 54
        print vth1

        mask_en_test = np.reshape(mask_en, 4096)

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac.astype(np.uint8))
        self.dut.write_inj_mask(mask_inj)
        self.dut.write_hitor_mask(mask_hitor)

        self.dut['trigger'].set_delay(00)
        self.dut['trigger'].set_width(8)
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

        max_data_count = kwargs.get("max_data_count", 1000)
        pixel_range = kwargs.get("pixel_range", [0, 4092])
        print pixel_range
        for pix in range(pixel_range[0], pixel_range[1]):
            if mask_en_test[pix] == True:
                self.dut.set_for_configuration()
                mask_hitor = mask_hitor.reshape(4096)
                mask_hitor[:] = False
                mask_hitor[pix] = True
                mask_hitor = mask_hitor.reshape(64, 64)
                mask_inj = mask_hitor
                mask_en = mask_hitor
                self.dut.write_en_mask(mask_en)
                self.dut.write_hitor_mask(mask_hitor)
                self.dut.write_inj_mask(mask_inj)

                self.set_local_config(vth1=vth1)
                logging.info('Starting Scan on Pixel %s' % pix)

                finished = False
                with self.readout(scan_param_id=pix):
                    self.dut['tdc']['ENABLE'] = True
                    self.dut['trigger'].set_en(True)

                    while not finished:
                        time.sleep(1.)
                        while not self.dut['trigger'].is_done():
                            time.sleep(0.05)
                        if self.fifo_readout.get_record_count() > max_data_count:
                            self.dut['trigger'].set_en(False)
                            self.dut['tdc'].ENABLE = False
                            finished = True
                            logging.info('Finished pixel %s Words Received %s' % (pix, str(self.fifo_readout.get_record_count())))


if __name__ == "__main__":

    scan = XRayTubeScan()
    scan.start(**local_configuration)
    scan.analyze()
