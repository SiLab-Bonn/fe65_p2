#===============================================================================
#
# script for hitor calibration with the pulser
#
# will start in volts and the convert to electrons later
# if ToT of 14 is not reached by the maximum pulser setting the need to use the
#
#
#===============================================================================

from fe65p2.scan_base import ScanBase
import fe65p2.DGC_plotting as DGC_plotting
import time
import logging
from fe65p2 import fifo_readout

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from progressbar import ProgressBar
import fe65p2.scans.noise_tuning_columns as noise_cols
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from basil.dut import Dut
import fe65p2.analysis as anal
import os
import yaml

yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip6.yaml'
pix_list = [[11, 6], [11, 20]]

local_configuration = {
    "quad_columns": [True] * 16
}

pixel_flav_list = ['nw15', 'nw20', 'nw25', 'nw30', 'dnw15', 'dnw20', 'dnw25', 'dnw30']


class HitOrCalib(ScanBase):
    scan_id = "hitor_calib"

    def scan(self, pixel=None, **kwargs):
        # pixels in column, row format
        if pixel:
            self.output_filename += str(pixel)

        try:  # pulser
            pulser = Dut(ScanBase.get_basil_dir(self) + '/examples/lab_devices/agilent33250a_pyserial.yaml')
            pulser.init()
            logging.info('Connected to ' + str(pulser['Pulser'].get_info()))
        except:
            logging.info("pulser required for this scan")
        mask_en = np.full([64, 64], False, dtype=np.bool)
        self.dut.write_inj_mask(mask_en)

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
        vth1 += 20
        print vth1
        logging.info("vth1: %s" % str(vth1))

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

        self.dut.write_tune_mask(mask_tdac)

        self.dut['trigger'].set_delay(395)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(False)

#         self.dut['inj'].set_delay(500)  # dealy betwean injection in 25ns unit
#         self.dut['inj'].set_width(1)
#         self.dut['inj'].set_repeat(100)
#         self.dut['inj'].set_en(False)

        self.dut['TLU'].TRIGGER_COUNTER = 0
        self.dut['TLU'].TRIGGER_MODE = 1
        self.dut['TLU'].TRIGGER_SELECT = 0  # 0-> disabled, 1-> hitOr, 2->RX0, 3-> RX1, 4->RX2
        self.dut['TLU'].DATA_FORMAT = 0  # only trigger id

        # enable TDC
        logging.debug('Enable TDC')
        self.dut['tdc']['RESET'] = True
        self.dut['tdc']['EN_TRIGGER_DIST'] = True
        self.dut['tdc']['ENABLE_EXTERN'] = False
        self.dut['tdc']['EN_ARMING'] = False
        self.dut['tdc']['EN_INVERT_TRIGGER'] = False
        self.dut['tdc']['EN_INVERT_TDC'] = False
        self.dut['tdc']['EN_WRITE_TIMESTAMP'] = True

        count_old = 0
        mask_tdc = np.full([64, 64], False, dtype=np.bool)
#         mask_en = np.full([64, 64], False, dtype=np.bool)
#         mask_inj = np.full([64, 64], False, dtype=np.bool)
#         pix_list = kwargs.get("pix_list")
#         for x in pix_list:
#             mask_tdc[x[0], x[1]] = True
#             mask_en[x[0] - 2:x[0] + 2, x[1] - 2:x[1] + 2] = True

        self.dut.write_hitor_mask(mask_tdc)
        self.dut.write_en_mask(mask_en)
        self.dut.write_inj_mask(mask_tdc)

        self.set_local_config(vth1=vth1)
        scan_range = np.arange(0.001, 1.21, 0.025)
        try:
            if pixel:
                mask_tdc[:, :] = False
                mask_en[:, :] = False
                mask_tdc[pixel[0], pixel[1]] = True
                mask_en[pixel[0] - 1:pixel[0] + 1, pixel[1] - 1:pixel[1] + 1] = True
                mask_en[mask_en_from_file == False] = False

                self.dut.write_hitor_mask(mask_tdc)
                self.dut.write_en_mask(mask_en)
                self.dut.write_inj_mask(mask_tdc)
                self.set_local_config(vth1=vth1)
                idx = 0
                print 'single', pixel
                for idx, k in enumerate(scan_range):
                    pulser['Pulser'].set_voltage(0., k, unit='V')
                    pulser['Pulser'].set_on_off("ON")
                    with self.readout(scan_param_id=idx):
                        self.dut['trigger'].set_en(True)
                        self.dut['tdc']['ENABLE'] = True
                        self.dut['TLU'].TRIGGER_ENABLE = 1

                        pulser['Pulser'].set_on_off("ON")
                        time.sleep(10.)
                        pulser['Pulser'].set_on_off("OFF")
    #                     self.dut['inj'].start()
    #
    #                     while not self.dut['inj'].is_done():
    #                         time.sleep(0.05)
                        t = 0
                        while not self.dut['trigger'].is_done():
                            time.sleep(0.05)
                            t += 1
                            if t > 1000:
                                print 'stuck in trigger wait!'

                        self.dut['tdc'].ENABLE = 0
                        self.dut['trigger'].set_en(False)
                        self.dut['TLU'].TRIGGER_ENABLE = 0
                        print k, self.fifo_readout.get_record_count() / 18
            else:
                print "working from pix_list"
                for pix in pix_list:
                    print
                    mask_tdc[:, :] = False
                    mask_en[:, :] = False
                    mask_tdc[pix[0], pix[1]] = True
                    mask_en[pix[0] - 1:pix[0] + 1, pix[1] - 1:pix[1] + 1] = True
                    mask_en[mask_en_from_file == False] = False

                    self.dut.write_hitor_mask(mask_tdc)
                    self.dut.write_en_mask(mask_en)
                    self.dut.write_inj_mask(mask_tdc)
                    self.set_local_config(vth1=vth1)
                    idx = 0
                    print pix
                    for idx, k in enumerate(scan_range):
                        pulser['Pulser'].set_voltage(0., k, unit='V')
                        pulser['Pulser'].set_on_off("ON")
                        with self.readout(scan_param_id=idx):
                            self.dut['trigger'].set_en(True)
                            self.dut['tdc']['ENABLE'] = True
                            self.dut['TLU'].TRIGGER_ENABLE = 1

                            pulser['Pulser'].set_on_off("ON")
                            time.sleep(1.)
                            pulser['Pulser'].set_on_off("OFF")
        #                     self.dut['inj'].start()
        #
        #                     while not self.dut['inj'].is_done():
        #                         time.sleep(0.05)
                            t = 0
                            while not self.dut['trigger'].is_done():
                                time.sleep(0.05)
                                t += 1
                                if t > 1000:
                                    print 'stuck in trigger wait!'

                            self.dut['tdc'].ENABLE = 0
                            self.dut['trigger'].set_en(False)
                            self.dut['TLU'].TRIGGER_ENABLE = 0
                            print k, self.fifo_readout.get_record_count()
    #             break
        except KeyboardInterrupt:
            logging.info('user exit, going to data analysis')
        scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Masks')
        self.h5_file.create_carray(scan_results, 'tdac_mask', obj=mask_tdac)
        self.h5_file.create_carray(scan_results, 'en_mask', obj=mask_en_from_file)

    def analyze(self, pixel=None):
        # get single hits, plot injection voltage vs tdc channel and inj voltage vs ToT
        h5_filename = self.output_filename + '.h5'
        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/hitor_calibration' + str(pixel) + '.pdf'
        pp = PdfPages(pdfName)

        with tb.open_file(h5_filename, 'r+') as io_file_h5:
            meta_data = io_file_h5.root.meta_data[:]
            raw_data = io_file_h5.root.raw_data[:]
            hit_data = self.dut.interpret_raw_data_w_tdc(raw_data, meta_data)
            io_file_h5.create_table(io_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)
            if not pixel:
                for pix in pix_list:
                    data_singles = anal.singular_hits_tdc_pix_flav(hit_data=hit_data)
                    fig1 = DGC_plotting.hitor_calibration(h5_file=h5_filename, hit_data=data_singles[(
                        data_singles['col'] == pix[0]) & (data_singles['row'] == pix[1])], pixel=pix)

                    pp.savefig(fig1, layout='tight')
                    plt.clf()
            else:
                data_singles = anal.singular_hits_tdc_pix_flav(hit_data=hit_data)
                fig1 = DGC_plotting.hitor_calibration(h5_file=h5_filename, hit_data=data_singles[(
                    data_singles['col'] == pixel[0]) & (data_singles['row'] == pixel[1])], pixel=pixel)

                pp.savefig(fig1, layout='tight')
                plt.clf()
#         fig = DGC_plotting.hitor_calibration(h5_file=h5_filename)
#         pp.savefig(fig, layout='tight')
#         plt.clf()
        pp.close()
        print self.output_filename[-11]


if __name__ == "__main__":
    test = HitOrCalib()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    for pix in pix_list:
        test.start(name=pix, pixel=pix, **local_configuration)
        test.analyze(pixel=pix)
