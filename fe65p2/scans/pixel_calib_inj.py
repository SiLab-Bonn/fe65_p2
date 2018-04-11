'''
Pixel calibration of TDC with pulser

Created by Daniel Coquelin on 7/2/18
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
import fe65p2.scans.noise_tuning_columns as noise_cols
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from basil.dut import Dut

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip4.yaml'

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
    #     "PrmpVbpDac": 120,
    #     "vthin1Dac": 130,
    #     "vthin2Dac": 0,
    #     "vffDac": 92,
    #     "PrmpVbnFolDac": 88,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 90,
    #     "preCompVbnDac": 140,

    # chip 4
    #     "PrmpVbpDac": 125,
    #     "vthin1Dac": 40,
    #     "vthin2Dac": 0,
    #     "vffDac": 73,
    #     "PrmpVbnFolDac": 61,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 45,
    #     "preCompVbnDac": 180,

    # tdc calib scan
    "repeat_command": 100,
    "scan_range": np.array([400, 600, 800, 1000, 1200, 1400, 1600,
                            1800, 2000, 2200, 2400, 3000, 3500, 4000,
                            4500, 5000, 6000, 7000, 8000, 9000, 9600]),
    "mask_filename": '/home/daniel/Documents/InterestingPlots/chip3/20180321_112749_noise_tuning.h5',
    "TDAC": 16,
    "pixel_range": [0, 4096],
}


class PixelCalib(ScanBase):
    scan_id = "pixel_tdc_calib"

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

        # pulser
        pulser = Dut(ScanBase.get_basil_dir(self) + '/examples/lab_devices/agilent33250a_pyserial.yaml')
        pulser.init()
        logging.info('Connected to ' + str(pulser['Pulser'].get_info()))
#         pulser['Pulser'].set_usr_func("FEI4_PULSE")

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

        mask_en_test = np.reshape(mask_en, 4096)

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac.astype(np.uint8))
        self.dut.write_inj_mask(mask_inj)
        self.dut.write_hitor_mask(mask_hitor)

        # this seems to be working OK problem is probably bad injection on GPAC
        # usually +0
        self.dut['inj'].set_delay(111111)  # dealy betwean injections in 25ns unit
        self.dut['inj'].set_width(1000)
        self.dut['inj'].set_repeat(repeat_command)

        self.dut['inj'].set_en(False)  # Working?

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

        pulse_width = 30000  # unit: ns
        pulser['Pulser'].set_pulse_period(pulse_width * 10**-9)
#         flavor_scan_params = []
        # scan range params:
        # max -> 10 200
        # min -> 1 800
        # step -> 64
#         scan_range = kwargs.get("scan_range", [200, 9800, 300])  # stop at 10 000 because range saturates at 1.2...V==VDDA
#         scan_range = np.array([400, 600, 800, 1000, 1200, 1400, 1600,
#                                1800, 2000, 2200, 2400, 3000, 3500, 4000,
#                                4500, 5000, 6000, 7000, 8000, 9000, 9600]),
#         scan_range = list(np.linspace(scan_range[0], scan_range[1], scan_range[2]))
#         scan_range = np.arange(scan_range[0], scan_range[1], scan_range[2])
        # loop over all pixels, pix number = scan_parameter
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

                for idx, elecs in enumerate(scan_range):
                    with self.readout(scan_param_id=((scan_range.shape[0] * pix) + idx)):

                        #                     print elecs / (1000 * analysis.cap_fac())
                        pulser['Pulser'].set_voltage(0., elecs / (1000 * analysis.cap_fac()), unit='V')
                        # looping of pulses, need to enable the tdc for each one then record the data and then save it

                        self.dut['tdc']['ENABLE'] = True
                        self.dut['inj'].start()
#                         time.sleep(.1)

                        while not self.dut['inj'].is_done():
                            time.sleep(0.05)

                        while not self.dut['trigger'].is_done():
                            time.sleep(0.05)

                        self.dut['tdc'].ENABLE = False
                        logging.info('Injected electrons %s Words Received: %s' % (str(elecs), str(self.fifo_readout.get_record_count())))
#                         time.sleep(0.)
    #                     break
    #             break

    def analyze(self):
        # open file and import data
        h5_file = self.output_filename + '.h5'
        analysis.create_tdc_inj_table(h5_file)
        analysis.analyze_pixel_calib_inj(h5_file)
        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/pix_calib_inj_testing.pdf'
        pp = PdfPages(pdfName)
        # in here have the following:
        # pixel number | inj electrons | mu_data | sigma_data | mu_delay | sigma_delay |percent of repeats seen
        # need to get the following:
        # plot -> num elecs injected vs average tdc value (error bars are the sigmas)
        #         for this take a random sampling or ~
        # plot -> 2d hist of tdc delay vs tdc data
#         with tb.open_file(h5_file, 'r+') as in_file:
#             raw_data = in_file.root.raw_data[:]
#             meta_data = in_file.root.meta_data[:]
#
#             tdc_data = raw_data & 0x0FFF  # only want last 12 bit
#             tdc_delay = (raw_data & 0x0FF00000) >> 20
#
# #             print tdc_data
#             print "length tdc_data: ", tdc_data.shape[0]
# #             print "len tdc_delay >= 255:", tdc_delay[tdc_delay > 255].shape[0], tdc_delay[tdc_delay == 255].shape[0]
# #             print "length tdc_delay <255: ", tdc_delay[tdc_delay < 253].shape[0]
#             print "length tdc_data < 253:", tdc_data[tdc_delay < 253].shape[0]
#
# #             y, bins = np.histogram(tdc_delay, bins=255, range=(0, 255))
# #             x = (bins[1:] + bins[:-1]) / 2.
# #             plt.bar(x, height=y, width=1)
# #             plt.show()
#
#             y, bins = np.histogram(tdc_data[tdc_delay < 253], bins=175, range=(0, 175))
# #             print "nz data bins:", np.nonzero(y)
# #             print "data:", y[np.nonzero(y)]
#             delay, bins_d = np.histogram(tdc_delay, bins=max(tdc_delay), range=(0, max(tdc_delay)))
#             nz = np.nonzero(delay)
#             print "nz delay:", nz
#             print "delay hist:", delay[nz]
#
#             x = (bins[1:] + bins[:-1]) / 2.
#             #plt.bar(x, height=y, width=1)
#             plt.hist2d(tdc_data, tdc_delay, bins=(max(tdc_data), max(tdc_delay)))
#             plt.show()
        line_hm, chi2lin = DGC_plotting.pix_inj_calib_lines(h5_file)
        pp.savefig(line_hm, layout='tight')
        plt.clf()
        pp.savefig(chi2lin, layout='tight')
        plt.clf()
        fig1, hm1, hm2 = DGC_plotting.pix_inj_calib_tdc_v_elec(h5_file)
        pp.savefig(fig1, layout='tight')
        plt.clf()
        pp.savefig(hm1, layout='tight')
        plt.clf()
        pp.savefig(hm2, layout='tight')
        plt.clf()
        fig2 = DGC_plotting.pixel_inj_calib_delay_vs_data(h5_file)
        pp.savefig(fig2, layout='tight')
        plt.clf()
        pp.close()


if __name__ == "__main__":
    scan = PixelCalib()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan.start(**local_configuration)
    scan.analyze()
