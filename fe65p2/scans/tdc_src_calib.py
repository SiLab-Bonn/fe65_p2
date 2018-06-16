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
import fe65p2.DGC_plotting as DGC_plotting
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import matplotlib.pyplot as plt
from basil.dut import Dut
import fe65p2.scans.noise_tuning_columns as noise_cols
import yaml


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


class TDCSrcCalib(ScanBase):
    scan_id = "tdc_src_calib"
    # sample pixels to test with sources for charge calibration
    # (15,7), (11,13), (15,24), (12, 42), (45, 8), (46, 14), (45, 23), (46, 46)

    # non responsive pixels
    # (26, 25), (58, 36), (62, 55), (32, 48), (25, 30), (18, 14), (13, 61), (9, 59), (19, 50)
    def scan(self, repeat_command=10, columns=[True] * 16, pix_list=[(15, 7), (11, 13), (15, 24), (12, 42), (45, 8), (46, 14), (45, 23), (46, 46)], **kwargs):
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

        # enable all pixels but will only inject to selected ones see mask_inj
        mask_en = np.full([64, 64], False, dtype=np.bool)
        mask_tdac = np.full([64, 64], 15., dtype=np.uint8)
        mask_hitor = np.full([64, 64], False, dtype=np.bool)
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
        vth1 += 125
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

        dut['Pulser'].set_on_off("ON")
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
            dut['Pulser'].set_on_off("ON")
            count_old = 0
            total_old = 0
            mask_tdc = np.full([64, 64], False, dtype=np.bool)
            mask_en = np.full([64, 64], False, dtype=np.bool)
            mask_tdc[pix[0], pix[1]] = True
            mask_en[pix[0] - 2:pix[0] + 3, pix[1] - 2:pix[1] + 3] = True
            logging.info("pixel number: %s" % str(np.where(np.reshape(mask_tdc, 4096) == True)[0][0]))
            self.dut.write_hitor_mask(mask_tdc)
            self.dut.write_en_mask(mask_en)  # _from_file)
            self.dut.write_inj_mask(mask_en)

            time.sleep(0.1)
            self.set_local_config(vth1=vth1)
            time.sleep(2.0)

            dut['Pulser'].set_on_off("OFF")
            time.sleep(0.5)

            with self.readout(scan_param_id=pix[0] * 64 + pix[1]):
                self.dut['trigger'].set_en(True)
                self.dut['tdc']['ENABLE'] = True
                repeat_loop = 432
                sleep_time = 200.
                pbar = tqdm(range(repeat_loop))
                for loop in pbar:

                    time.sleep(sleep_time)
                    count_loop = self.fifo_readout.get_record_count() - count_old
                    # print "words received in loop", loop, ":", count_loop, "\tcount rate per second: ", count_loop / sleep_time
                    pbar.set_description("Counts/s %s " % str(np.round(count_loop / (sleep_time * 16), 5)))
                    count_old = self.fifo_readout.get_record_count()
                    while not self.dut['trigger'].is_done():
                        time.sleep(0.05)

                self.dut['tdc'].ENABLE = 0
                self.dut['trigger'].set_en(False)
            self.dut.set_for_configuration()
#         print "end for loop"

    def analyze(self):
        h5_filename = self.output_filename + '.h5'
        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/tdc_src_calib_Am.pdf'
        pp = PdfPages(pdfName)

        with tb.open_file(h5_filename, 'r+') as io_file_h5:
            meta_data = io_file_h5.root.meta_data[:]
            raw_data = io_file_h5.root.raw_data[:]
            hit_data = self.dut.interpret_raw_data_w_tdc(raw_data, meta_data)
            io_file_h5.create_table(io_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)
#             for i in np.unique(meta_data['scan_param_id']):
#                 data = hit_data[hit_data['scan_param_id'] == i]
#
#                 fig1 = DGC_plotting.tdc_src_spectrum(h5_file=h5_filename, hit_data=data)
#
#                 pp.savefig(fig1, layout='tight')
#                 plt.clf()
        fig = DGC_plotting.tdc_src_spectrum(h5_file=h5_filename)
        pp.savefig(fig, layout='tight')
        plt.clf()
        occ_plot = DGC_plotting.plot_occupancy(h5_filename)
        pp.savefig(occ_plot)
        plt.clf()
        tot_plot = DGC_plotting.plot_tot_dist(h5_filename)
        pp.savefig(tot_plot)
        plt.clf()
        lv1id_plot = DGC_plotting.plot_lv1id_dist(h5_filename)
        pp.savefig(lv1id_plot)

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
    test = TDCSrcCalib()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    test.start(**local_configuration)
    test.analyze()
