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
import fe65p2.analysis as anal


yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip6.yaml'


local_configuration = {
    "max_data_count": 10000,
    "quad_columns": [True] * 16 + [False] * 0 + [True] * 0 + [False] * 0,
}

pixel_flav_list = ['nw15', 'nw20', 'nw25', 'nw30', 'dnw15', 'dnw20', 'dnw25', 'dnw30']

pixel_flav_dict = {'nw15': [[2, 2], [30, 7]],
                   'nw20': [[2, 10], [30, 15]],
                   'nw25': [[2, 18], [30, 23]],
                   'nw30': [[2, 25], [30, 61]],
                   'dnw15': [[33, 2], [61, 7]],
                   'dnw20': [[33, 10], [61, 15]],
                   'dnw25': [[33, 18], [61, 23]],
                   'dnw30': [[33, 25], [61, 61]]}

fe0 = [[0, 0], [7, 63]]
fe1 = [[8, 0], [15, 63]]
fe2 = [[16, 0], [23, 63]]
fe3 = [[24, 0], [31, 63]]
fe4 = [[32, 0], [39, 63]]
fe5 = [[40, 0], [47, 63]]
fe6 = [[48, 0], [55, 63]]
fe7 = [[56, 0], [63, 63]]

pix_list = [[11, 6], [10, 12], [11, 20], [9, 37], [44, 5], [44, 12], [41, 19], [42, 42],
            [5, 25], [10, 25], [19, 25], [29, 25], [33, 25], [46, 24], [49, 25], [60, 25],
            [2, 7], [10, 7], [20, 7], [29, 4], [35, 5], [42, 6], [54, 6], [58, 6],
            [3, 49], [12, 50], [19, 54], [24, 55], [37, 51], [45, 46], [49, 49], [58, 44]]


class TDCSrcCalib(ScanBase):
    scan_id = "tdc_src_calib"

    def scan(self, **kwargs):
        # pixels in column, row format

        try:  # pulser
            pulser = Dut(ScanBase.get_basil_dir(self) + '/examples/lab_devices/agilent33250a_pyserial.yaml')
            pulser.init()
            logging.info('Connected to ' + str(pulser['Pulser'].get_info()))
#             pulser['Pulser'].set_usr_func("FEI4_PULSE")
        except:
            pass

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
        _, mask_tdac, vth1 = noise_cols.combine_prev_scans(
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
        mask_en = np.full([64, 64], False, dtype=np.bool)
#         mask_inj = np.full([64, 64], False, dtype=np.bool)

        for x in pix_list:
            mask_tdc[x[0], x[1]] = True
            mask_en[x[0] - 2:x[0] + 2, x[1] - 2:x[1] + 2] = True

        self.dut.write_hitor_mask(mask_tdc)
        self.dut.write_en_mask(mask_en)
        self.dut.write_inj_mask(mask_tdc)

        self.set_local_config(vth1=vth1)
        time.sleep(2.0)

        with self.readout():
            try:
                self.dut['trigger'].set_en(True)
                self.dut['tdc']['ENABLE'] = True
                self.dut['TLU'].TRIGGER_ENABLE = 1
                repeat_loop = 6 * 3600
                sleep_time = 1.
                pbar = tqdm(range(repeat_loop))
                for _ in pbar:

                    time.sleep(sleep_time)
                    count_loop = self.fifo_readout.get_record_count() - count_old
                    pbar.set_description("Counts/pix %s " % str(np.round(self.fifo_readout.get_record_count() / (18. * len(pix_list)), 5)))
                    count_old = self.fifo_readout.get_record_count()

                    while not self.dut['trigger'].is_done():
                        time.sleep(0.05)

                self.dut['tdc'].ENABLE = 0
                self.dut['trigger'].set_en(False)
                self.dut['TLU'].TRIGGER_ENABLE = 0
            except:
                while not self.dut['trigger'].is_done():
                    time.sleep(0.05)

                self.dut['tdc'].ENABLE = 0
                self.dut['trigger'].set_en(False)
                self.dut['TLU'].TRIGGER_ENABLE = 0
        self.dut.set_for_configuration()

    def analyze(self):
        h5_filename = self.output_filename + '.h5'
        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/tdc_src_calib_Am.pdf'
        pp = PdfPages(pdfName)

        with tb.open_file(h5_filename, 'r+') as io_file_h5:
            meta_data = io_file_h5.root.meta_data[:]
            raw_data = io_file_h5.root.raw_data[:]
            hit_data = self.dut.interpret_raw_data_w_tdc(raw_data, meta_data)
#             print hit_data['tdc']
            io_file_h5.create_table(io_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)
            data_singles = anal.singular_hits_tdc_pix_flav(hit_data=hit_data)
            fig_list = DGC_plotting.tdc_src_spectrum(h5_file=h5_filename, hit_data=data_singles, pixel_list=pix_list, src_name='Am')
            for fig1 in fig_list:
                pp.savefig(fig1, layout='tight')
                plt.clf()
        fig = DGC_plotting.tdc_src_spectrum(h5_file=h5_filename)
        pp.savefig(fig[0], layout='tight')
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


if __name__ == "__main__":
    test = TDCSrcCalib()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    test.start(**local_configuration)
    test.analyze()
