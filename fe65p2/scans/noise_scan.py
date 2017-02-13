from fe65p2.scan_base import ScanBase
import fe65p2.plotting as  plotting
import time

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, hplot, save, show
from bokeh.models.layouts import Column, Row
from progressbar import ProgressBar
import os

local_configuration = {
    "columns": [True]*2 +[False]*14,
    "stop_pixel_count": 4,
    "repeats" : 1000,
    #pars
    "PrmpVbpDac": 36,
    "vthin1Dac": 100,
    "vthin2Dac": 0,
    "vffDac" : 24,
    "PrmpVbnFolDac" : 51,
    "vbnLccDac" : 1,
    "compVbnDac":25,
    "preCompVbnDac" : 50,

}


class NoiseScan(ScanBase):
    scan_id = "noise_scan"

    def __init__(self):
        super(NoiseScan, self).__init__()
        self.vth1Dac = 0

    def scan(self, columns=[True] * 16, stop_pixel_count=4, repeats=100000, **kwargs):
        '''Scan loop
        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''
        logging.info('\e[31m Starting Noise Scan \e[0m')
        INJ_LO = 0.2
        self.dut['INJ_LO'].set_voltage(INJ_LO, unit='V')
        self.dut['INJ_HI'].set_voltage(INJ_LO, unit='V')

        self.dut['global_conf']['PrmpVbpDac'] = kwargs['PrmpVbpDac']
        self.dut['global_conf']['vthin1Dac'] = kwargs['vthin1Dac']
        self.dut['global_conf']['vthin2Dac'] = kwargs['vthin2Dac']
        self.dut['global_conf']['vffDac'] = kwargs['vffDac']
        self.dut['global_conf']['PrmpVbnFolDac'] = kwargs['PrmpVbnFolDac']
        self.dut['global_conf']['vbnLccDac'] = kwargs['vbnLccDac']
        self.dut['global_conf']['compVbnDac'] = kwargs['compVbnDac']
        self.dut['global_conf']['preCompVbnDac'] = kwargs['preCompVbnDac']

        self.dut.write_global()
        self.dut['control']['RESET'] = 0b01
        self.dut['control']['DISABLE_LD'] = 0
        self.dut['control']['PIX_D_CONF'] = 0
        self.dut['control'].write()

        self.dut['control']['CLK_OUT_GATE'] = 1
        self.dut['control']['CLK_BX_GATE'] = 1
        self.dut['control'].write()
        time.sleep(0.01)

        self.dut['control']['RESET'] = 0b11
        self.dut['control'].write()

        # write InjEnLd & PixConfLd to '1
        self.dut['pixel_conf'].setall(True)
        self.dut.write_pixel_col()
        self.dut['global_conf']['SignLd'] = 1
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0b1111
        self.dut['global_conf']['PixConfLd'] = 0b11
        self.dut.write_global()

        # write SignLd & TDacLd to '0
        self.dut['pixel_conf'].setall(False)
        self.dut.write_pixel_col()
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut['global_conf']['TDacLd'] = 0b1111
        self.dut['global_conf']['PixConfLd'] = 0b11
        self.dut.write_global()

        # test hit
        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0

        self.dut['global_conf']['OneSr'] = 1  # all multi columns in parallel
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray([True] * 16)  # (columns)
        # self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        #logging.info('Temperature: %s', str(self.dut['ntc'].get_temperature('C')))

        mask_en = np.zeros([64, 64], dtype=np.bool)
        #mask_tdac = np.ones([64, 64], dtype=np.uint8)
        mask_tdac = np.full([64,64], 16, dtype=np.uint8)

        for inx, col in enumerate(columns):
            if col:
                mask_en[inx * 4:(inx + 1) * 4, :] = True

        self.dut.write_en_mask(mask_en)

        mask_tdac[:, :] = 1
        self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        # exit()

        self.dut['trigger'].set_delay(100)  # this seems to be working OK problem is probably bad injection on GPAC
        self.dut['trigger'].set_width(1)  # try single
        self.dut['trigger'].set_repeat(repeats)
        self.dut['trigger'].set_en(False)

        np.set_printoptions(linewidth=150)

        idx = 0
        finished = False

        vthin1DacInc = 1
        mask_disable_count = 0
        iteration = 0


        self.vth1Dac = kwargs['vthin1Dac']

        while not finished:
            with self.readout(scan_param_id=self.vth1Dac, fill_buffer=True, clear_buffer=True):
                self.dut['global_conf']['vthin1Dac'] = self.vth1Dac
                self.dut['global_conf']['vthin2Dac'] =  kwargs['vthin2Dac']
                self.dut['global_conf']['preCompVbnDac'] = kwargs['preCompVbnDac']
                self.dut['global_conf']['PrmpVbpDac'] = kwargs['PrmpVbpDac']
                self.dut.write_global()
                time.sleep(0.1)

                logging.info('Scan iteration: %d (vthin1Dac = %d)', iteration,  self.vth1Dac)
                pbar = ProgressBar(maxval=10).start()

                for i in range(10):

                    self.dut['trigger'].start()

                    pbar.update(i)

                    while not self.dut['trigger'].is_done():
                        pass

            self.dut['global_conf']['vthin1Dac'] = kwargs['vthin1Dac'] #setting again to default values
            self.dut['global_conf']['vthin2Dac'] = kwargs['vthin2Dac'] #values previously hardcoded
            self.dut['global_conf']['preCompVbnDac'] = kwargs['preCompVbnDac']
            self.dut['global_conf']['PrmpVbpDac'] = kwargs['PrmpVbpDac']

            self.dut.write_global()

            dqdata = self.fifo_readout.data
            data = np.concatenate([item[0] for item in dqdata])
            hit_data = self.dut.interpret_raw_data(data)
            hits = hit_data['col'].astype(np.uint16)
            hits = hits * 64
            hits = hits + hit_data['row']
            value = np.bincount(hits)
            value = np.pad(value, (0, 64 * 64 - value.shape[0]), 'constant')

            logging.info(
                'mean_tdac=' + str(np.mean(mask_tdac[mask_en == True])) + ' disabled=' + str(mask_disable_count)
                + ' hist_tdac=' + str(np.bincount(mask_tdac[mask_en == True])))

            nz = np.nonzero(value)

            corrected = False
            for i in nz[0]:
                col = i / 64
                row = i % 64

                if mask_tdac[col, row] < 31:  # (val > 1?)
                    mask_tdac[col, row] += 1
                    corrected = True

                if mask_tdac[col, row] == 31 and mask_en[col, row] == True:
                    mask_en[col, row] = False
                    mask_disable_count += 1

                logging.debug('col=%d row=%d val=%d mask=%d', i / 64, i % 64, value[i], mask_tdac[col, row])

            if self.vth1Dac < 1 or mask_disable_count >= stop_pixel_count:
                finished = True

            if not corrected:
                self.vth1Dac -= vthin1DacInc

            time.sleep(0.001)

            self.dut.write_en_mask(mask_en)
            self.dut.write_tune_mask(mask_tdac)

            iteration += 1
        self.dut['global_conf']['vthin1Dac'] = self.vth1Dac
        self.dut['global_conf']['vthin2Dac'] = kwargs['vthin2Dac']
        self.dut['global_conf']['preCompVbnDac'] = kwargs['preCompVbnDac']
        self.dut['global_conf']['PrmpVbpDac'] = kwargs['PrmpVbpDac']

        scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Results')
        self.h5_file.create_carray(scan_results, 'tdac_mask', obj=mask_tdac)
        self.h5_file.create_carray(scan_results, 'en_mask', obj=mask_en)
        logging.info('Final vthin1Dac value: %s', str(self.vth1Dac))
        self.final_vth1=self.vth1Dac

    def analyze(self):
        h5_filename = self.output_filename + '.h5'

        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]

            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            in_file_h5.create_table(in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)

        status_plot = plotting.plot_status(h5_filename)
        occ_plot, H = plotting.plot_occupancy(h5_filename)
        tot_plot, _ = plotting.plot_tot_dist(h5_filename)
        lv1id_plot, _ = plotting.plot_lv1id_dist(h5_filename)
        t_dac = plotting.t_dac_plot(h5_filename)
        # scan_pix_hist, _ = plotting.scan_pix_hist(h5_filename)

        output_file(self.output_filename + '.html', title=self.run_name)
        save(Column(Row(occ_plot, tot_plot), t_dac, status_plot))
        #show(t_dac)


if __name__ == "__main__":
    scan = NoiseScan()
    scan.start(**local_configuration)
    scan.analyze()
