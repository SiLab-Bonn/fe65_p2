''' TDAC tuning based on electronic noise.
    Tunes to the lowest possible threshold value. 
    No injection, only noise scan. 
    stop at pixel count, vthin1 at 0, or tdac average at 7
'''

from fe65p2.scan_base import ScanBase
import fe65p2.DGC_plotting as DGC_plotting
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time
import logging
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from basil.dut import Dut


local_configuration = {
    "columns": [False] * 0 + [True] * 16 + [False] * 0,
    "stop_pixel_percent": 1.5,
    "pixel_disable_switch": 10,
    "repeats": 1000,
    # chip 3
    "PrmpVbpDac": 36,
    "vthin1Dac": 120,
    "vthin2Dac": 0,
    "vffDac": 42,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 50,

}


class NoiseTuning(ScanBase):
    scan_id = "noise_tuning"

    def __init__(self):
        super(NoiseTuning, self).__init__()
        self.vth1Dac = 0

    def scan(self, stop_pixel_percent=2, pixel_disable_switch=4, repeats=100000, columns=None, **kwargs):
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

        self.dut.set_for_configuration()

        columns = kwargs.get('columns', columns)
        stop_pixel_percent = kwargs.get('stop_pixel_percent', stop_pixel_percent)
        stop_pixel_count = (float(stop_pixel_percent) / 100) * float(sum(columns)) * 4 * 64
        pixel_disable_switch = kwargs.get('pixel_disable_switch', pixel_disable_switch)
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
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        # self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        #logging.info('Temperature: %s', str(self.dut['ntc'].get_temperature('C')))

        mask_en = np.zeros([64, 64], dtype=np.bool)
        #mask_tdac = np.ones([64, 64], dtype=np.uint8)
        mask_tdac = np.full([64, 64], 16, dtype=np.uint8)

        for inx, col in enumerate(columns):
            if col:
                mask_en[inx * 4:(inx + 1) * 4, :] = True

        self.dut.write_en_mask(mask_en)

        mask_hitor = np.full([64, 64], False, dtype=np.bool)
        self.dut.write_hitor_mask(mask_hitor)

        mask_tdac[:, :] = 0
        self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        # exit()

        # this seems to be working OK problem is probably bad injection on GPAC
        self.dut['trigger'].set_delay(3000)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(repeats)
        self.dut['trigger'].set_en(False)

        np.set_printoptions(linewidth=180)

        finished = False

        vthin1DacInc = 1
        mask_disable_count = 0
        iteration = 0

        self.vth1Dac = kwargs['vthin1Dac']
        self.dut.set_for_configuration()

        while not finished:
            with self.readout(scan_param_id=self.vth1Dac, fill_buffer=True, clear_buffer=True):

                self.set_local_config(vth1=self.vth1Dac)

                logging.info('Scan iteration: %d (vthin1Dac = %d)', iteration,  self.vth1Dac)

                self.dut['trigger'].start()

                # time.sleep(1.)

                while not self.dut['trigger'].is_done():
                    time.sleep(0.01)

            # self.fifo_readout.stop(timeout=1.)

            dqdata = self.fifo_readout.data
            data = np.concatenate([item[0] for item in dqdata])
            hit_data = self.dut.interpret_raw_data(data)
            hits = hit_data['col'].astype(np.uint16)
            hits = hits * 64
            hits = hits + hit_data['row']
            value = np.bincount(hits)
            value = np.pad(value, (0, 64 * 64 - value.shape[0]), 'constant')

            # print mask_tdac[:8, :]
            # print mask_en[:8, :]

            logging.info('mean_tdac=' + str(np.mean(mask_tdac[mask_en == True])) + ' disabled=' +
                         str(mask_disable_count) + ' hist_tdac=' + str(np.bincount(mask_tdac[mask_en == True])))

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

                if mask_disable_count == pixel_disable_switch:
                    mask_en_out = mask_en
                    logging.info('saved enable mask as mask_en_out')
                    print 'saved mask_en_out'

                logging.debug('col=%d row=%d val=%d mask=%d', i / 64, i % 64, value[i], mask_tdac[col, row])

            # stop criteria:
            # np.mean(mask_tdac[mask_en == True]) >=7
            # if num disables is > 5% of chip/test area
            if self.vth1Dac < 1 or np.mean(mask_tdac[mask_en == True]) >= 15 or mask_disable_count >= stop_pixel_count:
                finished = True
                if self.vth1Dac < 1:
                    logging.info('exit from lowest vth1Dac')
                if np.mean(mask_tdac[mask_en == True]) >= 15:
                    logging.info('exit from average tdac >=15')
                if mask_disable_count >= stop_pixel_count:
                    logging.info('exit from hitting max diable pixel percent, disabled: %s' % str(mask_disable_count))

            # TODO: bin 16 doesnt get anything... all in bin 15

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
        self.h5_file.create_carray(scan_results, 'en_mask', obj=mask_en_out)
        logging.info('Final vthin1Dac value: %s', str(self.vth1Dac))
        self.final_vth1 = self.vth1Dac

    def analyze(self):
        h5_filename = self.output_filename + '.h5'
        pdfName = self.output_filename + '.pdf'  # '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/noise_tuning_testing.pdf'
        pp = PdfPages(pdfName)

        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]

            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            in_file_h5.create_table(in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)

            occ = np.histogram2d(x=hit_data['col'], y=hit_data['row'],
                                 bins=(64, 64), range=((0, 64), (0, 64)))[0]

            in_file_h5.create_carray(in_file_h5.root, name='HistOcc', title='Occupancy Histogram',
                                     obj=occ)

        status_plot = DGC_plotting.plot_status(h5_filename)
        pp.savefig(status_plot)
        occ_plot = DGC_plotting.plot_occupancy(h5_filename)
        pp.savefig(occ_plot)
        plt.clf()
        tot_plot = DGC_plotting.plot_tot_dist(h5_filename)
        pp.savefig(tot_plot)
        plt.clf()
        lv1id_plot = DGC_plotting.plot_lv1id_dist(h5_filename)
        pp.savefig(lv1id_plot)
        plt.clf()
        '''
        singlePixPolt, thresHM, thresVsPix, thresDist, noiseHM, noiseDist = DGC_plotting.scan_pix_hist(h5_filename)
        pp.savefig(singlePixPolt)
        plt.clf()
        pp.savefig(thresHM)
        plt.clf()
        pp.savefig(thresVsPix)
        plt.clf()
        pp.savefig(thresDist)
        plt.clf()
        pp.savefig(noiseHM)
        plt.clf()
        pp.savefig(noiseDist)
        plt.clf()
        '''
        t_dac_plot = DGC_plotting.t_dac_plot(h5_filename)
        pp.savefig(t_dac_plot)
        pp.close()

        #output_file(self.output_filename + '.html', title=self.run_name)
        #save(Column(Row(occ_plot, tot_plot), t_dac, status_plot))
        # show(t_dac)
    def output_filename(self):
        return self.output_filename()


if __name__ == "__main__":
    scan = NoiseTuning()
    scan.start(**local_configuration)
    scan.analyze()
