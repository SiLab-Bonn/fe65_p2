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

yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip3.yaml'

local_configuration = {
    "columns": [False] * 0 + [True] * 16 + [False] * 0,
    "stop_pixel_percent": 2,
    "pixel_disable_switch": 10,
    "repeats": 100000,
    # chip 3
    #"PrmpVbpDac": 120,
    #"vthin1Dac": 130,
    #"vthin2Dac": 0,
    #"vffDac": 92,
    #"PrmpVbnFolDac": 88,
    #"vbnLccDac": 1,
    #"compVbnDac": 90,
    #"preCompVbnDac": 140,

    # chip 4
    #     "PrmpVbpDac": 72,
    #     "vthin1Dac": 255,
    #     "vthin2Dac": 0,
    #     "vffDac": 73,
    #     "PrmpVbnFolDac": 61,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 55,
    #     "preCompVbnDac": 150,

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
        stop_noisy_pixel_count = (float(.5) / 100) * float(sum(columns)) * 4 * 64
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

        self.dut['trigger'].set_delay(395)
        self.dut['trigger'].set_width(1)
        self.dut['trigger'].set_repeat(repeats)
        self.dut['trigger'].set_en(False)

        np.set_printoptions(linewidth=180)

        finished = False
        mask_disable_count = 0
        iteration = 0
        vth1_step = 1

        self.vth1Dac = kwargs['vthin1Dac']
        self.set_local_config(vth1=self.vth1Dac)
        mask_en_hold = mask_en
        mask_tdac_hold = mask_tdac
        inner_loop_cnt = 0
        at_end = False
#         exit_code = 0
        while not finished:
            inner_loop_cnt += 1
            with self.readout(scan_param_id=self.vth1Dac, fill_buffer=True, clear_buffer=True):
                logging.info('Scan iteration: %d (vthin1Dac = %d)', iteration,  self.vth1Dac)

                self.set_local_config(vth1=self.vth1Dac)
                time.sleep(.2)

                self.dut['trigger'].start()
                while not self.dut['trigger'].is_done():
                    time.sleep(0.05)
            self.dut.set_for_configuration()

            dqdata = self.fifo_readout.data
            data = np.concatenate([item[0] for item in dqdata])
            hit_data = self.dut.interpret_raw_data(data)
            hits = hit_data['col'].astype(np.uint16)
            hits = hits * 64
            hits = hits + hit_data['row']
            value = np.bincount(hits, minlength=64 * 64)
            nz = np.nonzero(value)
            cnz = np.count_nonzero(value)

            corrected = False
#             abs_occ_limit = int(10**-5 * repeats) # can be used with 10**6 but 2 is fine for now
            abs_occ_limit = 1
            cnz = np.count_nonzero(value > abs_occ_limit)
            print "inner loop:", inner_loop_cnt, "occ_limit:", abs_occ_limit, "cnz(value > abs_occ_limit)", cnz, "nz", nz[0].shape[0]

            for i in nz[0]:
                col = i / 64
                row = i % 64
                if mask_tdac[col, row] < 32:
                    mask_tdac[col, row] += 1
                    corrected = True
                if mask_tdac[col, row] == 32 and mask_en[col, row] == True:
                    mask_en[col, row] = False
                    mask_disable_count += 1

            value = np.nan
            mean_tdac = np.mean(mask_tdac[(mask_en == True) & (mask_tdac > 0)])

            logging.info('mean_tdac >0: ' + str(np.round(mean_tdac, 2)) + ' disabled: ' +
                         str(mask_disable_count) + ' hist_tdac: ' + str(np.bincount(mask_tdac[mask_en == True])))

            if mean_tdac >= 14.5:
                at_end = True
            # stop criteria:
            # np.mean(mask_tdac[mask_en == True]) >=14.5
            # if num disables is > x% of chip/test area
            if mask_disable_count >= stop_pixel_count or self.vth1Dac < 1:
                finished = True
                self.vth1Dac += 1
                if mask_disable_count >= stop_pixel_count:
                    logging.info('exit from hitting max diable pixel percent, disabled: %s' % str(mask_disable_count))

                if self.vth1Dac < 1:
                    logging.info('exit from lowest vth1Dac')

                print np.where(np.reshape(mask_en, 4096) == False)

            elif mean_tdac >= 14.5 and cnz < 2:
                finished = True
                logging.info('exit from average tdac >=15, disabled count: %s' % str(mask_disable_count))
                print np.histogram(np.where(np.reshape(mask_en, 4096) == False), bins=8, range=(0, 4096))

            elif cnz <= stop_noisy_pixel_count and at_end == False:  # inner_loop_cnt >= 32 or
                corrected = False
#                 print "corrected loop reset", cnz, stop_noisy_pixel_count

            self.dut.write_en_mask(mask_en)
            self.dut.write_tune_mask(mask_tdac)

            if not corrected:
                inner_loop_cnt = 0
                mask_en_hold = mask_en
                mask_tdac_hold = mask_tdac
                self.dut.set_for_configuration()

                if mean_tdac <= 10.:  # and np.mean(mask_tdac[mask_en == True]) <= 6:
                    vth1_step = 3

                elif 10. < mean_tdac <= 13:
                    vth1_step = 2

                else:
                    vth1_step = 1

                self.vth1Dac -= vth1_step

            if mean_tdac >= 12.5:
                #                 self.dut['trigger'].set_delay(4000)
                self.dut['trigger'].set_repeat(1000000)
            elif 12.5 > mean_tdac >= 9.:
                #                 self.dut['trigger'].set_delay(3000)
                self.dut['trigger'].set_repeat(500000)
            else:
                self.dut['trigger'].set_repeat(50000)

            nz = np.nan
            iteration += 1

        scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Results')
        self.h5_file.create_carray(scan_results, 'tdac_mask', obj=mask_tdac_hold)
        self.h5_file.create_carray(scan_results, 'en_mask', obj=mask_en_hold)
        logging.info('Final vthin1Dac value: %s', str(self.vth1Dac))
        self.final_vth1 = self.vth1Dac
#         return self.final_vth1, mask_disable_count, tdac_hist

    def analyze(self):
        h5_filename = self.output_filename + '.h5'
#         pdfName = self.output_filename + '.pdf'  #
        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/noise_tuning_testing.pdf'
        pp = PdfPages(pdfName)

        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]
            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)

            in_file_h5.create_table(in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)
            mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]

            mask_en = in_file_h5.root.scan_results.en_mask[:]
            scan_kwargs = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
            scan_vth1 = yaml.load(in_file_h5.root.meta_data.attrs.vth1)
            columns = scan_kwargs['columns']
            mask_cols = np.full((64, 64), False, dtype=np.bool)
            for inx, col in enumerate(columns):
                if col:
                    mask_cols[inx * 4:(inx + 1) * 4, :] = True
            if float(scan_kwargs['stop_pixel_percent']) == (float(scan_kwargs['stop_pixel_percent']) / 100) * float(sum(scan_kwargs['columns'])) * 4 * 64:
                exit_code = 1
            elif scan_vth1 < 2:
                exit_code = 2
            else:
                exit_code = 0

            print "exit code:", exit_code

            occ = np.histogram2d(x=hit_data['col'], y=hit_data['row'],
                                 bins=(64, 64), range=((0, 64), (0, 64)))[0]

            in_file_h5.create_carray(in_file_h5.root, name='HistOcc', title='Occupancy Histogram',
                                     obj=occ)

        status_plot = DGC_plotting.plot_status(h5_filename)
        pp.savefig(status_plot)
        occ_plot = DGC_plotting.plot_occupancy(h5_filename)
        pp.savefig(occ_plot, layout='tight')
        plt.clf()
        tot_plot = DGC_plotting.plot_tot_dist(h5_filename)
        pp.savefig(tot_plot, layout='tight')
        plt.clf()
        lv1id_plot = DGC_plotting.plot_lv1id_dist(h5_filename)
        pp.savefig(lv1id_plot, layout='tight')
        plt.clf()
        tdac_hm, tdac_hist = DGC_plotting.tdac_heatmap(h5_file_name=None, en_mask_in=mask_en, tdac_mask_in=mask_tdac)
        pp.savefig(tdac_hm, layout='tight')
        plt.clf()
        pp.savefig(tdac_hist, layout='tight')
        plt.clf()
        t_dac_plot = DGC_plotting.t_dac_plot(h5_filename)
        pp.savefig(t_dac_plot, layout='tight')
        pp.close()

        #output_file(self.output_filename + '.html', title=self.run_name)
        #save(Column(Row(occ_plot, tot_plot), t_dac, status_plot))
        # show(t_dac)
        mask_disable_count = mask_en[(mask_en == False) & (mask_cols == True)].shape[0]
        tdac_hist = np.bincount(mask_tdac[mask_en == True])
        return scan_vth1, mask_disable_count, tdac_hist, exit_code

    def output_filename(self):
        return self.output_filename()


if __name__ == "__main__":
    scan = NoiseTuning()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan.start(**local_configuration)
    _, _, _, _ = scan.analyze()
