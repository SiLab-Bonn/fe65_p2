'''
Updated threshold scan to change the injection mask with the mask 
step instead of the pixel configuration (old threshold scan)

Created by Daniel Coquelin 1/12/2018
'''


from fe65p2.scan_base import ScanBase
import fe65p2.plotting as plotting
import fe65p2.DGC_plotting as DGC_plotting
import time
import fe65p2.analysis as analysis
import yaml
import logging
import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, save
import fe65p2.scans.inj_tuning_columns as inj_cols
import fe65p2.scans.noise_tuning_columns as noise_cols
from bokeh.models.layouts import Column, Row
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

    # chip 4
    #     "PrmpVbpDac": 125,
    #     "vthin1Dac": 40,
    #     "vthin2Dac": 0,
    #     "vffDac": 73,
    #     "PrmpVbnFolDac": 61,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 45,
    #     "preCompVbnDac": 180,

    #   thrs scan
    "mask_steps": 4,
    "repeat_command": 100,
    "scan_range": [0.001, 0.24, 0.0085],
    "TDAC": 16
}


class ThresholdScan(ScanBase):
    scan_id = "threshold_scan"

    def scan(self, mask_steps=4, TDAC=16, repeat_command=1000, mask_filename='', **kwargs):
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

        inj_factor = 1.0

        try:  # pulser
            pulser = Dut(ScanBase.get_basil_dir(self) + '/examples/lab_devices/agilent33250a_pyserial.yaml')
            pulser.init()
            logging.info('Connected to ' + str(pulser['Pulser'].get_info()))
#             pulser['Pulser'].set_usr_func("FEI4_PULSE")
            pulser_test = True
        except:
            INJ_LO = 0.2
            self.dut['INJ_LO'].set_voltage(float(INJ_LO), unit='V')
            #inj_factor = 2.0
            pulser_test = False
            logging.info('*******External injector not connected. Switch to internal one')

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
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        mask_en = np.full([64, 64], False, dtype=np.bool)
        mask_tdac = np.full([64, 64], TDAC, dtype=np.uint8)
        mask_inj = np.full([64, 64], False, dtype=np.bool)
        mask_hitor = np.full([64, 64], True, dtype=np.bool)

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
#         if mask_filename:
#             logging.info('***** Using pixel mask from file: %s', mask_filename)
#
#             with tb.open_file(str(mask_filename), 'r') as in_file_h5:
#                 mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
#                 mask_en_from_file = in_file_h5.root.scan_results.en_mask[:]
#                 mask_tdac[mask_tdac == 32] = 31
# #                 dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
#                 vth1 = yaml.load(in_file_h5.root.meta_data.attrs.vth1) + 20
#                 logging.info("vth1: %s" % str(vth1))
#                 print vth1

#         mask_en = mask_en_from_file
        ex_pix_disable_list = kwargs.get("ex_pix_disable_list")
        mask_en_from_file = mask_en_from_file.reshape(4096)
        mask_en_from_file[ex_pix_disable_list] = False
        mask_en_from_file = mask_en_from_file.reshape(64, 64)
        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac.astype(np.uint8))
        self.dut.write_inj_mask(mask_inj)
        self.dut.write_hitor_mask(mask_hitor)

        pulse_width = 30000  # unit: ns
        # this seems to be working OK problem is probably bad injection on GPAC
        # usually +0
        wait_for_read = (16 + columns.count(True) * (4 * 64 / mask_steps) * 2) * (20 / 2) + 10000
        self.dut['inj'].set_delay(wait_for_read * 10)
        self.dut['inj'].set_width(1000)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(402)
        self.dut['trigger'].set_width(12)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        pulser['Pulser'].set_pulse_period(pulse_width * 10**-9)
        mask_steps = kwargs.get("mask_steps", mask_steps)
        scan_range = kwargs.get("scan_range")
        scan_range = np.arange(scan_range[0], scan_range[1], scan_range[2]) / inj_factor

        for idx, k in enumerate(scan_range):
            if pulser_test:
                pulser['Pulser'].set_voltage(0., k, unit='V')
            else:
                self.dut['INJ_HI'].set_voltage(float(INJ_LO + k), unit='V')

            time.sleep(0.2)
            with self.readout(scan_param_id=idx):
                logging.info('Scan Parameter %d of %d, Volts: %f, Electrons: %f',
                             idx + 1, len(scan_range), k, k * 1000 * analysis.cap_fac())

                for i in range(mask_steps):
                    self.dut.set_for_configuration()

                    mask_inj[:, :] = False
                    mask_en[:, :] = False

                    mask_inj = mask_inj.reshape(4096)
                    mask_inj[i::mask_steps] = True
                    mask_inj = mask_inj.reshape(64, 64)
                    mask_en = mask_inj
                    mask_hitor = mask_inj
                    mask_en[(mask_en_from_file == False) & (mask_inj == True)] = False
                    self.dut.write_inj_mask(mask_inj)
                    self.dut.write_en_mask(mask_en)
                    self.dut.write_hitor_mask()
                    self.set_local_config(vth1=vth1)

                    self.dut['inj'].start()
                    # time.sleep(0.3)
                    while not self.dut['inj'].is_done():
                        time.sleep(0.05)

                    while not self.dut['trigger'].is_done():
                        time.sleep(0.05)

                    print "finished mask_step: ", i, " words recieved: ", self.fifo_readout.get_record_count()
#                 time.sleep(0.5)
        scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Masks')
        self.h5_file.create_carray(scan_results, 'tdac_mask', obj=mask_tdac)
        self.h5_file.create_carray(scan_results, 'en_mask', obj=mask_en_from_file)

    def analyze(self):

        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/threshold_scan_testing.pdf'
        pp = PdfPages(pdfName)
        print pp
        h5_filename = self.output_filename + '.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]
            scan_args = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)

            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            in_file_h5.create_table(
                in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)

            occ = np.histogram2d(x=hit_data['col'], y=hit_data['row'], bins=(64, 64), range=((0, 64), (0, 64)))[0]
            print "occ average: ", occ.mean(), " occ std: ", occ.std(), " Num == repeats %: ", (float(occ[occ == scan_args['repeat_command']].shape[0]) / 4096) * 100
            print "num > repeats: ", occ[occ > scan_args['repeat_command']].shape[0]
            print np.where(occ > scan_args['repeat_command'] + 3)
            in_file_h5.create_carray(in_file_h5.root, name='HistOcc', title='Occupancy Histogram', obj=occ)

            # self.meta_data_table.attrs.dac_status
        analysis.analyze_threshold_scan(h5_filename)
        status_plot = DGC_plotting.plot_status(h5_filename)
        pp.savefig(status_plot)
        occ_plot = DGC_plotting.plot_occupancy(h5_filename)
        pp.savefig(occ_plot, layout="tight")
        plt.clf()
        tot_plot = DGC_plotting.plot_tot_dist(h5_filename)
        pp.savefig(tot_plot, layout="tight")
        plt.clf()
        lv1id_plot = DGC_plotting.plot_lv1id_dist(h5_filename)
        pp.savefig(lv1id_plot, layout="tight")
        plt.clf()
        threshm2, noisehm2 = DGC_plotting.thresh_pix_heatmap(h5_filename)
        pp.savefig(threshm2, layout="tight")
        plt.clf()
        pp.savefig(noisehm2, layout="tight")
        plt.clf()
        singlePixPolt, thresHM, thresVsPix, thresDist, noiseHM, noiseDist, noiseFlav, chi2plot = DGC_plotting.scan_pix_hist(h5_filename)
        pp.savefig(singlePixPolt, layout="tight")
        plt.clf()
        pp.savefig(thresHM, layout="tight")
        plt.clf()
        pp.savefig(thresVsPix, layout="tight")
        plt.clf()
        pp.savefig(thresDist, layout="tight")
        plt.clf()
        pp.savefig(noiseHM, layout="tight")
        plt.clf()
        pp.savefig(noiseDist, layout="tight")
        plt.clf()
        pp.savefig(noiseFlav, layout="tight")
        plt.clf()
        pp.savefig(chi2plot, layout="tight")
        plt.clf()
        t_dac_plot = DGC_plotting.t_dac_plot(h5_filename)
        pp.savefig(t_dac_plot, layout="tight")
        pp.close()

#         status_plot1 = plotting.plot_status(h5_filename)
#         occ_plot1, H = plotting.plot_occupancy(h5_filename)
#         tot_plot1, _ = plotting.plot_tot_dist(h5_filename)
#         lv1id_plot1, _ = plotting.plot_lv1id_dist(h5_filename)
#         scan_pix_hist1, _ = plotting.scan_pix_hist(h5_filename)
#         t_dac1 = plotting.t_dac_plot(h5_filename)
#
#         output_file(self.output_filename + '.html', title=self.run_name)
#
#         save(Column(Row(occ_plot1, tot_plot1, lv1id_plot1), scan_pix_hist1, t_dac1, status_plot1))

        #output_file(self.output_filename + '.html', title=self.run_name)
        # save(Column(Row(occ_plot, tot_plot, lv1id_plot),
        #            scan_pix_hist, t_dac, status_plot))


if __name__ == "__main__":
    scan = ThresholdScan()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan.start(**local_configuration)
    scan.analyze()
