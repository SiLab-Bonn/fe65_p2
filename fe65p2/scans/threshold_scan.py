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
from bokeh.models.layouts import Column, Row

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfFile
import matplotlib.pyplot as plt


from progressbar import ProgressBar
from basil.dut import Dut
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

local_configuration = {
    "columns": [True] * 16,
    #   DAC parameters
    "PrmpVbpDac": 36,
    "vthin1Dac": 60,
    "vthin2Dac": 0,
    "vffDac": 42,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 50,

    # "PrmpVbpDac": 36,
    # "vthin1Dac": 70,
    # "vthin2Dac": 0,
    # "vffDac": 24,
    # "PrmpVbnFolDac": 51,
    # "vbnLccDac": 1,
    # "compVbnDac": 25,
    # "preCompVbnDac": 110,

    #   thrs scan
    "mask_steps": 4,
    "repeat_command": 100,
    "scan_range": [0.05, 0.7, 0.01],
    "mask_filename": '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20180111_170315_noise_tuning.h5',
    "TDAC": 16
}


class ThresholdScan(ScanBase):
    scan_id = "threshold_scan"

    def scan(self, mask_steps=1, TDAC=16, scan_range=[0.0, 1.0, 0.02], repeat_command=1000, mask_filename='', **kwargs):
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
#         def load_vthin1Dac(mask):
#             if os.path.exists(mask):
#                 in_file = tb.open_file(mask, 'r')
#                 dac_status = yaml.load(in_file.root.meta_data.attrs.dac_status)
#                 vthrs1 = dac_status['vthin1Dac'] + 4
#                 logging.info(
#                     "Loaded vth1 from noise scan + 10: %d", vthrs1 + 10)
#                 return vthrs1 + 10
#             else:
#                 return kwargs['vthin1Dac']

#         vth1 = load_vthin1Dac(mask_filename)
        vth1 = kwargs.get("vthin1Dac", 100)

        inj_factor = 1.0

        try:  # pulser
            pulser = Dut(ScanBase.get_basil_dir(self) +
                         '/examples/lab_devices/agilent33250a_pyserial.yaml')
            pulser.init()
            logging.info('Connected to ' + str(pulser['Pulser'].get_info()))
            pulser['Pulser'].set_usr_func("FEI4_PULSE")
#             pulser['Pulser'].set_output_pol("NORM")
            pulser_test = True
            # pulser['Pulser'].set_burst_mode("TRIG")
            # pulser['Pulser'].burst_cycles("1.0")
            # pulser['Pulser'].set_voltage(0.2, 1., unit='V')
            # rise = 5e-9
            # pulser['Pulser'].set_rise_time("%s" % rise)
            # logging.info('Pulse rise time set to %s' % rise)
        except:
            # INJ_LO = 0.2
            #inj_factor = 2.0
            pulser_test = False
            logging.info(
                'External injector not connected. Switch to internal one')

        logging.info('starting THRESHOLD SCAN')
        self.dut['global_conf']['PrmpVbpDac'] = kwargs['PrmpVbpDac']
        self.dut['global_conf']['vthin1Dac'] = vth1
        self.dut['global_conf']['vthin2Dac'] = kwargs['vthin2Dac']
        self.dut['global_conf']['vffDac'] = kwargs['vffDac']
        self.dut['global_conf']['PrmpVbnFolDac'] = kwargs['PrmpVbnFolDac']
        self.dut['global_conf']['vbnLccDac'] = kwargs['vbnLccDac']
        self.dut['global_conf']['compVbnDac'] = kwargs['compVbnDac']
        self.dut['global_conf']['preCompVbnDac'] = kwargs['preCompVbnDac']

        self.dut['global_conf']['vthin1Dac'] = 255
        self.dut['global_conf']['vthin2Dac'] = 0
        self.dut['global_conf']['preCompVbnDac'] = 50
        self.dut['global_conf']['PrmpVbpDac'] = 80

        self.dut.write_global()
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

        self.dut['global_conf']['OneSr'] = 1

        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        columns = kwargs['columns']
        # self.dut['global_conf']['OneSr'] = 0  #all multi columns in parallel
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        self.dut['pixel_conf'].setall(False)
        self.dut.write_pixel()
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut.write_global()
        self.dut['global_conf']['InjEnLd'] = 0

        mask_en = np.full([64, 64], False, dtype=np.bool)
        mask_tdac = np.full([64, 64], TDAC, dtype=np.uint8)

        for inx, col in enumerate(kwargs['columns']):
            if col:
                mask_en[inx * 4:(inx + 1) * 4, :] = True

        if mask_filename:
            logging.info('***** Using pixel mask from file: %s', mask_filename)

            with tb.open_file(str(mask_filename), 'r') as in_file_h5:
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                mask_en = in_file_h5.root.scan_results.en_mask[:]

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['OneSr'] = 0
        self.dut.write_global()
        pulse_width = 800  # unit: ns
        # this seems to be working OK problem is probably bad injection on GPAC
        # usually +0
        self.dut['inj'].set_delay(
            25 * pulse_width + 100)  # 25ns per clock cycle
        self.dut['inj'].set_width(100)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(395)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        pulser['Pulser'].set_pulse_period(pulse_width * 10**-9)
        mask_steps = kwargs.get("mask_steps", mask_steps)

        lmask = [1] + ([0] * (mask_steps - 1))
        lmask = lmask * ((64 * 64) / mask_steps + 1)
        lmask = lmask[:64 * 64]

        scan_range = np.arange(
            scan_range[0], scan_range[1], scan_range[2]) / inj_factor

        INJ_LO = scan_range[0]
        for idx, k in enumerate(scan_range):
            if pulser_test:
                pulser['Pulser'].set_voltage(
                    scan_range[0], scan_range[0] + k, unit='V')
                time.sleep(0.2)
            else:
                self.dut['INJ_HI'].set_voltage(float(INJ_LO + k), unit='V')

            bv_mask = bitarray.bitarray(lmask)

            #logging.info('Temperature: %s', str(self.dut['ntc'].get_temperature('C')))

            with self.readout(scan_param_id=idx):
                logging.info('Scan Parameter: %f (%d of %d)',
                             k, idx + 1, len(scan_range))
                pbar = ProgressBar(maxval=mask_steps).start()
                for i in range(mask_steps):
                    # look at bv mask for each step then maybe disable pixels around the injected ones
                    # disabling solves the problem but it doesnt say anything about why it was happening
                    # make sure to look at the bv mask before each injection
                    # and see if any pixels are injected twice
                    self.dut.set_for_configuration()

                    self.dut['pixel_conf'][:] = bv_mask
                    self.dut.write_pixel_col()
                    self.dut['global_conf']['InjEnLd'] = 1
                    #self.dut['global_conf']['PixConfLd'] = 0b11
                    self.dut.write_global()

                    bv_mask[1:] = bv_mask[0:-1]
                    bv_mask[0] = 0

                    self.set_local_config()

                    self.dut['inj'].start()
                    pbar.update(i)

                    while not self.dut['inj'].is_done():
                        time.sleep(0.05)

                    while not self.dut['trigger'].is_done():
                        time.sleep(0.05)
                        print 'waiting for trigger'
                print "recieved words: ", self.fifo_readout.get_record_count()

        scan_results = self.h5_file.create_group(
            "/", 'scan_results', 'Scan Masks')
        self.h5_file.create_carray(scan_results, 'tdac_mask', obj=mask_tdac)
        self.h5_file.create_carray(scan_results, 'en_mask', obj=mask_en)

    def analyze(self):

        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/threshold_scan_testing.pdf'
        pp = PdfPages(pdfName)
        print pp
        h5_filename = self.output_filename + '.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]

            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            in_file_h5.create_table(
                in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)

            occ = np.histogram2d(x=hit_data['col'], y=hit_data['row'],
                                 bins=(64, 64), range=((0, 64), (0, 64)))[0]

            in_file_h5.create_carray(in_file_h5.root, name='HistOcc', title='Occupancy Histogram',
                                     obj=occ)
            # self.meta_data_table.attrs.dac_status
        analysis.analyze_threshold_scan(h5_filename)
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
        threshm2, noisehm2 = DGC_plotting.thresh_pix_heatmap(h5_filename)
        pp.savefig(threshm2)
        plt.clf()
        pp.savefig(noisehm2)
        plt.clf()
        singlePixPolt, thresHM, thresVsPix, thresDist, noiseHM, noiseDist, chi2plot = DGC_plotting.scan_pix_hist(
            h5_filename)
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
        pp.savefig(chi2plot)
        plt.clf()
        t_dac_plot = DGC_plotting.t_dac_plot(h5_filename)
        pp.savefig(t_dac_plot)
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
#         save(Column(Row(occ_plot1, tot_plot1, lv1id_plot1),
#                     scan_pix_hist1, t_dac1, status_plot1))

#output_file(self.output_filename + '.html', title=self.run_name)
#save(Column(Row(occ_plot, tot_plot, lv1id_plot), scan_pix_hist, t_dac, status_plot))


if __name__ == "__main__":
    scan = ThresholdScan()
    scan.start(**local_configuration)
    scan.analyze()
