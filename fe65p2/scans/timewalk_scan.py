from fe65p2.scan_base import ScanBase
import fe65p2.plotting as plotting
import fe65p2.analysis as analysis
import time
import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, save, show
from bokeh.models.layouts import Column, Row
import yaml
from basil.dut import Dut
import logging
import os
import itertools
import serial
import fe65p2.DGC_plotting as DGC_plotting
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfFile
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

local_configuration = {
    "mask_steps": 1,
    "repeat_command": 101,
    # [0.05, 0.55, 0.01], #[0.005, 0.30, 0.01], # [0.01, 0.2, 0.01],# [0.01, 0.20, 0.01], #[0.005, 0.2, 0.005],
    "scan_range": [0.55, 0.6, 0.05],  # V
    "columns": [True] * 2 + [False] * 14,
    #"mask_filename": '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/20171121_162820_threshold_scan.h5',
    "pix_list": [(3, 3)],
    # DAC parameters
    "PrmpVbpDac": 36,
    "vthin1Dac": 255,
    "vthin2Dac": 0,
    "vffDac": 24,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 110
}


class TimewalkScan(ScanBase):
    scan_id = "timewalk_scan"

    def scan(self, mask_steps=4, repeat_command=101, columns=[True] * 16, pix_list=[], scan_range=[], mask_filename='', **kwargs):
        '''Scan loop
        This scan is to measure time walk. The charge injection can be driven by the GPAC or an external device.
        In the latter case the device is Agilent 33250a connected through serial port.
        The time walk and TOT are measured by a TDC module in the FPGA.
        The output is an .h5 file (data) and an .html file with plots.

        To perform a proper timewalk scan a mask_filename i.e. the output of the tuned threshold scan has to be provided.
        '''

        def load_vthin1Dac(mask):
            if os.path.exists(mask):
                in_file = tb.open_file(mask, 'r')
                dac_status = yaml.load(in_file.root.meta_data.attrs.dac_status)
                vthrs1 = dac_status['vthin1Dac']
                logging.info("Loaded vth1 from noise scan: %s", str(vthrs1))
                return int(vthrs1)
            else:
                return 110

        vth1 = load_vthin1Dac(mask_filename)

        # Offset needed, why?
        self.dut['INJ_LO'].set_voltage(0.2, unit='V')

        self.set_local_config()
        self.dut.set_for_configuration

        self.dut.start_up()

        self.dut['global_conf']['OneSr'] = 1

        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(
            [True] * 16)  # (columns)
        self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray([True] * 16)
        self.dut.write_global()

        self.dut['pixel_conf'].setall(False)
        self.dut.write_pixel()
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut.write_global()
        self.dut['global_conf']['InjEnLd'] = 0

        mask_en = np.full([64, 64], False, dtype=np.bool)
        mask_tdac = np.full([64, 64], 16, dtype=np.uint8)

        for inx, col in enumerate(columns):
            if col:
                mask_en[inx * 4:(inx + 1) * 4, :] = True

        if mask_filename:
            logging.info('Using pixel mask from file: %s', mask_filename)

            with tb.open_file(mask_filename, 'r') as in_file_h5:
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                mask_en = in_file_h5.root.scan_results.en_mask[:]

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['OneSr'] = 1
        self.dut.write_global()

        self.dut['inj'].set_delay(50000)  # 1 zero more
        self.dut['inj'].set_width(1000)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(400 - 4)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(False)

        # for the set_en command
        # If true: The pulse comes with a fixed delay with respect to the external trigger (EXT_START).
        # If false: The pulse comes only at software start.

        logging.debug('Enable TDC')
        self.dut['tdc']['RESET'] = True
        self.dut['tdc']['EN_TRIGGER_DIST'] = True
        self.dut['tdc']['ENABLE_EXTERN'] = False
        self.dut['tdc']['EN_ARMING'] = False
        self.dut['tdc']['EN_INVERT_TRIGGER'] = False
        self.dut['tdc']['EN_INVERT_TDC'] = False
        self.dut['tdc']['EN_WRITE_TIMESTAMP'] = True

        self.pixel_list = pix_list

        p_counter = 0

        # Calculate scan parameters from given parameter range
        scan_pars = np.arange(scan_range[0], scan_range[1], scan_range[2])

        for pix in pix_list:
            mask_en = np.full([64, 64], False, dtype=np.bool)
            mask_en[pix[0], pix[1]] = True
            self.dut.write_en_mask(mask_en)
            self.dut.write_inj_mask(mask_en)

            self.inj_charge = []
            for idx, par in enumerate(scan_pars):
                self.dut['INJ_HI'].set_voltage(float(par), unit='V')
                self.inj_charge.append(float(par) * 1000.0 * analysis.cap_fac())

                time.sleep(0.5)

                with self.readout(scan_param_id=idx):
                    logging.info('Scan Parameter: %f', par)

                    

                    self.dut['global_conf']['vthin1Dac'] = 255
                    self.dut['global_conf']['vthin2Dac'] = 0
                    self.dut['global_conf']['preCompVbnDac'] = 50
                    self.dut['global_conf']['PrmpVbpDac'] = 80

                    time.sleep(0.1)
                    self.dut['global_conf']['vthin1Dac'] = int(vth1)
                    self.dut['global_conf']['vthin2Dac'] = int(kwargs.get('vthin2Dac', 0))
                    self.dut['global_conf']['PrmpVbpDac'] = int(kwargs.get('PrmpVbpDac', 36))
                    self.dut['global_conf']['preCompVbnDac'] = int(kwargs.get('preCompVbnDac', 110))
                    self.dut.write_global()
                    time.sleep(0.1)

                    #self.dut['global_conf']['PrmpVbnFolDac'] = kwargs['PrmpVbnFolDac']
                    #self.dut['global_conf']['vbnLccDac'] = kwargs['vbnLccDac']
                    #self.dut['global_conf']['compVbnDac'] = kwargs['compVbnDac']
                    #self.dut['global_conf']['preCompVbnDac'] = kwargs['preCompVbnDac']
                    # self.dut.write_global()
                    # time.sleep(0.1)
                    # self.dut.write_global()
                    # time.sleep(0.1)

                    self.dut['tdc']['ENABLE'] = True
                    self.dut['inj'].start()

                    while not self.dut['inj'].is_done():
                        # time.sleep(0.05)
                        pass

                    while not self.dut['trigger'].is_done():
                        # time.sleep(0.05)
                        pass

                    self.dut['tdc'].ENABLE = 0
            p_counter += 1


    def tdc_table(self, scanrange):
        h5_filename = self.output_filename + '.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]
            if (meta_data.shape[0] == 0):
                print 'empty output'
                return
            repeat_command = in_file_h5.root.meta_data.attrs.kwargs
            a = repeat_command.rfind("repeat_command: ")
            repeat_command = repeat_command[a + len("repeat_command: "):a + len("repeat_command: ") + 7]
            a = repeat_command.rfind("\n")
            repeat_command = int(repeat_command[0:a])
            param, index = np.unique(meta_data['scan_param_id'], return_index=True)
            pxl_list = []
            for p in param:
                pix_no = int(p) / int(len(self.inj_charge))
                pxl_list.append(self.pixel_list[pix_no][0] * 64 + self.pixel_list[pix_no][1])
            index = index[1:]
            index = np.append(index, meta_data.shape[0])
            index = index - 1
            stops = meta_data['index_stop'][index]
            split = np.split(raw_data, stops)
            avg_tdc = []
            avg_tdc_err = []
            avg_del = []
            avg_del_err = []
            hits = []
            deletelist = ()
            for i in range(len(split[:-1])):  # loop on pulses
                rwa_data_param = split[i]
                tdc_data = rwa_data_param & 0xFFF  # take last 12 bit
                tdc_delay = (rwa_data_param & 0x0FF00000) >> 20
                counter = 0.0
                TOT_sum = 0.0
                DEL_sum = 0.0
                if (tdc_data.shape[0] == 0 or tdc_data.shape[0] == 1):
                    counter = 1.0
                for j in range(tdc_data.shape[0]):  # loop on repeats
                    if (j > 0):
                        counter += 1
                        TOT_sum += tdc_data[j]
                        DEL_sum += tdc_delay[j]
                if (counter > 1):
                    hits.append(counter)
                    avg_tdc.append((float(TOT_sum) / float(counter)) * 1.5625)
                    avg_tdc_err.append(1.5625 / (np.sqrt(12.0 * counter)))
                    avg_del.append((float(DEL_sum) / float(counter)) * 1.5625)
                    avg_del_err.append(1.5625 / (np.sqrt(12.0 * counter)))
                else:
                    deletelist = np.append(deletelist, i)
            pxl_list = np.delete(pxl_list, deletelist)
            newpix = [0]
            pix_no_old = pxl_list[0]
            runparam = 0
            for p in pxl_list:
                if p != pix_no_old:
                    newpix = np.append(newpix, runparam)
                pix_no_old = p
                runparam = runparam + 1
                addedvalues = 0
            for pixels in range(len(newpix)):
                missingvalues = 0
                if newpix[pixels] == newpix[-1]:
                    missingvalues = scanrange - abs(newpix[pixels] + addedvalues - len(hits))
                else:
                    if abs(newpix[pixels] - newpix[pixels + 1]) < scanrange:
                        missingvalues = scanrange - abs(newpix[pixels] - newpix[pixels + 1])
                if missingvalues != 0:
                    hits = np.insert(hits, newpix[pixels] + addedvalues, np.zeros(missingvalues))
                    avg_tdc = np.insert(avg_tdc, newpix[pixels] + addedvalues, np.zeros(missingvalues))
                    avg_tdc_err = np.insert(avg_tdc_err, newpix[pixels] + addedvalues, np.zeros(missingvalues))
                    avg_del = np.insert(avg_del, newpix[pixels] + addedvalues, np.zeros(missingvalues))
                    avg_del_err = np.insert(avg_del_err, newpix[pixels] + addedvalues, np.zeros(missingvalues))
                    pxl_list = np.insert(pxl_list, newpix[pixels] + addedvalues,
                                         (pxl_list[newpix[pixels] + addedvalues]) * np.ones(missingvalues))
                addedvalues = addedvalues + missingvalues
            injections = []
            for pixels in range(int(len(pxl_list) / len(self.inj_charge))):
                for i in range(len(self.inj_charge)):
                    injections = np.append(injections, self.inj_charge[i])
            pix, stop = np.unique(pxl_list, return_index=True)
            stop = np.sort(stop)
            stop = list(stop)
            stop.append(len(avg_tdc))
            repeat_command_dic={}
            repeat_command_dic['repeat_command']=repeat_command
            avg_tab = np.rec.fromarrays([injections, pxl_list, hits, avg_tdc, avg_tdc_err, avg_del, avg_del_err],
                                        dtype=[('charge', float), ('pixel_no', int), ('hits', int),
                                               ('tot_ns', float), ('err_tot_ns', float), ('delay_ns', float),
                                               ('err_delay_ns', float)])
            tdc_table=in_file_h5.create_table(in_file_h5.root, 'tdc_data', avg_tab, filters=self.filter_tables)
            tdc_table.attrs.repeat_command = repeat_command_dic
            thresholds = ()
            expfit0 = ()
            expfit1 = ()
            expfit2 = ()
            expfit3 = ()
            pixels = ()
            for i in range(len(stop) - 1):
                s1 = int(stop[i])
                s2 = int(stop[i + 1])
                A, mu, sigma = analysis.fit_scurve(hits[s1:s2], injections[s1:s2],repeat_command)
                if np.max(hits[s1:s2]) > (repeat_command + 200):  # or mu > 3000:
                    thresholds = np.append(thresholds, 0)
                    expfit0 = np.append(expfit0, 0)
                    expfit1 = np.append(expfit1, 0)
                    expfit2 = np.append(expfit2, 0)
                    expfit3 = np.append(expfit3, 0)
                    pixels = np.append(pixels, pxl_list[s1])
                    continue
                for values in range(s1, s2):
                    if injections[values] >= 5 / 4 * mu:
                        s1 = values
                        break
                numberer = 0
                hitvaluesold = hits[-1]
                for hitvalues in hits[s1:s2]:
                    if abs(hitvalues - hitvaluesold) <= 1 and hitvalues != 0:
                        break
                    numberer = numberer + 1
                    hitvaluesold = hitvalues
                if numberer == len(avg_del[s1:s2]):
                    numberer = 0
                expfit = analysis.fit_exp(injections[s1:s2], avg_del[s1:s2], mu, abs(numberer))
                startexp = -expfit[0] * np.log((25.0 + np.min(avg_del[s1:s2]) - expfit[3]) / expfit[2]) - expfit[1]
                if np.isnan(startexp) or startexp >= 2000:
                    startexp = 0
                thresholds = np.append(thresholds, startexp)
                expfit0 = np.append(expfit0, expfit[0])
                expfit1 = np.append(expfit1, expfit[1])
                expfit2 = np.append(expfit2, expfit[2])
                expfit3 = np.append(expfit3, expfit[3])
                pixels = np.append(pixels, pxl_list[s1])
            thresh = np.rec.fromarrays([pixels, thresholds, expfit0, expfit1, expfit2, expfit3],
                                       dtype=[('pixel_no', int), ('td_threshold', float),
                                              ('expfit0', float), ('expfit1', float), ('expfit2', float),
                                              ('expfit3', float)])
            in_file_h5.create_table(in_file_h5.root, 'td_threshold', thresh, filters=self.filter_tables)
        p1, p2, single_scan = plotting.plot_timewalk(h5_filename)
        output_file(self.output_filename + '.html', title=self.run_name)
        status = plotting.plot_status(h5_filename)
        save(Row(Column(p1, p2, status), single_scan))
        #show(p1)

   

if __name__ == "__main__":
    timewalk_scan = TimewalkScan()
    timewalk_scan.start(**local_configuration)

    print timewalk_scan.output_filename

    with tb.open_file(timewalk_scan.output_filename + ".h5") as in_file:
        raw = in_file.root.raw_data[:]
        tdc_data = raw & 0xFFF  # take last 12 bit
        plt.hist(tdc_data, bins=100)
        plt.title("N %d" % tdc_data.sum())
        plt.show()
        