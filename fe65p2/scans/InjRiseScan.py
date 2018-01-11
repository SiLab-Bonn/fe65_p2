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
    "repeat_command": 4000,

    # DAC parameters
    "PrmpVbpDac": 36,
    "vthin1Dac": 80,
    "vthin2Dac": 0,
    "vffDac": 24,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 110,
    # can put more here, ex [(3,3), (2,16)], formal col,row (true column not qcols)
    "pix_list": [(2, 3)],
    "rise_list": [5e-9, 25e-9, 2e-9],
    # rise_list is in seconds
    "TDC": 15
}


class InjRiseScan(ScanBase):
    scan_id = "injRise_scan"

    def scan(self, repeat_command=10, columns=[True] * 16, pix_list=[(3, 3)], **kwargs):
        '''Scan loop
        parameters:
            mask (int) :                number of mask steps
            repeat_command (int) :      number of injections/measurements
            to sync the pulse generator and the boards need to have A: 'TX1 -> to Ext Input' and B: 'RXO -> to Sync/Trigger out'
        '''

        # load mask???
        logging.info('starting Injection Rise time scan')
        # pix_list = kwargs['pix_list']
        # columns = [True] + [False] * 15
        repeat_command = kwargs.get('repeat_command', repeat_command)
        print 'repeat_command: ', repeat_command

        # connect to the pulser to change rise time
        inj_factor = 1.0
        INJ_LO = 0.0
        try:
            pulser = Dut(ScanBase.get_basil_dir(self) +
                         '/examples/lab_devices/agilent33250a_pyserial.yaml')
            pulser.init()
            logging.info('Connected to ' + str(pulser['Pulser'].get_info()))
        except RuntimeError:
            INJ_LO = 0.2
            inj_factor = 2.0
            logging.info(
                'External injector not connected. Switch to internal one')
            self.dut['INJ_LO'].set_voltage(INJ_LO, unit='V')

        self.dut['INJ_LO'].set_voltage(0.2, unit='V')
        self.dut['INJ_HI'].set_voltage(1.0, unit='V')

        self.set_local_config()
        self.dut.set_for_configuration()

        # enable all pixels but will only inject to selected ones see mask_inj
        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], kwargs['TDC'], dtype=np.uint8)

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        self.dut.start_up()

        # enable inj pulse and trigger
        self.dut['inj'].set_delay(5000)
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

        # enable TDC
        logging.debug('Enable TDC')
        self.dut['tdc']['RESET'] = True
        self.dut['tdc']['EN_TRIGGER_DIST'] = True
        self.dut['tdc']['ENABLE_EXTERN'] = False
        self.dut['tdc']['EN_ARMING'] = False
        self.dut['tdc']['EN_INVERT_TRIGGER'] = False
        self.dut['tdc']['EN_INVERT_TDC'] = False
        self.dut['tdc']['EN_WRITE_TIMESTAMP'] = True
        rise_list = [i for i in kwargs.get('rise_list', [0.5, 1, 0.5])]
        rise_list = np.arange(rise_list[0], rise_list[1], rise_list[2])
        print "len rise: ", len(rise_list)

        # TODO: save the pixel list ot the h5 file
        # then inject to only those. need to seperate them in the data to be analized later
        # loop to set the different injections for the different pixels (if any)
        pix_list = kwargs.get('pix_list', pix_list)
        print pix_list
        for pix in pix_list:
            mask_inj = np.full([64, 64], False, dtype=np.bool)
            mask_inj[pix[0], pix[1]] = True
            self.dut.write_inj_mask(mask_inj)
            for i, rise in enumerate(rise_list):
                print ("%s" % rise)
                pulser['Pulser'].set_rise_time("%s" % rise)
                time.sleep(0.1)
                with self.readout(scan_param_id=i):
                    self.dut.set_for_configuration()
                    self.set_local_config()

                    self.dut['tdc']['ENABLE'] = True
                    self.dut['inj'].start()

                    while not self.dut['inj'].is_done() or not self.dut['trigger'].is_done():
                        time.sleep(0.02)

                    self.dut['tdc'].ENABLE = 0
            print "end of pixel loop"

    def analyze(self):
        h5_filename = self.output_filename + '.h5'
        with tb.open_file(h5_filename, 'r+') as io_file_h5:
            raw_data = io_file_h5.root.raw_data[:]
            meta_data = io_file_h5.root.meta_data[:]
            if (meta_data.shape[0] == 0):
                print 'empty output'
                return

            kwargs = yaml.load(io_file_h5.root.meta_data.attrs.kwargs)

            repeat_command = kwargs['repeat_command']
            pixels = kwarg.get('pix_list')
            rise_list = kwargs.get('rise_list')
            rise_list = np.arange(rise_list[0], rise_list[1], rise_list[2])
            # TODO: make the table with the following collumns
            # TDC data and delay (+ errors), charge, tot +err, pixel
            param = np.unique(meta_data['scan_param_id'])
            for i in param:
                # get data start and stop from meta_data then read the values in for each scan parameter
                # after sorting the data for each scan parameter need the avg and sigma for each, for both the tdc data and the delay
                # two tables, one with the arranged data and one with the averages
                # need to calc the charge as well..see old timewalk scan
                data_for_param_start = meta_data['index_start'][meta_data['scan_param_id'] == i]
                data_for_param_end = meta_data['index_end'][meta_data['scan_param_id'] == i]

                # read in raw data, convert tdc to charge/???, save with the rise time

                # TODO: need to implement code to test different pixels as well

                #     raw = io_file_h5.root.raw_data[:]

                #         tdc_data = raw & 0xFFF  # only want last 12 bit
                #         # cut out data from the second part of the pulse
                #         tdc_delay = (raw & 0x0FF00000) >> 20
                #         tdc_data = tdc_data[tdc_delay < 255][1:]

                #     print "tdc mean: ", tdc_data.mean(), " sigma: ", tdc_data.std(), " length: ", tdc_data.shape[0]
                #     print "tdc delay mean: ", tdc_delay[tdc_delay < 253].mean(), " sigma: ", tdc_delay[tdc_delay != 255].std()


if __name__ == "__main__":
    test = InjRiseScan()
    test.start(**local_configuration)
    test.analyze()
