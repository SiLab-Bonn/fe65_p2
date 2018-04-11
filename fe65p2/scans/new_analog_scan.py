# updated/rewritten version of analog scan
# updated by Daniel Coquelin
from fe65p2.scan_base import ScanBase
import fe65p2.plotting as plotting
import time

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, show, vplot, hplot, save
from progressbar import ProgressBar
import os

local_configutaion = {
    "mask_steps": 4,
    "repeat_command": 100,

    # DAC parameters
    "PrmpVbpDac": 36,
    "vthin1Dac": 255,
    "vthin2Dac": 0,
    "vffDac": 24,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 50
}


class AnalogScan(ScanBase):
    scan_id = "analog_scan"

    def scan(self, mask_steps=4, repeat_command=100, columns=[True] * 16, *kwargs):
        # scan loop
        # need the parameters as follows:
        #     mask -> int -> number of mask steps
        #     repeat -> int -> number of injections
        self.dut['INJ_LO'].set_voltage(0.1, unit='V')
        self.dut['INJ_HI'].set_voltage(1.2, unit='V')

        self.dut['global_conf']['PrmpVbpDac'] = kwargs['PrmpVbpDac']
        self.dut['global_conf']['vthin1Dac'] = kwargs['vthin1Dac']
        self.dut['global_conf']['vthin2Dac'] = kwargs['vthin2Dac']
        self.dut['global_conf']['vffDac'] = kwargs['vffDac']
        self.dut['global_conf']['PrmpVbnFolDac'] = kwargs['PrmpVbnFolDac']
        self.dut['global_conf']['vbnLccDac'] = kwargs['vbnLccDac']
        self.dut['global_conf']['compVbnDac'] = kwargs['compVbnDac']
        self.dut['global_conf']['preCompVbnDac'] = kwargs['preCompVbnDac']

        self.dut.write_global()

        self.dut.write_global()

        # write InjEnLd & PixConfLd to '1
        self.dut['pixel_conf'].setall(True)
        self.dut.write_pixel_col()
        self.dut['global_conf']['SignLd'] = 1
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut['global_conf']['TDacLd'] = 0b1111
        self.dut['global_conf']['PixConfLd'] = 0b11
        self.dut.write_global()

        # write SignLd & TDacLd to '0
        self.dut['pixel_conf'].setall(False)
        self.dut.write_pixel_col()
        self.dut['global_conf']['SignLd'] = 1
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut['global_conf']['TDacLd'] = 0b1000
        self.dut['global_conf']['PixConfLd'] = 0b00
        self.dut.write_global()
