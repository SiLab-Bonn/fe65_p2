# no injection, only need to turn the chip on and then measure the IV curve
from fe65p2.scan_base import ScanBase
import fe65p2.plotting as plotting
import time

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import csv
import tables as tb
from itertools import izip
import matplotlib.pyplot as plt
from basil.dut import Dut
import os

local_configuration = {
    "mask_steps": 4,
    "repeat_command": 1,

    # DAC parameters
    "PrmpVbpDac": 36,
    "vthin1Dac": 80,
    "vthin2Dac": 0,
    "vffDac": 24,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 110,

    # scan from val1(V) to val2(V) with steps of val3(V)
    # will start to run into the breakdown at -345
    "scan_range": [300, 400, 5],
    "current_limit": 0.000105  # set to 0.000105 for machine to work better...sometimes
}


class IVCurveMeasurement(ScanBase):
    scan_id = "iv_curve_measurement"

    def scan(self, repeat_command=10, **kwargs):
        '''Scan loop
        parameters:
            mask (int) :                number of mask steps
            repeat_command (int) :      number of injections/measurements
            to sync the pulse generator and the boards need to have A: 'TX1 -> to Ext Input' and B: 'RXO -> to Sync/Trigger out'
        '''

        # load mask???
        logging.info('starting IV Curve measurement')

        try:
            sourcemeter_dut = Dut(ScanBase.get_basil_dir(self) +
                                  '/examples/lab_devices/keithley2410_pyserial.yaml')
            sourcemeter_dut.init()
            # logging.info('Connected to ' +
            #              str(sourcemeter_dut['Sourcemeter'].get_info()))
        except RuntimeError:
            logging.info('No source meter, exiting')

        def cur_below_0(curr):
            if curr >= 0:
                time.sleep(0.2)
                curr_full = sourcemeter_dut['Sourcemeter'].get_current()
                curr_spt = curr_full.split(",")
                # print "curr loop ", curr_spt[1]
                return cur_below_0(float(curr_spt[1]))
            else:
                return curr
        sourcemeter_dut['Sourcemeter'].set_voltage(-250)
        time.sleep(1.0)
        self.dut.set_for_configuration()
        self.set_local_config()
        self.dut.start_up()
        repeat_command = kwargs.get('repeat_command', 1)

        scan_range = kwargs.get('scan_range', [100, 300, 1])
        print scan_range
        scan_range = np.arange(scan_range[0], scan_range[1], scan_range[2])
        repeat_list = np.arange(0, repeat_command, 1)
        current_limit = kwargs.get('current_limit', 0.000105)
        sourcemeter_dut['Sourcemeter'].set_current_limit(current_limit)
        app_vol = []
        meas_curr = []
        scn_num = []
        for i in repeat_list:
            for test_vol in scan_range:
                print "test voltage: ", -1 * test_vol
                # must change to negative voltage!
                sourcemeter_dut['Sourcemeter'].set_voltage(-1 * test_vol)
                time.sleep(0.5)
                ful_str_cur = sourcemeter_dut['Sourcemeter'].get_current()
                cur_split = ful_str_cur.split(",")

                curr_fin = cur_below_0(float(cur_split[1]))
                print "curr_fin= ", curr_fin
                full_str_vol = sourcemeter_dut['Sourcemeter'].get_voltage()
                vol_meas = full_str_vol.split(",")
                app_vol = np.append(app_vol, vol_meas[0])
                meas_curr = np.append(meas_curr, curr_fin)
                scn_num = np.append(scn_num, i)
                # print app_vol
                # print meas_curr
                if curr_fin <= -0.000010:
                    print "current limit hit"
                    break
        # data = np.vstack((app_vol, meas_curr, scn_num))
        csv_file_name = self.output_filename + '.csv'
        print csv_file_name
        # with tb.open_file(h5_file_name, 'r+') as out_file_h5:
        #     iv_curves = out_file_h5.create_carray(where="/",
        #                                           name='iv_data', title='iv curve data', obj=data)
        with open(csv_file_name, 'wb') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(izip(meas_curr, app_vol))

        plt.scatter(app_vol, meas_curr)
        plt.ylim(min(meas_curr).astype(np.float) - 1e-8,
                 max(meas_curr).astype(np.float) + 1e-8)
        plt.show()


if __name__ == "__main__":
    scan = IVCurveMeasurement()
    scan.start(**local_configuration)
    # scan.analyze()
