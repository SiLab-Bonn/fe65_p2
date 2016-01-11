#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import unittest
import os
from basil.utils.sim.utils import cocotb_compile_and_run, cocotb_compile_clean
import sys
import yaml
import mock
import numpy as np
import time

from fe65p2.fe65p2 import fe65p2
from fe65p2.scans.scan_digital import DigitalScan

def _preprocess_conf(self, conf):
    
    with open(conf, 'r') as f:
        cnfg = yaml.load(f)
    
    cnfg['transfer_layer'][0]['type'] = 'SiSim'
    cnfg['hw_drivers'][0]['no_calibration'] = True

    return cnfg
    
class TestScanDigital(unittest.TestCase):
    def setUp(self):
        
        proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #../
 
        verilog_src_fe65p2 = 'fe65p2.enc.sv'
        if os.path.isfile(proj_dir + '/tests/fe65p2.sv') :
            verilog_src_fe65p2 = 'fe65p2.sv'
        
        cocotb_compile_and_run(
            sim_files = [proj_dir + '/tests/fe65p2_tb.v', proj_dir + '/tests/' + verilog_src_fe65p2], 
            extra_defines = ['TEST_DC=1'],
            sim_bus = 'basil.utils.sim.SiLibUsbBusDriver',
            include_dirs = (proj_dir, proj_dir + "/firmware/src"),
            extra = 'export SIMULATION_MODULES='+yaml.dump({'HitDefaultDriver' : {} })
        )
        

    @mock.patch('fe65p2.fe65p2.fe65p2._preprocess_conf', autospec=True, side_effect=lambda *args, **kwargs: _preprocess_conf(*args, **kwargs)) #change interface to SiSim
    def test_scan(self, mock_preprocess):
        
        mask_steps = 4
        repeat_command = 2 
        
        self.scan = DigitalScan()
        self.scan.start(mask_steps = mask_steps, repeat_command = repeat_command, columns = [True] + [False] * 15)
        self.scan.analyze()
        
        data = np.concatenate([item[0] for item in scan.fifo_readout.data])
        exp_count = mask_steps * repeat_command * 8 + 2 * 4 * 64 * repeat_command
        
        self.assertEqual(len(data),  exp_count) 
        #TODO: more checks
        

    def tearDown(self):
        self.scan.dut.close()
        time.sleep(10)
        cocotb_compile_clean()

if __name__ == '__main__':
    unittest.main()
