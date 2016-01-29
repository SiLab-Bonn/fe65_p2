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
import time

from fe65p2.fe65p2 import fe65p2

def _preprocess_conf(self, conf):
    
    with open(conf, 'r') as f:
        cnfg = yaml.load(f)
    
    cnfg['transfer_layer'][0]['type'] = 'SiSim'
    cnfg['hw_drivers'][0]['no_calibration'] = True

    return cnfg
    
class TestSimSr(unittest.TestCase):

    def setUp(self):
        
        proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #../

        verilog_src_fe65_p2 = 'fe65p2.enc.sv'
        if os.path.isfile(proj_dir + '/tests/fe65p2.sv') :
            verilog_src_fe65_p2 = 'fe65p2.sv'
        
        cocotb_compile_and_run(
            sim_files = [proj_dir + '/tests/fe65p2_tb.v', proj_dir + '/tests/' + verilog_src_fe65_p2], 
            extra_defines = ['TEST_DC=1'],
            sim_bus = 'basil.utils.sim.SiLibUsbBusDriver',
            include_dirs = (proj_dir, proj_dir + "/firmware/src"),
            extra = 'export SIMULATION_MODULES='+yaml.dump({'HitDefaultDriver' : {} })
        )
        
    @mock.patch('fe65p2.fe65p2.fe65p2._preprocess_conf', autospec=True, side_effect=lambda *args, **kwargs: _preprocess_conf(*args, **kwargs)) #change interface to SiSim
    def test_sr(self, mock_preprocess):
    
        self.dut = fe65p2()
        self.dut.init()

        self.dut['control']['RESET'] = 1
        self.dut['control'].write()
        self.dut['control']['RESET'] = 0
        self.dut['control'].write()
        
        #global reg
        self.dut['global_conf']['PrmpVbpDac'] = 36
        self.dut['global_conf']['vthin1Dac'] = 255
        self.dut['global_conf']['vthin2Dac'] = 0
        self.dut['global_conf']['PrmpVbnFolDac'] = 0
        self.dut['global_conf']['vbnLccDac'] = 51
        self.dut['global_conf']['compVbnDac'] = 25
        self.dut['global_conf']['preCompVbnDac'] = 50
        self.dut['global_conf']['ColSrEn'].setall(True) #enable programming of all columns
        self.dut.write_global()
        self.dut.write_global()
        
        send = self.dut['global_conf'].tobytes()
        rec =  self.dut['global_conf'].get_data(size=19)
        
        self.assertEqual(send,rec)

        #pixel reg
        self.dut['pixel_conf'][0] = 1
        self.dut.write_pixel()
        
        self.dut['control']['RESET'] = 0b11
        self.dut['control'].write()
        
        #TODO: check for pixels
        

        
    def tearDown(self):
        self.dut.close()
        time.sleep(10)
        cocotb_compile_clean()

if __name__ == '__main__':
    unittest.main()
