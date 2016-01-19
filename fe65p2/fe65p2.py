#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import yaml
from basil.dut import Dut
import logging
logging.getLogger().setLevel(logging.DEBUG)
import os
import numpy as np
import time

from numba import jit, njit

@njit
def _interpret_raw_data(data, pix_data):
    irec = 0
    prev_bcid = 0
    bcid = 0
    lv1id = 0
    for inx in range(data.shape[0]):
        if (data[inx] & 0x800000):
            bcid = data[inx] & 0x7fffff
            if(prev_bcid + 1 != bcid):
                lv1id = 0
            else:
                lv1id += 1
            prev_bcid = bcid
        else:
            col = (data[inx] & 0b111100000000000000000) >> 17
            row = (data[inx] & 0b11111100000000000) >>11
            rowp = (data[inx] & 0b10000000000) >> 10
            tot1 = (data[inx] & 0b11110000) >> 4
            tot0 = (data[inx] & 0b1111)
            
            # !!! THIS MAPPING MAY BE WRONG !!! 
            if(tot0 != 15):
                pix_data[irec].bcid = bcid
                pix_data[irec].lv1id = lv1id
                pix_data[irec].row = (row % 32) * 2 + rowp
                pix_data[irec].col = col * 4 + (row / 32) * 2 
                pix_data[irec].tot = tot0
                irec += 1

            if(tot1 != 15):
                pix_data[irec].bcid = bcid
                pix_data[irec].lv1id = lv1id
                pix_data[irec].row = (row % 32) * 2 + rowp
                pix_data[irec].col = col * 4 + 1 + (row / 32) * 2
                pix_data[irec].tot = tot1
                irec += 1
            
    return pix_data[:irec]
    
class fe65p2(Dut):

    def __init__(self,conf=None):
        
        if conf==None:
            conf = os.path.dirname(os.path.abspath(__file__)) + os.sep + "fe65p2.yaml"
        
        logging.info("Loading configuration file from %s" % conf)
        
        conf = self._preprocess_conf(conf)
        super(fe65p2, self).__init__(conf)
    
    def init(self):
        super(fe65p2, self).init()

    def _preprocess_conf(self, conf):
        return conf
        
    def write_global(self):
    
        #size of global register
        self['global_conf'].set_size(145)
        
        #write + start
        self['global_conf'].write()
        
        #wait for finish
        while not self['global_conf'].is_ready:
            pass
    
    def write_pixel(self, ld = False):
    
        #pixels in multi_column
        self['pixel_conf'].set_size(16*4*64)
    
        #enable writing pixels
        self['control']['GATE_EN_PIX_SR'] = 1
        self['control'].write()
        
        self['pixel_conf'].write()
        
        while not self['pixel_conf'].is_ready:
            pass
        
        self['control']['GATE_EN_PIX_SR'] = 0
        self['control'].write()
        
        if(ld):
            self['control']['LD'] = 1
            self['control'].write()
            self['control']['LD'] = 0
            self['control'].write()
        
    def write_pixel_col(self, ld = False):
    
        #pixels in multi_column
        self['pixel_conf'].set_size(4*64)
    
        #enable writing pixels
        self['control']['GATE_EN_PIX_SR'] = 1
        self['control'].write()
        
        self['pixel_conf'].write(4*64/8)
        
        while not self['pixel_conf'].is_ready:
            pass
        
        self['control']['GATE_EN_PIX_SR'] = 0
        self['control'].write()
        
        if(ld):
            self['control']['LD'] = 1
            self['control'].write()
            self['control']['LD'] = 0
            self['control'].write()
        
    def interpret_raw_data(self, data):
        pix_data = np.recarray((data.shape[0] * 2,), dtype={'names':['bcid','col','row','tot', 'lv1id'], 'formats':['uint32','uint8','uint8','uint8','uint8']})
        return _interpret_raw_data(data, pix_data)
       
    def power_up(self):
    
        self['VDDA'].set_current_limit(200, unit='mA')
        self['VDDA'].set_voltage(1.2, unit='V')
        self['VDDA'].set_enable(True)
        
        self['VDDD'].set_current_limit(200, unit='mA')
        self['VDDD'].set_voltage(1.2, unit='V')
        self['VDDD'].set_enable(True)
        
        self['VAUX'].set_current_limit(100, unit='mA')
        self['VAUX'].set_voltage(1.2, unit='V')
        self['VAUX'].set_enable(True)

    def power_status(self):
        staus = {}
       
        staus['VAUX'] = {'I':  self['VAUX'].get_current(unit='mA'), 'V': self['VAUX'].get_voltage(unit='V') }
        staus['VDDA'] = {'I':  self['VDDA'].get_current(unit='mA'), 'V': self['VDDA'].get_voltage(unit='V') }
        staus['VDDD'] = {'I':  self['VDDD'].get_current(unit='mA'), 'V': self['VDDD'].get_voltage(unit='V') }
         
        return staus
        
if __name__=="__main__":
    chip = fe65p2()
    chip.init()
    chip.power_up()
    