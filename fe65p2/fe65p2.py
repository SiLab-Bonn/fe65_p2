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

class fe65p2(Dut):

    def __init__(self,conf=None):
        
        if conf==None:
            conf = os.path.dirname(os.path.abspath(__file__)) + os.sep + "fe65p2.yaml"
        
        logging.info("Loading configuration file from %s" % conf)
        super(fe65p2, self).__init__(conf)
    
    def init(self):
        super(fe65p2, self).init()

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
        '''TODO: speed this up. With Numba?'''
        
        trig_data = np.array([], dtype={'names':['bcid','col','row','rowp','tot1','tot0'], 'formats':['uint32','uint8','uint8','uint8','uint8','uint8']})
        
        bcid = -1
        for inx, i in enumerate(data):
            if (i & 0x800000):
                bcid = i & 0x7fffff
            else:
                col = (i & 0b111100000000000000000) >> 17
                row = (i & 0b11111100000000000) >>11
                rowp = (i & 0b10000000000) >> 10
                tot1 = (i & 0b11110000) >> 4
                tot0 = (i & 0b1111)
                datat = np.array([(bcid, col, row, rowp, tot1, tot0)], dtype=trig_data.dtype)
                trig_data = np.append(trig_data, datat)
                
        return trig_data
   
    def interpret_pix_data(self, data):
        
        pix_data = np.array([], dtype={'names':['bcid','col','row','tot'], 'formats':['uint32','uint8','uint8','uint8']})
        
        for word in data:
            bcid = word['bcid']
            row = word['row']*2 + word['rowp']
            
            if(word['tot0'] != 15):
                col = word['col']*2
                
                datat = np.array([(bcid, col, row, word['tot0'])], dtype=pix_data.dtype)
                pix_data = np.append(pix_data, datat)
            
            if(word['tot1'] != 15):
                col = word['col']*2+1
                
                datat = np.array([(bcid, col, row, word['tot1'])], dtype=pix_data.dtype)
                pix_data = np.append(pix_data, datat)
                
        return pix_data
        
    
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
    
    chip['control'].reset()
    chip.power_up()
    print chip.power_status(),  'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size(), 'DECODER_ERROR_COUNTER=', chip['rx'].DECODER_ERROR_COUNTER
        
        
    time.sleep(0.1)
    #global reg
    chip['global_conf']['PrmpVbpDac'] = 36
    chip['global_conf']['vthin1Dac'] = 255
    chip['global_conf']['vthin2Dac'] = 0
    chip['global_conf']['VctrCF0Dac'] = 42
    chip['global_conf']['VctrCF1Dac'] = 0
    chip['global_conf']['PrmpVbnFolDac'] = 51
    chip['global_conf']['vbnLccDac'] = 1
    chip['global_conf']['compVbnDac'] = 25
    chip['global_conf']['preCompVbnDac'] = 50
    
    chip['global_conf']['Latency'] = 400
    #chip['global_conf']['ColEn'][0] = 1
    chip['global_conf']['ColEn'].setall(True)
    chip['global_conf']['ColSrEn'].setall(True) #enable programming of all columns
    chip['global_conf']['ColSrOut'] = 15
    
    chip['global_conf']['OneSr'] = 1
    chip.write_global()

    #write InjEnLd & PixConfLd to '1
    chip['pixel_conf'].setall(True)
    chip.write_pixel()
    chip['global_conf']['SignLd'] = 1
    chip['global_conf']['InjEnLd'] = 1
    chip['global_conf']['TDacLd'] = 0b1111
    chip['global_conf']['PixConfLd'] = 0b11
    chip.write_global()
    
    #write SignLd & TDacLd to '0
    chip['pixel_conf'].setall(False)
    chip.write_pixel()
    chip['global_conf']['SignLd'] = 0
    chip['global_conf']['InjEnLd'] = 0
    chip['global_conf']['TDacLd'] = 0b0000
    chip['global_conf']['PixConfLd'] = 0b01
    chip.write_global()
   
    #off
    chip['global_conf']['SignLd'] = 0
    chip['global_conf']['InjEnLd'] = 0
    chip['global_conf']['TDacLd'] = 0
    chip['global_conf']['PixConfLd'] = 0
    chip.write_global()
    
    print 'LD3', chip.power_status(), 'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size()
    
    chip['control']['RESET'] = 0b01
    chip['control']['DISABLE_LD'] = 1 # so that standard LD does not make hit
    chip['control'].write()
        
    #make a tetshit & trigger

    #enable on pixle for test hit
    chip['pixel_conf'].setall(False)
    #chip['pixel_conf'].setall(True)
    #enable first and last 
    chip['pixel_conf'][0] = 1 
    #chip['pixel_conf'][1] = 1
    #chip['pixel_conf'][2] = 1
    #chip['pixel_conf'][3] = 1
    #chip['pixel_conf'][63] = 1
    #chip['pixel_conf'][62] = 1
    #chip['pixel_conf'][61] = 1
    #chip['pixel_conf'][31] = 1
    #chip['pixel_conf'][2*64-1] = 1
    #chip['pixel_conf'][3*64-1] = 1
    #chip['pixel_conf'][4*64-1] = 1
    #chip['pixel_conf'][5*64-1] = 1
    #chip['pixel_conf'][6*64-1] = 1
    #chip['pixel_conf'][7*64-1] = 1
    #chip['pixel_conf'][8*64-1] = 1
    #chip['pixel_conf'][9*64-1] = 1
    
    #chip['pixel_conf'][64*4-1] = 1
    chip['pixel_conf'][64*64-1] = 1
    #chip['pixel_conf'][64*61] = 1        
    chip.write_pixel()

    print 'WR_PIXEL', chip.power_status(), 'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size()
    
    #logic reset and enable clock
    #chip['control']['RESET'] = 0b01
    #chip['control'].write()
 
    chip['control']['CLK_OUT_GATE'] = 1
    chip['control'].write()
    time.sleep(0.1)
    print 'CLK_OUT_GATE', chip.power_status(), 'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size()
    
    chip['control']['CLK_BX_GATE'] = 1
    chip['control'].write()
    time.sleep(0.1)
    print 'CLK_BX_GATE', chip.power_status(), 'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size()

    
    chip['control']['RESET'] = 0b11
    #chip['control']['DISABLE_LD'] = 1 # so that standard LD does not make hit
    chip['control'].write()
    
    print 'LOGIC RST', chip.power_status(), 'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size(), 'DECODER_ERROR_COUNTER=', chip['rx'].DECODER_ERROR_COUNTER
    
    chip['rx'].reset()
    chip['fifo'].reset()
    chip['rx'].reset()


    time.sleep(0.1)
        
    print 'RX RST', chip.power_status(), 'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size(), 'DECODER_ERROR_COUNTER=', chip['rx'].DECODER_ERROR_COUNTER
    
    for _ in range(2):
        print 'fifo_size1', chip['fifo'].get_fifo_size()
        
        #enable testhit pulse and trigger
        chip['testhit'].set_delay(500)
        chip['testhit'].set_width(3)
        chip['testhit'].set_repeat(1)
        chip['testhit'].set_en(True)

        chip['trigger'].set_delay(400-8)
        chip['trigger'].set_width(16)
        chip['trigger'].set_repeat(1)
        chip['trigger'].set_en(True)
        
        chip['global_conf']['TestHit'] = 1
        chip['global_conf']['InjEnLd'] = 0
        chip['global_conf']['SignLd'] = 0
        chip['global_conf']['TDacLd'] = 0
        chip['global_conf']['PixConfLd'] = 0
        chip['global_conf'].set_wait(200)        
        chip.write_global() #send some test hit
        
        print 'fifo_size2', chip['fifo'].get_fifo_size()
        time.sleep(0.1)
        print 'fifo_size3', chip['fifo'].get_fifo_size()
        
        print 'INJ', 'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size(), 'DECODER_ERROR_COUNTER=', chip['rx'].DECODER_ERROR_COUNTER

        while not chip['trigger'].is_done():
            print 'wait'
        
           
        print 'END', 'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size(), 'DECODER_ERROR_COUNTER=', chip['rx'].DECODER_ERROR_COUNTER

        data = chip['fifo'].get_data()
        #print(data[:100])
        for inx, i in enumerate(data[:100]):
            if (i & 0x800000):
                print(inx, hex(i), 'BcId=', i & 0x7fffff)
            else:
                print(inx, hex(i), 'col=', (i & 0b111100000000000000000) >> 17, 'row=', (i & 0b11111100000000000) >>11, 'rowp=', (i & 0b10000000000) >> 10, 'tot1=', (i & 0b11110000) >> 4, 'tot0=', (i & 0b1111))
    
    time.sleep(0.1)
    
    #chip['control']['RESET'] = 0b00
    #chip['control']['DISABLE_LD'] = 0
    #chip['control'] = 0
    #chip['control'].write()
    
    
    print 'FIN', 'rx=', chip['rx'].READY, 'fifo=', chip['fifo'].get_fifo_size(), 'DECODER_ERROR_COUNTER=', chip['rx'].DECODER_ERROR_COUNTER
    print chip.power_status()
