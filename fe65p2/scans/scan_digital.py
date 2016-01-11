
from fe65p2.scan_base import ScanBase
import time

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray

local_configuration = {
    "mask_steps": 4,
    "repeat_command": 100
}

class DigitalScan(ScanBase):
    scan_id = "digital_scan"

    def scan(self, mask_steps=4, repeat_command=100, columns = [True] * 16, **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''
        #write InjEnLd & PixConfLd to '1
        self.dut['pixel_conf'].setall(True)
        self.dut.write_pixel_col()
        self.dut['global_conf']['SignLd'] = 1
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut['global_conf']['TDacLd'] = 0b1111
        self.dut['global_conf']['PixConfLd'] = 0b11
        self.dut.write_global()
        
        #write SignLd & TDacLd to '0
        self.dut['pixel_conf'].setall(False)
        self.dut.write_pixel_col()
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0b0000
        self.dut['global_conf']['PixConfLd'] = 0b01
        self.dut.write_global()
       
        #test hit
        self.dut['global_conf']['TestHit'] = 1
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        
        self.dut['global_conf']['OneSr'] = 0 #all multi columns in parallel
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)        
        self.dut.write_global()
    
        self.dut['control']['RESET'] = 0b01
        self.dut['control']['DISABLE_LD'] = 1
        self.dut['control'].write()
        
        self.dut['control']['CLK_OUT_GATE'] = 1
        self.dut['control']['CLK_BX_GATE'] = 1
        self.dut['control'].write()
        time.sleep(0.1)
        
        self.dut['control']['RESET'] = 0b11
        self.dut['control'].write()
                
        #enable testhit pulse and trigger
        self.dut['testhit'].set_delay(columns.count(True) * 5000) #this should based on mask and enabled columns
        self.dut['testhit'].set_width(3)
        self.dut['testhit'].set_repeat(repeat_command)
        self.dut['testhit'].set_en(False)

        self.dut['trigger'].set_delay(400-4)
        self.dut['trigger'].set_width(8)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)
        
        lmask = [1] + ( [0] * (mask_steps-1) )
        lmask = lmask * ( (64 * 64) / mask_steps  + 1 )
        lmask = lmask[:64*64]
        bv_mask = bitarray.bitarray(lmask)
        
        with self.readout(fill_buffer=True):
            
            for i in range(mask_steps):

                self.dut['pixel_conf'][:]  = bv_mask
                bv_mask[1:] = bv_mask[0:-1] 
                bv_mask[0] = 0
                
                self.dut.write_pixel_col()
                
                self.dut['testhit'].start()
                
                while not self.dut['testhit'].is_done():
                    pass
                    
                while not self.dut['trigger'].is_done():
                    pass
            
            #just some time for last read
            time.sleep(1)
            self.dut['trigger'].set_en(False)
            self.dut['testhit'].start()
    
    def analyze(self):
        dqdata =  self.fifo_readout.data        
        data = np.concatenate([item[0] for item in dqdata])
        
        #for inx, i in enumerate(data[:200]):
        #    if (i & 0x800000):
        #        print(inx, hex(i), 'BcId=', i & 0x7fffff)
        #    else:
        #        print(inx, hex(i), 'col=', (i & 0b111100000000000000000) >> 17, 'row=', (i & 0b11111100000000000) >>11, 'rowp=', (i & 0b10000000000) >> 10, 'tot1=', (i & 0b11110000) >> 4, 'tot0=', (i & 0b1111))
    
        int_pix_data = self.dut.interpret_raw_data(data)
        H, _, _ = np.histogram2d(int_pix_data['col'], int_pix_data['row'], bins = (range(64), range(64)))
       
        np.set_printoptions(threshold=np.nan)
        print H
        return H

        #output_file = scan.scan_data_filename + "_interpreted.h5"
        
if __name__ == "__main__":

    scan = DigitalScan()
    scan.start(**local_configuration)
    scan.analyze()
