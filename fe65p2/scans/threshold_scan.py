
from fe65p2.scan_base import ScanBase
import time

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb

local_configuration = {
    "mask_steps": 4*64,
    "repeat_command": 100
}

class ThresholdScan(ScanBase):
    scan_id = "threshold_scan"

    def scan(self, mask_steps=4, repeat_command=100, columns = [True] * 16, **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''
        
        #columns =  [False] * 2 + [True] + [False] * 13

        #self.dut['INJ_LO'].set_voltage(0, unit='V')
        #self.dut['INJ_HI'].set_voltage(1.2, unit='V')
        
        
        self.dut['INJ_LO'].set_voltage(1.0, unit='V')
        self.dut['INJ_HI'].set_voltage(0.8, unit='V')
        
        self.dut['global_conf']['vthin1Dac'] = 160
        self.dut['global_conf']['vthin2Dac'] = 0
        self.dut.write_global() 

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
        self.dut['global_conf']['SignLd'] = 1
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut['global_conf']['TDacLd'] = 0b1000
        self.dut['global_conf']['PixConfLd'] = 0b00
        self.dut.write_global()
       
        #test hit
        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        
        self.dut['global_conf']['OneSr'] = 1 #!
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)        
        self.dut.write_global()
    
        self.dut['control']['RESET'] = 0b01
        self.dut['control']['DISABLE_LD'] = 0
        self.dut['control'].write()
        
        self.dut['control']['CLK_OUT_GATE'] = 1
        self.dut['control']['CLK_BX_GATE'] = 1
        self.dut['control'].write()
        time.sleep(0.1)
        
        self.dut['control']['RESET'] = 0b11
        self.dut['control'].write()
                
        #enable inj pulse and trigger
        self.dut['inj'].set_delay(columns.count(True) * 5000*2) #this should based on mask and enabled columns
        self.dut['inj'].set_width(100)
        self.dut['inj'].set_repeat(100)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(400-4)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)
        
        lmask = [1] + ( [0] * (mask_steps-1) )
        lmask = lmask * ( (64 * 64) / mask_steps  + 1 )
        lmask = lmask[:64*64]
        bv_mask = bitarray.bitarray(lmask)
        
        
        #set all InjEn to 0
        self.dut['pixel_conf'].setall(False)
        self.dut.write_pixel_col()
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut['global_conf']['PixConfLd'] = 0b00               
        self.dut.write_global()
        
        #self.dut['global_conf']['InjEnLd'] = 0
        #self.dut['global_conf']['PixConfLd'] = 0b00               
        #self.dut.write_global()
        
        bv_mask[1:] = bv_mask[0:-1] 
        bv_mask[0] = 0
        
        for _ in range (10):
            bv_mask[1:] = bv_mask[0:-1] 
            bv_mask[0] = 0

        bv_mask.setall(False)
        bv_mask[64*8+5] = 1
        bv_mask[64*8+10] = 1
        bv_mask[64*8+15] = 1
        bv_mask[64*8+20] = 1
        bv_mask[64*8+25] = 1
        bv_mask[64*8+30] = 1
        bv_mask[64*8+35] = 1
        bv_mask[64*8+40] = 1
        bv_mask[64*8+45] = 1
        bv_mask[64*8+50] = 1
        
        self.dut['pixel_conf'][:]  = bv_mask
        self.dut.write_pixel()
        self.dut['global_conf']['PixConfLd'] = 0b11   
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut.write_global()
        
        self.dut['global_conf']['PixConfLd'] = 0b00   
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut.write_global()  
        
        #self.fifo_readout.reset_rx()
        #time.sleep(0.1)
        #self.fifo_readout.print_readout_status()
        
        np.set_printoptions(threshold=np.nan)
        
        
        #with self.readout(fill_buffer=True):
        #scan_range = np.arange(0.545, 0.547, 0.0001)
        scan_range = np.arange(0.5, 0.6, 0.01)
        
        for k in scan_range: #range(10):
            with self.readout(scan_param_id = k):
            #for k in range(100):

                #self.scan_param_id = k
                
                vol = float(k) #0.2+k*(1/10.0)
                self.dut['INJ_HI'].set_voltage( vol, unit='V')
                logging.info('Scan Parameter: %f', vol)
                time.sleep(1)
        
                #self.dut.write_pixel_col()
                
                self.dut['inj'].start()
                #self.dut['trigger'].start()
                
                while not self.dut['inj'].is_done():
                    pass
                    
                while not self.dut['trigger'].is_done():
                    pass
                
                #print('.'),
                
                #just some time for last read
                #time.sleep(1)
                
                #self.dut['trigger'].set_en(False)
                #self.dut['inj'].start()
                #print('.')
                #print self.fifo_readout.data  
            
                #print vol
            
            
            #print vol, np.concatenate([item[0] for item in self.fifo_readout.data ]).shape[0]
            #dqdata =  self.fifo_readout.data        
            #data = np.concatenate([item[0] for item in dqdata])
            #int_pix_data = self.dut.interpret_raw_data(data)
            #H, _, _ = np.histogram2d(int_pix_data['col'], int_pix_data['row'], bins = (range(65), range(65)))
            #print H
            #print(k, H[0][1], 'Mean ToT:', np.mean(int_pix_data['tot']), vol)
            #self.fifo_readout.data.clear()
        
        
    def analyze(self):
        with tb.open_file(self.output_filename +'.h5', 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]
            #print raw_data, meta_data
            param, index = np.unique(meta_data['scan_param_id'], return_index=True)
            index = index[1:]
            index = np.append(index, meta_data.shape[0])
            index = index - 1
            print index
            stops = meta_data['index_stop'][index]
            print stops
            split = np.split(raw_data, stops)
            #print split
            print len(split)
            for i in range(len(split[:-1])):
                #print param[i], stops[i], len(split[i]), split[i]
                int_pix_data = self.dut.interpret_raw_data(split[i])
                H, _, _ = np.histogram2d(int_pix_data['col'], int_pix_data['row'], bins = (range(65), range(65)))
                print param[i], H[8:10] #3
                
        return None

        #output_file = scan.scan_data_filename + "_interpreted.h5"
        
if __name__ == "__main__":

    scan = ThresholdScan()
    scan.start(**local_configuration)
    scan.analyze()
