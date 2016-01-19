
from fe65p2.scan_base import ScanBase
import fe65p2.plotting as  plotting
import time

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, show, vplot, hplot

local_configuration = {
    "mask_steps": 4*64,
    "repeat_command": 100
}

class AnalogScan(ScanBase):
    scan_id = "analog_scan"

    def scan(self, mask_steps=4, repeat_command=100, columns = [True] * 16, **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''
        
        #columns = [True] + [False] * 15

        #self.dut['INJ_LO'].set_voltage(0, unit='V')
        #self.dut['INJ_HI'].set_voltage(1.2, unit='V')
        
        
        self.dut['INJ_LO'].set_voltage(1.0, unit='V')
        self.dut['INJ_HI'].set_voltage(0.3, unit='V')
        
        self.dut['global_conf']['vthin1Dac'] = 150
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
        
        self.dut['global_conf']['OneSr'] = 0 #all multi columns in parallel
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
        self.dut['inj'].set_delay(columns.count(True) * 5000) #this should based on mask and enabled columns
        self.dut['inj'].set_width(100)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(400)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)
        
        lmask = [1] + ( [0] * (mask_steps-1) )
        lmask = lmask * ( (64 * 64) / mask_steps  + 1 )
        lmask = lmask[:64*64]
        bv_mask = bitarray.bitarray(lmask)
        

        with self.readout():
            
            for i in range(mask_steps):

                #set all InjEn to 0
                self.dut['pixel_conf'].setall(False)
                self.dut.write_pixel_col()
                self.dut['global_conf']['InjEnLd'] = 1
                self.dut['global_conf']['PixConfLd'] = 0b00                
                self.dut.write_global()
                
                #self.dut['global_conf']['InjEnLd'] = 0
                #self.dut['global_conf']['PixConfLd'] = 0b00               
                #self.dut.write_global()
                
                self.dut['pixel_conf'][:]  = bv_mask
                self.dut.write_pixel_col()
                self.dut['global_conf']['PixConfLd'] = 0b11   
                self.dut['global_conf']['InjEnLd'] = 1
                self.dut.write_global()
                
                bv_mask[1:] = bv_mask[0:-1] 
                bv_mask[0] = 0
                
                #self.dut.write_pixel_col()
                
                self.dut['inj'].start()
                
                while not self.dut['inj'].is_done():
                    pass
                    
                while not self.dut['trigger'].is_done():
                    pass
                
                #print('.'),
                #self.fifo_readout.print_readout_status()
                
            #just some time for last read
            self.dut['trigger'].set_en(False)
            self.dut['inj'].start()
            #print('.')
            
    def analyze(self):
        with tb.open_file(self.output_filename +'.h5', 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
    
            hit_data = self.dut.interpret_raw_data(raw_data)
            self.h5_file.createTable(self.h5_file.root, 'hit_data', hit_data, filters=self.filter_tables)
            
            occ_plot, H = plotting.plot_occupancy(hit_data)
            tot_plot,_ = plotting.plot_tot_dist(hit_data)
            lv1id_plot, _ = plotting.plot_lv1id_dist(hit_data)
                             
            output_file(self.output_filename + '.html', title=self.run_name)
            show(vplot(occ_plot, tot_plot, lv1id_plot))
        
if __name__ == "__main__":

    scan = AnalogScan()
    scan.start(**local_configuration)
    scan.analyze()
