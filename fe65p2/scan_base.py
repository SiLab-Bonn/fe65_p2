
import logging
from fe65p2 import fe65p2
from fifo_readout import FifoReadout
from contextlib import contextmanager
import time

class ScanBase(object):
    '''Basic run meta class.

    Base class for scan- / tune- / analyze-class.
    '''

    def __init__(self, dut_conf=None):
        logging.info('Initializing %s', self.__class__.__name__)
        
        self.dut = fe65p2(dut_conf)
        self.dut.init()
        
        
        
    def start(self, **kwargs):
        
        self.dut['control'].reset()
        self.dut.power_up()
        time.sleep(0.1)
        
        
        #default config 
        #TODO: load from file
        self.dut['global_conf']['PrmpVbpDac'] = 36
        self.dut['global_conf']['vthin1Dac'] = 255
        self.dut['global_conf']['vthin2Dac'] = 0
        self.dut['global_conf']['VctrCF0Dac'] = 42
        self.dut['global_conf']['VctrCF1Dac'] = 0
        self.dut['global_conf']['PrmpVbnFolDac'] = 51
        self.dut['global_conf']['vbnLccDac'] = 1
        self.dut['global_conf']['compVbnDac'] = 25
        self.dut['global_conf']['preCompVbnDac'] = 50
        
        self.dut['global_conf']['Latency'] = 400
        #chip['global_conf']['ColEn'][0] = 1
        self.dut['global_conf']['ColEn'].setall(True)
        self.dut['global_conf']['ColSrEn'].setall(True) #enable programming of all columns
        self.dut['global_conf']['ColSrOut'] = 15
        
        self.dut['global_conf']['OneSr'] = 1
        self.dut.write_global() 
        
        logging.info('Power Status: %s', str(self.dut.power_status()))
        
        self.scan(**kwargs)
        
    def stop(self):
        pass
        
    def scan_loop(self):
        pass
        
        
    def analyze(self):
        raise NotImplementedError('ScanBase.analyze() not implemented')

    def scan(self, **kwargs):
        raise NotImplementedError('ScanBase.scan() not implemented')

    @contextmanager
    def readout(self, *args, **kwargs):
        self.fifo_readout = FifoReadout(self.dut)
        
        #self.fifo_readout.readout_interval = 10
        
        timeout = kwargs.pop('timeout', 10.0)
        self.fifo_readout.print_readout_status()
        self.start_readout(*args, **kwargs)
        yield
        self.fifo_readout.print_readout_status()
        self.fifo_readout.stop(timeout=timeout)

    def start_readout(self, *args, **kwargs):
        # Pop parameters for fifo_readout.start
        callback = kwargs.pop('callback', self.handle_data)
        clear_buffer = kwargs.pop('clear_buffer', False)
        fill_buffer = kwargs.pop('fill_buffer', False)
        reset_sram_fifo = kwargs.pop('reset_sram_fifo', False)
        errback = kwargs.pop('errback', None) #self.handle_err)
        no_data_timeout = kwargs.pop('no_data_timeout', None)
        if args or kwargs:
            self.set_scan_parameters(*args, **kwargs)
        self.fifo_readout.start(reset_sram_fifo=reset_sram_fifo, fill_buffer=fill_buffer, clear_buffer=clear_buffer, callback=callback, errback=errback, no_data_timeout=no_data_timeout)

    def handle_data(self, data):
        '''Handling of the data.
        '''
        pass #TODO: store data to file

    