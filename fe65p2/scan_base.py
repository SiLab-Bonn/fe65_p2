import logging
import basil
from fe65p2 import fe65p2
from fifo_readout import FifoReadout
from contextlib import contextmanager
import time
import os
import tables as tb
import yaml
import zmq


class MetaTable(tb.IsDescription):
    index_start = tb.UInt32Col(pos=0)
    index_stop = tb.UInt32Col(pos=1)
    data_length = tb.UInt32Col(pos=2)
    timestamp_start = tb.Float64Col(pos=3)
    timestamp_stop = tb.Float64Col(pos=4)
    scan_param_id = tb.UInt32Col(pos=5)
    error = tb.UInt32Col(pos=6)
    trigger = tb.Float64Col(pos=7)


# def send_meta_data(socket, conf, name):
#     '''Sends the config via ZeroMQ to a specified socket. Is called at the beginning of a run and when the config changes. Conf can be any config dictionary.
#     '''
#     meta_data = dict(
#         name=name,
#         conf=conf
#     )
#     try:
#         socket.send_json(meta_data, flags=zmq.NOBLOCK)
#     except zmq.Again:
#         pass


def send_data(socket, data, scan_parameters={}, name='ReadoutData'):
    '''Sends the data of every read out (raw data and meta data) via ZeroMQ to a specified socket
    '''
    if not scan_parameters:
        scan_parameters = {}
    data_meta_data = dict(
        name=name,
        dtype=str(data[0].dtype),
        shape=data[0].shape,
        timestamp_start=data[1],  # float
        timestamp_stop=data[2],  # float
        readout_error=data[3],  # int
        scan_parameters=scan_parameters  # dict
    )
    try:
        socket.send_json(data_meta_data, flags=zmq.SNDMORE | zmq.NOBLOCK)
        socket.send(data[0], flags=zmq.NOBLOCK)  # PyZMQ supports sending numpy arrays without copying any data
    except zmq.Again:
        pass


class ScanBase(object):
    '''Basic run meta class.

    Base class for scan- / tune- / analyse-class.
    '''

    def __init__(self, dut_conf=None):

        logging.info('Initializing %s', self.__class__.__name__)

        self.dut = fe65p2(dut_conf)

        self.working_dir = os.path.join(os.getcwd(), "output_data")
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        self.final_vth1 = -99
        self.run_name = time.strftime("%Y%m%d_%H%M%S_") + self.scan_id
        self.output_filename = os.path.join(self.working_dir, self.run_name)

        self.fh = logging.FileHandler(self.output_filename + '.log')
        self.fh.setLevel(logging.DEBUG)
        self.logger = logging.getLogger()
        self.logger.addHandler(self.fh)

        self.dut.init()

    def cap_fac(self):
        return 7.9891

    def get_basil_dir(self):
        return str(os.path.dirname(os.path.dirname(basil.__file__)))

    def start(self, name=None, **kwargs):

        self._first_read = False
        self.scan_param_id = 0
        if name:
            filename = self.output_filename + str(name) + '.h5'
        else:
            filename = self.output_filename + '.h5'
        filter_raw_data = tb.Filters(
            complib='blosc', complevel=5, fletcher32=False)
        self.filter_tables = tb.Filters(complib='zlib', complevel=5, fletcher32=False)
        self.h5_file = tb.open_file(filename, mode='w', title=self.scan_id)
        self.raw_data_earray = self.h5_file.create_earray(self.h5_file.root, name='raw_data', atom=tb.UIntAtom(),
                                                          shape=(0,), title='raw_data', filters=filter_raw_data)
        self.meta_data_table = self.h5_file.create_table(self.h5_file.root, name='meta_data', description=MetaTable,
                                                         title='meta_data', filters=self.filter_tables)

        self.meta_data_table.attrs.kwargs = yaml.dump(kwargs)
        self._kwargs = kwargs

        self.dut['control']['RESET'] = 0b00
        self.dut['control'].write()
        self.dut.power_up()
        time.sleep(0.1)

        self.fifo_readout = FifoReadout(self.dut)

        # default config
        # TODO: load from file
        self.dut['global_conf']['PrmpVbpDac'] = 80
        self.dut['global_conf']['vthin1Dac'] = 255
        self.dut['global_conf']['vthin2Dac'] = 0
        self.dut['global_conf']['vffDac'] = 42
        self.dut['global_conf']['PrmpVbnFolDac'] = 51
        self.dut['global_conf']['vbnLccDac'] = 1
        self.dut['global_conf']['compVbnDac'] = 25
        self.dut['global_conf']['preCompVbnDac'] = 50

        self.dut['global_conf']['Latency'] = 400
        # chip['global_conf']['ColEn'][0] = 1
        self.dut['global_conf']['ColEn'].setall(True)
        self.dut['global_conf']['ColSrEn'].setall(
            True)  # enable programming of all columns
        self.dut['global_conf']['ColSrOut'] = 15

        self.dut['global_conf']['OneSr'] = 0  # all multi columns in parallel
        self.dut.write_global()

        self.dut['control']['RESET'] = 0b10
        self.dut['control'].write()
        pw = self.dut.power_status()
        logging.info('Power Status: %s', str(pw))

        self.scan(**kwargs)

        self.fifo_readout.print_readout_status()

        self.meta_data_table.attrs.power_status = yaml.dump(pw)
        self.meta_data_table.attrs.dac_status = yaml.dump(self.dut.dac_status())
        self.meta_data_table.attrs.vth1 = yaml.dump(self.final_vth1)

        # temperature
        #temp = self.dut['ntc'].get_temperature('C')
        #self.meta_data_table.attrs.temp = yaml.dump(str(temp))

        self.h5_file.close()
        logging.info('Data Output Filename: %s', self.output_filename + '.h5')

        # temp and power log
        if (self.scan_id != "status_scan" and self.scan_id != "temp_scan"):
            logname = 'reg_' + str(self.scan_id) + '.dat'
            vth1_set = self.final_vth1

            legend = "Time \t Dig[mA] \t Ana[mA] \t Aux[mA] \t Dig[V] \t vth1 \n"
            if not os.path.exists("./" + logname):
                with open(logname, "a") as t_file:
                    t_file.write(legend)
            t_log = time.strftime("%d-%b-%H:%M:%S") + "\t" + str(pw['VDDD[mA]']) + "\t" + str(
                pw['VDDA[mA]']) + "\t" + str(pw['VAUX[mA]']) + "\t" + str(pw['VDDD[V]']) + "\t" + str(vth1_set) + "\n"
            with open(logname, "a") as t_file:
                t_file.write(t_log)

        self.logger.removeHandler(self.fh)
        # self.dut.power_down()

    def analyze(self):
        raise NotImplementedError('ScanBase.analyze() not implemented')

    def scan(self, **kwargs):
        raise NotImplementedError('ScanBase.scan() not implemented')

    @contextmanager
    def readout(self, *args, **kwargs):
        timeout = kwargs.pop('timeout', 10.0)

        # self.fifo_readout.readout_interval = 10
        if not self._first_read:
            self.fifo_readout.reset_rx()
            time.sleep(0.1)
            self.fifo_readout.print_readout_status()
            self._first_read = True

        self.dut['rx'].ENABLE_RX = 1

        self.start_readout(*args, **kwargs)
        yield
        self.fifo_readout.stop(timeout=timeout)

    def start_readout(self, scan_param_id=0, *args, **kwargs):
        # Pop parameters for fifo_readout.start
        callback = kwargs.pop('callback', self.handle_data)
        clear_buffer = kwargs.pop('clear_buffer', False)
        fill_buffer = kwargs.pop('fill_buffer', False)
        reset_sram_fifo = kwargs.pop('reset_sram_fifo', False)
        errback = kwargs.pop('errback', self.handle_err)
        no_data_timeout = kwargs.pop('no_data_timeout', None)
        self.scan_param_id = scan_param_id
        self.fifo_readout.start(reset_sram_fifo=reset_sram_fifo, fill_buffer=fill_buffer, clear_buffer=clear_buffer,
                                callback=callback, errback=errback, no_data_timeout=no_data_timeout)
#         self.start_tlu_triggers()

    def start_tlu_triggers(self):
        #         self.dut.set_for_configuration()
        self.dut['control']['EXT_TRIGGER_ENABLE'] = 1
        self.dut['control'].write()
        self.dut['TLU'].TRIGGER_ENABLE = 1

    def stop_tlu_triggers(self):
        self.dut['TLU'].TRIGGER_ENABLE = 0
        #         self.dut.set_for_configuration()
        self.dut['control']['EXT_TRIGGER_ENABLE'] = 0
        self.dut['control'].write()
        print self.dut['control']['EXT_TRIGGER_ENABLE']

    def set_local_config(self, vth1=None, **kwargs):

        self.dut['global_conf']['PrmpVbpDac'] = self._kwargs['PrmpVbpDac']
        if vth1:
            self.dut['global_conf']['vthin1Dac'] = int(vth1)
        else:
            self.dut['global_conf']['vthin1Dac'] = self._kwargs['vthin1Dac']
        self.dut['global_conf']['vthin2Dac'] = self._kwargs['vthin2Dac']
        self.dut['global_conf']['vffDac'] = self._kwargs['vffDac']
        self.dut['global_conf']['PrmpVbnFolDac'] = self._kwargs['PrmpVbnFolDac']
        self.dut['global_conf']['vbnLccDac'] = self._kwargs['vbnLccDac']
        self.dut['global_conf']['compVbnDac'] = self._kwargs['compVbnDac']
        self.dut['global_conf']['preCompVbnDac'] = self._kwargs['preCompVbnDac']
        self.dut.write_global()
#         time.sleep(0.7)

    def handle_data(self, data_tuple):
        '''Handling of the data.
        '''
        def get_bin(x, n): return format(x, 'b').zfill(n)
        # print data_tuple[0].shape[0] #, data_tuple

        total_words = self.raw_data_earray.nrows

        self.raw_data_earray.append(data_tuple[0])
        self.raw_data_earray.flush()

        len_raw_data = data_tuple[0].shape[0]
        self.meta_data_table.row['timestamp_start'] = data_tuple[1]
        self.meta_data_table.row['timestamp_stop'] = data_tuple[2]
        self.meta_data_table.row['error'] = data_tuple[3]
        self.meta_data_table.row['data_length'] = len_raw_data
        self.meta_data_table.row['index_start'] = total_words
        total_words += len_raw_data
        self.meta_data_table.row['index_stop'] = total_words
        self.meta_data_table.row['scan_param_id'] = self.scan_param_id
        counter = 0
        for i in data_tuple[0]:
            counter = counter + int(get_bin(int(data_tuple[0][0]), 32)[1])
        self.meta_data_table.row['trigger'] = counter / len(data_tuple[0])
        self.meta_data_table.row.append()
#
#         if self.socket:
#             send_data(self.socket, data_tuple, self.scan_parameters)

        self.meta_data_table.flush()
        # print len_raw_data

    def handle_err(self, exc):
        msg = '%s' % exc[1]
        if msg:
            logging.error('%s%s Aborting run...', msg, msg[-1])
        else:
            logging.error('Aborting run...')
