''' Changes the frequency and digital supply voltage and tests pixel response.
    It is called Shmoo Plot.
'''

import os
import time
import logging
import yaml
from contextlib import contextmanager
import tables as tb
from bokeh.charts import output_file, show, hplot, save
from bokeh.models.layouts import Column
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
from fe65p2.fe65p2 import fe65p2
from fe65p2.fifo_readout import FifoReadout
import fe65p2.plotting as plotting
import numpy as np
import bitarray
import sys
from collections import OrderedDict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


class MetaTable(tb.IsDescription):
    index_start = tb.UInt32Col(pos=0)
    index_stop = tb.UInt32Col(pos=1)
    data_length = tb.UInt32Col(pos=2)
    timestamp_start = tb.Float64Col(pos=3)
    timestamp_stop = tb.Float64Col(pos=4)
    scan_param_id = tb.UInt16Col(pos=5)
    error = tb.UInt32Col(pos=6)


local_configuration = {
    "mask_steps": 4,
    "repeat_command": 100,
    "scan_type": 'data',
    "columns": [True] * 16,
    # DAC parameters
    "PrmpVbpDac": 36,
    "vthin1Dac": 255,
    "vthin2Dac": 0,
    "vffDac": 42,  # not subject to change
    "PrmpVbnFolDac": 51,  # not subject to change
    "vbnLccDac": 1,  # not subject to change
    "compVbnDac": 25,  # not subject to change
    "preCompVbnDac": 50,
}

'''
	In each of the 4096 pixels a voltage pulse is injected 100 times.
	Different clock frequencies and supply voltages settings are tested.
	The pixel response is recorded in terms of occupancy, TOT and lv1id;
	in addition the total number of missed injections for each (V/MHz) setup
	and reported as a Shmoo plot.
'''


class DigitalScanFreq(object):
    scan_id = "digital_scan"

    def __init__(self, dut_conf=None, **kwargs):
        self.dut = fe65p2(dut_conf)
        self.dut.init()
        self.dut.power_up()
        self.plots = False

#        self.dut['control']['INV_BX_CLK'] = 0
#        self.dut['control'].write()

        time.sleep(0.1)
        self.dut['global_conf']['PrmpVbpDac'] = 36
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

        self.working_dir = os.path.join(os.getcwd(), "output_data")
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

    @contextmanager
    def readout(self, *args, **kwargs):
        timeout = kwargs.pop('timeout', 10.0)

        # self.fifo_readout.readout_interval = 10
        if not self._first_read:
            self.fifo_readout.reset_rx()
            time.sleep(0.1)
            self.fifo_readout.print_readout_status()
            self._first_read = True

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

    def handle_data(self, data_tuple):
        '''Handling of the data.
        '''
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
        self.meta_data_table.row.append()
        self.meta_data_table.flush()

    def handle_err(self, exc):
        msg = '%s' % exc[1]
        if msg:
            logging.error('%s%s Aborting run...', msg, msg[-1])
        else:
            logging.error('Aborting run...')

    def scan(self, mask_steps=4, repeat_command=100, columns=[True] * 16, **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''

        #scan_path = os.path.dirname(os.path.realpath(sys.argv[0]))
        #path = scan_path.replace('fe65p2/scans', 'firmware/bits/')
        path = "/home/topcoup/Applications/fe65_p2/firmware/bits/"
        self.scantype = kwargs['scan_type']

        if self.scantype == 'cmd':
            self.clock_name = 'CMD clock'
            path = path + 'goodCMD_bits/'
            #self.bitfiles = OrderedDict([(50, "fe65p2_mio_CMD50.bit")])
            self.bitfiles = OrderedDict(
                [(20, "fe65p2_mio_CMD20.bit"), (30, "fe65p2_mio_CMD30.bit"), (40, "fe65p2_mio_CMD40.bit"),
                 (50, "fe65p2_mio_CMD50.bit"), (60,
                                                "fe65p2_mio_CMD60.bit"), (70, "fe65p2_mio_CMD70.bit"),
                 (80, "fe65p2_mio_CMD80.bit"), (90, "fe65p2_mio_CMD90.bit"), (100, "fe65p2_mio_CMD100.bit")])  # , (130,"fe65p2_mio_CMD130.bit"), (140,"fe65p2_mio_CMD140.bit"), (150,"fe65p2_mio_CMD150.bit"),  (160,"fe65p2_mio_CMD160.bit")])

        if self.scantype == 'data':
            self.clock_name = 'DATA clock'
            path = path + 'goodDATA_bits/'
            self.bitfiles = OrderedDict(
                [(40, "fe65p2_mio_DATA40.bit"), (60, "fe65p2_mio_DATA60.bit"), (80, "fe65p2_mio_DATA80.bit"),
                 (100, "fe65p2_mio_DATA100.bit"), (120, "fe65p2_mio_DATA120.bit"), (160, "fe65p2_mio_DATA160.bit")])
        self.voltages = [1.25, 1.2, 1.1, 1.0, 0.95, 0.90]
        self.not_fired = []
        for freq in self.bitfiles.iterkeys():
            self.dut.power_down()
            logging.info("Loading " + self.bitfiles[freq])  # loading bitfile
            self.dut['intf']._sidev.DownloadXilinx(path + self.bitfiles[freq])
            self.dut.power_up()

            for volt in self.voltages:
                # to change the supply voltage
                self.dut['VDDA'].set_current_limit(200, unit='mA')
                self.dut['VDDA'].set_voltage(volt, unit='V')
                self.dut['VDDA'].set_enable(True)
                self.dut['VDDD'].set_voltage(volt, unit='V')
                self.dut['VDDD'].set_enable(True)
                self.dut['VAUX'].set_voltage(1.25, unit='V')
                self.dut['VAUX'].set_enable(True)

                self.dut['global_conf']['PrmpVbpDac'] = kwargs['PrmpVbpDac']
                self.dut['global_conf']['vthin1Dac'] = kwargs['vthin1Dac']
                self.dut['global_conf']['vthin2Dac'] = kwargs['vthin2Dac']
                self.dut['global_conf']['vffDac'] = kwargs['vffDac']
                self.dut['global_conf']['PrmpVbnFolDac'] = kwargs['PrmpVbnFolDac']
                self.dut['global_conf']['vbnLccDac'] = kwargs['vbnLccDac']
                self.dut['global_conf']['compVbnDac'] = kwargs['compVbnDac']
                # kwargs.get('preCompVbnDac',50)
                self.dut['global_conf']['preCompVbnDac'] = 50
                self.dut.write_global()

                logging.info(self.dut.power_status())  # prints power supply
                self.run_name = time.strftime(
                    "%Y%m%d_%H%M%S_") + "_" + str(freq) + "MHz_" + str(volt) + "V"
                self.output_filename = os.path.join(
                    self.working_dir, str(self.scantype) + '_' + str(self.run_name))
                self._first_read = False
                self.scan_param_id = 0

                # .h5 output management
                filename = self.output_filename + '.h5'
                filter_raw_data = tb.Filters(
                    complib='blosc', complevel=5, fletcher32=False)
                self.filter_tables = tb.Filters(
                    complib='zlib', complevel=5, fletcher32=False)
                self.h5_file = tb.open_file(
                    filename, mode='w', title=self.scan_id)
                self.raw_data_earray = self.h5_file.create_earray(self.h5_file.root, name='raw_data', atom=tb.UIntAtom(),
                                                                  shape=(0,), title='raw_data', filters=filter_raw_data)
                self.meta_data_table = self.h5_file.create_table(self.h5_file.root, name='meta_data', description=MetaTable,
                                                                 title='meta_data', filters=self.filter_tables)

                self.meta_data_table.attrs.kwargs = yaml.dump(kwargs)
                self.dut['control']['RESET'] = 0b00
                self.dut['control'].write()
                time.sleep(0.1)

                self.fifo_readout = FifoReadout(self.dut)

                # write InjEnLd & PixConfLd to '1
                self.dut['pixel_conf'].setall(True)
                self.dut.write_pixel_col()
                self.dut['global_conf']['SignLd'] = 1
                self.dut['global_conf']['InjEnLd'] = 1
                self.dut['global_conf']['TDacLd'] = 0b1111
                self.dut['global_conf']['PixConfLd'] = 0b11
                self.dut.write_global()
                # write SignLd & TDacLd to '0
                self.dut['pixel_conf'].setall(False)
                self.dut.write_pixel_col()
                self.dut['global_conf']['SignLd'] = 0
                self.dut['global_conf']['InjEnLd'] = 0
                self.dut['global_conf']['TDacLd'] = 0b0000
                self.dut['global_conf']['PixConfLd'] = 0b00
                self.dut.write_global()
                # test hit
                self.dut['global_conf']['TestHit'] = 1
                self.dut['global_conf']['SignLd'] = 0
                self.dut['global_conf']['InjEnLd'] = 0
                self.dut['global_conf']['TDacLd'] = 0
                self.dut['global_conf']['PixConfLd'] = 0
                # all multi columns in parallel
                self.dut['global_conf']['OneSr'] = 0
                self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(
                    columns)
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

                # enable testhit pulse and trigger
                wiat_for_read = (16 + columns.count(True) *
                                 (4 * 64 / mask_steps) * 2) * (20 / 2) + 100
                # this should based on mask and enabled columns
                self.dut['testhit'].set_delay(wiat_for_read * 8)
                self.dut['testhit'].set_width(3)
                self.dut['testhit'].set_repeat(repeat_command)
                self.dut['testhit'].set_en(False)

                self.dut['trigger'].set_delay(400 - 4)
                self.dut['trigger'].set_width(8)
                self.dut['trigger'].set_repeat(1)
                self.dut['trigger'].set_en(True)

                lmask = [1] + ([0] * (mask_steps - 1))
                lmask = lmask * ((64 * 64) / mask_steps + 1)
                lmask = lmask[:64 * 64]
                bv_mask = bitarray.bitarray(lmask)

                with self.readout():

                    for i in range(mask_steps):

                        self.dut['pixel_conf'][:] = bv_mask
                        bv_mask[1:] = bv_mask[0:-1]
                        bv_mask[0] = 0

                        self.dut.write_pixel_col()
                        time.sleep(0.1)

                        self.dut['testhit'].start()

                        if os.environ.get('TRAVIS'):
                            logging.debug('.')

                        while not self.dut['testhit'].is_done():
                            time.sleep(0.005)
                            pass

                        while not self.dut['trigger'].is_done():
                            time.sleep(0.005)
                            pass

                        # just some time for last read
                    self.dut['trigger'].set_en(False)

                    # self.fifo_readout.print_readout_status()

                time.sleep(1)
                self.meta_data_table.attrs.power_status = yaml.dump(
                    self.dut.power_status())
                self.meta_data_table.attrs.dac_status = yaml.dump(
                    self.dut.dac_status())
                temp = self.dut['ntc'].get_temperature('C')
                logging.info('Temperature: %s', str(temp))
                self.meta_data_table.attrs.temp = yaml.dump(str(temp))
                self.h5_file.close()
                logging.info('Data Output Filename: %s',
                             self.output_filename + '.h5')
                self.analyze()
                self.dut.power_down()
                time.sleep(2)

        self.shmoo_plotting()

    def analyze(self):
        h5_filename = self.output_filename + '.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]
            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            in_file_h5.create_table(
                in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)
            hits = hit_data['col'].astype(np.uint16)
            hits = hits * 64
            hits = hits + hit_data['row']
            value = np.bincount(hits)
            value = np.pad(
                value, (0, 64 * 64 - value.shape[0]), 'constant', constant_values=0)
            full_occupation = np.full(4096, 100, dtype=int)
            difference = full_occupation - value
            tot_diff = abs(np.sum(difference))
            if tot_diff < 400000:
                plots = True
            self.not_fired.append(tot_diff)
            logging.info('Shmoo plot entry: %s', str(tot_diff))

        if self.plots == True:
            occ_plot, H = plotting.plot_occupancy(h5_filename)
            tot_plot, _ = plotting.plot_tot_dist(h5_filename)
            lv1id_plot, _ = plotting.plot_lv1id_dist(h5_filename)
            output_file(self.output_filename + '.html', title=self.run_name)
            save(Column(occ_plot, tot_plot, lv1id_plot))
            return H

    def shmoo_plotting(self):
        ''' pixel register shmoo plot '''
        plotname = self.scantype + "_Shmoo_" + \
            str(time.strftime("%Y%m%d_%H%M%S_")) + ".pdf"
        shmoopdf = PdfPages(plotname)
        shmoonp = np.array(self.not_fired)
        data = shmoonp.reshape(len(self.voltages), -1, order='F')
        fig, ax = plt.subplots()
        plt.title('Missed Voltage Pulses (100 inj. x 4096 pix.)')
        ax.set_axis_off()
        fig.text(0.70, 0.05, self.clock_name + ' (MHz)', fontsize=14)
        fig.text(0.02, 0.90, 'Supply voltage (V)', fontsize=14, rotation=90)
        tb = Table(ax, bbox=[0.01, 0.01, 0.99, 0.99])
        ncols = len(self.bitfiles)
        nrows = len(self.voltages)
        width, height = 1.0 / ncols, 1.0 / nrows
        # Add cells
        for (i, j), val in np.ndenumerate(data):
            color = ''
            val = abs(val)
            # use different colors for frequencies above 110MHz where the FPGA is no more reliable
            if (j < 10):
                if (val == 0):
                    color = 'green'
                if (val > 0 & val < 50):
                    color = 'yellow'
                if val > 50:
                    color = 'red'
                # greyscale
            if (j > 9):
                if (val == 0):
                    color = 'white'
                if (val > 0 & val < 50):
                    color = '#b2b2b2'
                if val > 50:
                    color = '#9a9a9a'
            tb.add_cell(i, j, width, height, text=str(val),
                        loc='center', facecolor=color)
        # Row Labels...
        for i in range(len(self.voltages)):
            tb.add_cell(i, -1, width, height, text=str(self.voltages[i]), loc='right',
                        edgecolor='none', facecolor='none')
        # Column Labels...
        colj = 0
        for j in self.bitfiles.iterkeys():
            tb.add_cell(nrows + 1, colj, width, height / 2, text=str(j), loc='center',
                        edgecolor='none', facecolor='none')
            colj += 1
        ax.add_table(tb)
        shmoopdf.savefig()
        shmoopdf.close()


if __name__ == "__main__":
    scan = DigitalScanFreq(**local_configuration)
    scan.scan(**local_configuration)
