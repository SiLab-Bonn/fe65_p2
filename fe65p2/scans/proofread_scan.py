from fe65p2.scan_base import ScanBase
import time
import os
import sys
import bitarray
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
local_configuration = {
    "mask_steps" : 4,
    "columns" : [True] * 16,
    # DAC parameters
    "PrmpVbpDac": 36,
    "vthin1Dac": 255,
    "vthin2Dac": 0,
    "vffDac": 42,
    "PrmpVbnFolDac": 51,
    "vbnLccDac": 1,
    "compVbnDac": 25,
    "preCompVbnDac": 50
}

'''
	This scan writes a patterns of bits in the pixel registers
	and reads them back. The occuring errors are counted and a
	Shmoo plot is produced and printed on a .pdf
	The supply voltage and the .bit file loaded can be changed
	from here.
	Global registers are checked as well and a second Shmoo
	plot is printed in the same pdf.
'''


class proofread_scan(ScanBase):
    scan_id = "proof_read_scan"

    def __init__(self, dut_conf=None):
        super(proofread_scan, self).__init__(dut_conf)

    def scan(self, mask_steps=4, columns=[True] * 16, **kwargs):

        # bitfiles = ["fe65p2_mio_40MHz.bit"]
        #		voltages = [2.0]

        self.dut['global_conf']['PrmpVbpDac'] = kwargs['PrmpVbpDac']
        self.dut['global_conf']['vthin1Dac'] = kwargs['vthin1Dac']
        self.dut['global_conf']['vthin2Dac'] = kwargs['vthin2Dac']
        self.dut['global_conf']['vffDac'] = 42
        self.dut['global_conf']['PrmpVbnFolDac'] = kwargs['PrmpVbnFolDac']
        self.dut['global_conf']['vbnLccDac'] = kwargs['vbnLccDac']
        self.dut['global_conf']['compVbnDac'] = kwargs['compVbnDac']
        self.dut['global_conf']['preCompVbnDac'] = 50

        #scan_path = os.path.dirname(os.path.realpath(sys.argv[0]))
        #path = scan_path.replace('fe65p2/scans','firmware/bits/goodSPI_bits/')
        path = "/home/topcoup/Applications/fe65_p2/firmware/bits/goodSPI_bits/"
        self.bitfiles = ["fe65p2_mio_3MHz.bit", "fe65p2_mio_4MHz.bit", "fe65p2_mio_6MHz.bit",
                         "fe65p2_mio_8MHz.bit", "fe65p2_mio_12MHz.bit", "fe65p2_mio_16MHz.bit",
                         "fe65p2_mio_24MHz.bit", "fe65p2_mio_32MHz.bit"]
        self.voltages = [1.25, 1.2, 1.1, 1.0, 0.95, 0.90]

        self.shmoo_errors = []
        self.shmoo_global_errors = []

        for bitfile in self.bitfiles:
            logging.info("Loading " + bitfile)
            setstatus = self.dut['intf']._sidev.DownloadXilinx(path + bitfile)
            try:
                setstatus == 0
            except:
                break

            for volt in self.voltages:

                self.dut['control']['RESET'] = 1
                self.dut['control'].write()
                self.dut['control']['RESET'] = 0
                self.dut['control'].write()
                # to change the supply voltage
                self.dut['VDDA'].set_current_limit(200, unit='mA')
                self.dut['VDDA'].set_voltage(volt, unit='V')
                self.dut['VDDA'].set_enable(True)
                self.dut['VDDD'].set_voltage(volt, unit='V')
                self.dut['VDDD'].set_enable(True)
                self.dut['VAUX'].set_voltage(1.25, unit='V')
                self.dut['VAUX'].set_enable(True)
                # global reg
                self.dut['global_conf']['PrmpVbpDac'] = kwargs['PrmpVbpDac']
                self.dut['global_conf']['vthin1Dac'] = kwargs['vthin1Dac']
                self.dut['global_conf']['vthin2Dac'] = kwargs['vthin2Dac']
                self.dut['global_conf']['vffDac'] = 42
                self.dut['global_conf']['PrmpVbnFolDac'] = 51
                self.dut['global_conf']['vbnLccDac'] = 51
                self.dut['global_conf']['compVbnDac'] = kwargs['compVbnDac']
                self.dut['global_conf']['preCompVbnDac'] = 50

                self.dut['global_conf']['OneSr'] = 1
                self.dut['global_conf']['SPARE'] = 0  # added by me, default 0
                self.dut['global_conf']['ColEn'] = 0  # added by me, default 0
                self.dut['global_conf']['ColSrEn'] = 15  # added by me, default 15
                self.dut['global_conf']['Latency'] = 400  # added by me, default 0
                self.dut['global_conf']['ColSrEn'].setall(True)  # enable programming of all columns

                self.dut.write_global()
                self.dut.write_global()  # need to write 2 times!

                logging.info(self.dut.power_status())  # prints power supply

                send = self.dut['global_conf'].tobytes()
                rec = self.dut['global_conf'].get_data(size=19)
                rec[18] = rec[18] & 0b1000000
                glob_errors = [i for i in range(len(send)) if send[i] != rec[i]]
                if (len(glob_errors) > 0): logging.warning("*** GLOBAL ERRORS " + str(len(glob_errors)))
                self.shmoo_global_errors.append(len(glob_errors))
                #				for j in range(len(glob_errors)):
                #					print "in position ", j, "value ", glob_errors[j]	#if you want to know where is the error

                # pixel reg
                self.dut['pixel_conf'][0] = 1
                self.dut.write_pixel()
                self.dut['control']['RESET'] = 0b11
                self.dut['control'].write()

                lmask = ([0] * (mask_steps - 1)) + [1]  # 1,0,0,0 pattern
                lmask = lmask * ((64 * 64) / mask_steps + 1)
                lmask = lmask[:64 * 64]  # 1,0,0,0 pattern for a total of 4096 bits
                bv_mask = bitarray.bitarray(lmask)  # convert in binary
                errors = []  # used for screen output - debug
                ERR = []  # pixel errors storage
                err_count = 0

                logging.info('Temperature: %s', str(self.dut['ntc'].get_temperature('C')))

                for i in range(0, 4):
                    self.dut['pixel_conf'][:] = bv_mask
                    self.dut.write_pixel()
                    self.dut.write_pixel()
                    time.sleep(0.5)
                    returned_data = ''.join(format(x, '08b') for x in self.dut['pixel_conf'].get_data())
                    returned_data_reversed = returned_data[::-1]  # the readout comes upside down

                    pix_send = bv_mask
                    pix_rec = bitarray.bitarray(returned_data_reversed)

                    logging.debug('s ' + str(pix_send[:8]))
                    logging.debug('r ' + str(pix_rec[:8]))

                    errors.append([])
                    for bit in xrange(len(pix_send)):
                        if pix_send[bit] != pix_rec[bit]:
                            errors[i].append(bit)
                            ERR.append(bit)
                            err_count += 1

                    time.sleep(0.2)
                    bv_mask = bv_mask[1:] + bv_mask[:1]  # shift the bit pattern

                self.shmoo_errors.append(err_count)

                if len(errors[i]) > 0:
                    logging.warning("*** PIXEL ERRORS  ***")
                    for i in range(0, len(errors)):
                        logging.warning("iteration " + str(i) + " errors " + str(
                            len(errors[i])))  # , " at ", ' '.join([str(x) for x in errors[i]])


    def shmoo_plotting(self):
        ''' pixel register shmoo plot '''
        plotname = "PixReg_"+str(time.strftime("%Y%m%d_%H%M%S_"))+".pdf"
        shmoopdf = PdfPages(plotname)
        shmoonp = np.array(self.shmoo_errors)
        data = shmoonp.reshape(len(self.voltages), -1, order='F')
        fig, ax = plt.subplots()
        plt.title('Pixel registers errors')
        ax.set_axis_off()
        fig.text(0.70, 0.05, 'SPI clock (MHz)', fontsize=14)
        fig.text(0.02, 0.90, 'Supply voltage (V)', fontsize=14, rotation=90)
        tb = Table(ax, bbox=[0.01,0.01,0.99,0.99])
        ncols = len(self.bitfiles)
        nrows = len(self.voltages)
        width, height = 1.0 / ncols, 1.0 / nrows
        # Add cells
        for (i, j), val in np.ndenumerate(data):
            color = ''
            if val == 0: color = 'green'
            if (val > 0 & val < 10): color = 'yellow'
            if val > 10: color = 'red'
            tb.add_cell(i, j, width, height, text=str(val),
                        loc='center', facecolor=color)
        # Row Labels...
        for i in range(len(self.voltages)):
            tb.add_cell(i, -1, width, height, text=str(self.voltages[i]) + 'V', loc='right',
                        edgecolor='none', facecolor='none')
        # Column Labels...
        for j in range(len(self.bitfiles)):
            freq_label = self.bitfiles[j].replace('fe65p2_mio_', '').replace('MHz.bit','')
            tb.add_cell(nrows + 1, j, width, height / 2, text=freq_label + ' MHz', loc='center',
                        edgecolor='none', facecolor='none')
        ax.add_table(tb)
        shmoopdf.savefig()

        ''' global register shmoo plot '''
        shmoo_glob_np = np.array(self.shmoo_global_errors)
        data_g = shmoo_glob_np.reshape(len(self.voltages), -1, order='F')
        fig_g, ax_g = plt.subplots()
        ax_g.set_axis_off()
        fig_g.text(0.70, 0.05, 'SPI clock (MHz)', fontsize=14)
        fig_g.text(0.02, 0.90, 'Supply voltage (V)', fontsize=14, rotation=90)
        tb_g = Table(ax_g, bbox=[0.01,0.01,0.99,0.99])
        plt.title('Global registers errors')
        # Add cells
        for (i, j), val_g in np.ndenumerate(data_g):
            color = ''
            if val_g == 0: color = 'green'
            if val_g > 0: color = 'red'
            tb_g.add_cell(i, j, width, height, text=str(val_g),
                          loc='center', facecolor=color)
        # Row Labels...
        for i in range(len(self.voltages)):
            tb_g.add_cell(i, -1, width, height, text=str(self.voltages[i]) + 'V', loc='right',
                          edgecolor='none', facecolor='none')
        # Column Labels...
        for j in range(len(self.bitfiles)):
            freq_label = self.bitfiles[j].replace('fe65p2_mio_', '').replace('MHz.bit','')
            tb_g.add_cell(nrows + 1, j, width, height / 2, text=freq_label + ' MHz', loc='center',
                          edgecolor='none', facecolor='none')
        ax_g.add_table(tb_g)
        shmoopdf.savefig()
        shmoopdf.close()


if __name__ == "__main__":
    scan = proofread_scan("/home/user/Desktop/carlo/fe65_p2/fe65p2/fe65p2.yaml")
    scan.start(**local_configuration)
    scan.shmoo_plotting()
