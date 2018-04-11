
from fe65p2.scan_base import ScanBase

import fe65p2.DGC_plotting as DGC_plotting
import time

import logging
from fe65p2 import fifo_readout

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from progressbar import ProgressBar
import fe65p2.scans.noise_tuning_columns as noise_cols

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from basil.dut import Dut
import os
import yaml

yaml_file = '/home/daniel/MasterThesis/fe65_p2/fe65p2/chip4.yaml'

local_configuration = {
    "mask_steps": 4,
    "repeat_command": 100,

    # DAC parameters
    #     "PrmpVbpDac": 36,
    #     "vthin1Dac": 20,
    #     "vthin2Dac": 0,
    #     "vffDac": 42,
    #     "PrmpVbnFolDac": 51,
    #     "vbnLccDac": 0,
    #     "compVbnDac": 25,
    #     "preCompVbnDac": 50

    #     "PrmpVbpDac": 120,
    #     "vthin1Dac": 130,
    #     "vthin2Dac": 0,
    #     "vffDac": 92,
    #     "PrmpVbnFolDac": 88,
    #     "vbnLccDac": 1,
    #     "compVbnDac": 90,
    #     "preCompVbnDac": 140,

    #"mask_filename": '/home/daniel/Documents/InterestingPlots/chip3/noise_tuning_14.22_31_0.h5',

}


class AnalogScan(ScanBase):
    scan_id = "analog_scan"

    def scan(self, mask_steps=4, repeat_command=500, columns=[True] * 16, mask_filename='', **kwargs):
        '''Scan loop

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''

        #columns = [True] + [False] * 15
        try:  # pulser
            pulser = Dut(ScanBase.get_basil_dir(self) + '/examples/lab_devices/agilent33250a_pyserial.yaml')
            pulser.init()
            logging.info('Connected to Pulser: ' + str(pulser['Pulser'].get_info()))
#             pulser['Pulser'].set_usr_func("FEI4_PULSE")
            pulse_width = 30000
            pulser['Pulser'].set_pulse_period(pulse_width * 10**-9)
            pulser['Pulser'].set_voltage(0., 0.6, unit='V')
        except:
            INJ_LO = 0.2
            self.dut['INJ_LO'].set_voltage(float(INJ_LO), unit='V')
            logging.info('External injector not connected. Switch to internal one')
            self.dut['INJ_LO'].set_voltage(0.1, unit='V')
            self.dut['INJ_HI'].set_voltage(1.2, unit='V')

        self.dut.set_for_configuration()

        self.dut['control']['RESET'] = 0b01
        self.dut['control']['DISABLE_LD'] = 0
        self.dut['control']['PIX_D_CONF'] = 0
        self.dut['control'].write()

        self.dut['control']['CLK_OUT_GATE'] = 1
        self.dut['control']['CLK_BX_GATE'] = 1
        self.dut['control'].write()
        time.sleep(0.1)

        self.dut['control']['RESET'] = 0b11
        self.dut['control'].write()

        self.dut.start_up()

        self.dut['global_conf']['OneSr'] = 1

        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray(columns)
        self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray(columns)
        self.dut.write_global()

        mask_inj = np.full([64, 64], False, dtype=np.bool)
        mask_en = np.full([64, 64], True, dtype=np.bool)
        mask_tdac = np.full([64, 64], 15, dtype=np.uint8)
        mask_hitor = np.full([64, 64], True, dtype=np.bool)

        file0 = kwargs.get("noise_col0")
        file1 = kwargs.get("noise_col1")
        file2 = kwargs.get("noise_col2")
        file3 = kwargs.get("noise_col3")
        file4 = kwargs.get("noise_col4")
        file5 = kwargs.get("noise_col5")
        file6 = kwargs.get("noise_col6")
        file7 = kwargs.get("noise_col7")
        mask_en_from_file, mask_tdac, vth1 = noise_cols.combine_prev_scans(
            file0=file0, file1=file1, file2=file2, file3=file3, file4=file4, file5=file5, file6=file6, file7=file7)
        vth1 += 20
        print vth1

#         if mask_filename:
#             logging.info('***** Using pixel mask from file: %s', mask_filename)
#
#             with tb.open_file(str(mask_filename), 'r') as in_file_h5:
#                 mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
#                 mask_en_from_file = in_file_h5.root.scan_results.en_mask[:]
#                 mask_en = in_file_h5.root.scan_results.en_mask[:]
#                 mask_tdac[mask_tdac == 32] = 31
#                 vth1 = yaml.load(in_file_h5.root.meta_data.attrs.vth1) + 20
#                 logging.info("vth1: %s" % str(vth1))
#                 print vth1

        ex_pix_disable_list = kwargs.get("ex_pix_disable_list")
        mask_en_from_file = mask_en_from_file.reshape(4096)
        mask_en_from_file[ex_pix_disable_list] = False
        mask_en_from_file = mask_en_from_file.reshape(64, 64)
        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac.astype(np.uint8))
        self.dut.write_inj_mask(mask_inj)
        self.dut.write_hitor_mask(mask_hitor)

        repeat_command = kwargs.get("repeat_command", repeat_command)

        # enable inj pulse and trigger
        wait_for_read = (16 + columns.count(True) * (4 * 64 / mask_steps) * 2) * (20 / 2) + 10000
        self.dut['inj'].set_delay(wait_for_read * 10)
        self.dut['inj'].set_width(1000)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(402)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        with self.readout():
            #             pbar = ProgressBar(maxval=mask_steps).start()
            for i in range(mask_steps):
                self.dut.set_for_configuration()

                mask_inj[:, :] = False
                mask_en[:, :] = False

                mask_inj = mask_inj.reshape(4096)
                mask_inj[i::mask_steps] = True
                mask_inj = mask_inj.reshape(64, 64)
                mask_en = mask_inj
                mask_hitor = mask_inj
                mask_en[(mask_en_from_file == False) & (mask_inj == True)] = False
                self.dut.write_inj_mask(mask_inj)
                self.dut.write_en_mask(mask_en)
                self.dut.write_hitor_mask()
                self.set_local_config()

                self.dut['inj'].start()
                # time.sleep(0.3)
                while not self.dut['inj'].is_done():
                    time.sleep(0.05)

                while not self.dut['trigger'].is_done():
                    time.sleep(0.05)
                print self.fifo_readout.get_record_count()
            # just some time for last read
            # self.dut['trigger'].set_en(False)
            # self.dut['inj'].start()

    def analyze(self):
        h5_filename = self.output_filename + '.h5'

        with tb.open_file(h5_filename, 'r+') as out_file_h5:
            raw_data = out_file_h5.root.raw_data[:]
            meta_data = out_file_h5.root.meta_data[:]
            scan_args = yaml.load(out_file_h5.root.meta_data.attrs.kwargs)

            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            out_file_h5.create_table(
                out_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)

            occ = np.histogram2d(x=hit_data['col'], y=hit_data['row'], bins=(64, 64), range=((0, 64), (0, 64)))[0]

            out_file_h5.create_carray(out_file_h5.root, name='HistOcc', title='Occupancy Histogram', obj=occ)
            pix_perc_great_repeats = float(occ[occ > scan_args['repeat_command']].shape[0]) / 4096
            pix_perc_less_repeats = float(occ[occ < scan_args['repeat_command']].shape[0]) / 4096
            pix_perc_eq_rep = float(occ[occ == scan_args['repeat_command']].shape[0]) / 4096
            print "occ average: ", occ.mean(), " occ std: ", occ.std(), " Num == repeats %: ", (float(occ[occ == scan_args['repeat_command']].shape[0]) / 4096) * 100
            print "num > repeats: ", occ[occ > scan_args['repeat_command']].shape[0]
            print np.where(occ > scan_args['repeat_command'] + 3)
            print np.where(np.reshape(occ, 4096) > scan_args['repeat_command'] + 3)
        pdfName = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/analog_scan_testing3.pdf'
        pp = PdfPages(pdfName)
        occ_plot = DGC_plotting.plot_occupancy(h5_filename)
        pp.savefig(occ_plot)
        plt.clf()
        tot_plot = DGC_plotting.plot_tot_dist(h5_filename)
        pp.savefig(tot_plot)
        plt.clf()
        lv1id_plot = DGC_plotting.plot_lv1id_dist(h5_filename)
        pp.savefig(lv1id_plot)
        pp.close()

        # return % occ, % > repeats, % < repeats
        return pix_perc_eq_rep, pix_perc_great_repeats, pix_perc_less_repeats


if __name__ == "__main__":
    scan = AnalogScan()
    yaml_kwargs = yaml.load(open(yaml_file))
    local_configuration.update(dict(yaml_kwargs))
    scan.start(**local_configuration)
    _, _, _ = scan.analyze()
