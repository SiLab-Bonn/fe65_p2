
from fe65p2.scan_base import ScanBase
import fe65p2.plotting as  plotting
import fe65p2.analysis as analysis
import time
import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, show, vplot, hplot, save

from progressbar import ProgressBar
import yaml
from basil.dut import Dut
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


local_configuration = {
    "mask_steps": 1,
    "repeat_command": 101,
    "scan_range": [0.05, 1.25, 0.05],
    "vthin1Dac": 55,
    "preCompVbnDac" : 115,
    "columns" : [True] * 2 + [True] * 14,
    "mask_filename": '',
    "pix_list": [100,200,300,400,500]
}


class TimewalkScan(ScanBase):
    scan_id = "timewalk_scan"

    def scan(self, pix_list=((6, 20),), mask_steps=4, repeat_command=101, columns = [True] * 16, scan_range = [0, 1.2, 0.1], vthin1Dac = 80, vthin2Dac= 0, PrmpVbpDac=80, preCompVbnDac = 50, mask_filename='', **kwargs):

        '''Scan loop
        This scan is to measure time walk. The charge injection can be driven by the GPAC or an external device.
        In the latter case the device is Agilent 33250a connected through serial port.
        The time walk and TOT are measured by a TDC module in the FPGA.
        The output is an .h5 file (data) and an .html file with plots.

        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''

        inj_factor = 1.0
        INJ_LO = 0.0
        try:
            dut = Dut(ScanBase.get_basil_dir(self)+'/examples/lab_devices/agilent33250a_pyserial.yaml')
            dut.init()
            logging.info('Connected to '+str(dut['Pulser'].get_info()))
        except RuntimeError:
            INJ_LO = 0.2
            inj_factor = 2.0
            logging.info('External injector not connected. Switch to internal one')
            self.dut['INJ_LO'].set_voltage(INJ_LO, unit='V')

        self.dut['global_conf']['PrmpVbpDac'] = 80
        self.dut['global_conf']['vthin1Dac'] = 255
        self.dut['global_conf']['vthin2Dac'] = 0
        self.dut['global_conf']['vffDac'] = 24
        self.dut['global_conf']['PrmpVbnFolDac'] = 51
        self.dut['global_conf']['vbnLccDac'] = 1
        self.dut['global_conf']['compVbnDac'] = 25
        self.dut['global_conf']['preCompVbnDac'] = 50
        
        self.dut.write_global() 
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

        self.dut['global_conf']['OneSr'] = 1  
        
        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray([True] * 16) #(columns)
        self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray([True] * 16)     
        self.dut.write_global()

        self.dut['pixel_conf'].setall(False)
        self.dut.write_pixel()
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut.write_global()
        self.dut['global_conf']['InjEnLd'] = 0

        mask_en = np.full([64,64], False, dtype = np.bool)
        mask_tdac = np.full([64,64], 16, dtype = np.uint8)
        
        for inx, col in enumerate(columns):
           if col:
                mask_en[inx*4:(inx+1)*4,:]  = True
        
        if mask_filename:
            logging.info('Using pixel mask from file: %s', mask_filename)
        
            with tb.open_file(mask_filename, 'r') as in_file_h5:
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                mask_en = in_file_h5.root.scan_results.en_mask[:]
        
        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)
        
        self.dut['global_conf']['OneSr'] = 1
        self.dut.write_global()


        self.dut['inj'].set_delay(1000000)
        self.dut['inj'].set_width(1000)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(400-4)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(False)

        logging.debug('Enable TDC')
        self.dut['tdc']['RESET'] = True
        self.dut['tdc']['EN_TRIGGER_DIST'] = True
        self.dut['tdc']['ENABLE_EXTERN'] = False
        self.dut['tdc']['EN_ARMING'] = False
        self.dut['tdc']['EN_INVERT_TRIGGER'] = False
        self.dut['tdc']['EN_INVERT_TDC'] = False
        self.dut['tdc']['EN_WRITE_TIMESTAMP'] = True

        scan_range = np.arange(scan_range[0], scan_range[1] , scan_range[2]) / inj_factor
        scan_range=np.append(scan_range,0.3/inj_factor)
        scan_range = np.append(scan_range, 0.6 / inj_factor)
        scan_range=np.append(scan_range, 1.0 / inj_factor)
        self.pixel_list = pix_list

        p_counter = 0

        for pix in pix_list:
            mask_en = np.full([64, 64], False, dtype=np.bool)
            mask_en[pix[0], pix[1]] = True
            self.dut.write_en_mask(mask_en)
            self.dut.write_inj_mask(mask_en)

            self.inj_charge = []
            for idx, k in enumerate(scan_range):
                dut['Pulser'].set_voltage(INJ_LO, float(INJ_LO + k), unit='V')
                self.dut['INJ_HI'].set_voltage( float(INJ_LO + k), unit='V')
                self.inj_charge.append(float(k)*1000.0*ScanBase.cap_fac(self))

                time.sleep(0.5)

                with self.readout(scan_param_id = idx+p_counter*len(scan_range)):
                    logging.info('Scan Parameter: %f (%d of %d)', k, idx+1, len(scan_range))
                    self.dut['tdc']['ENABLE'] = True

                    self.dut['global_conf']['vthin1Dac'] = 255
                    self.dut['global_conf']['vthin2Dac'] = 0
                    self.dut['global_conf']['PrmpVbpDac'] = 80
                    self.dut['global_conf']['preCompVbnDac'] = 160 #50
                    self.dut.write_global()
                    time.sleep(0.1)

                    self.dut['global_conf']['vthin1Dac'] = vthin1Dac
                    self.dut['global_conf']['vthin2Dac'] = vthin2Dac
                    self.dut['global_conf']['PrmpVbpDac'] = PrmpVbpDac
                    self.dut['global_conf']['preCompVbnDac'] = preCompVbnDac
                    self.dut.write_global()
                    time.sleep(0.1)

                    self.dut['inj'].start()


                    while not self.dut['inj'].is_done():
                        pass

                    while not self.dut['trigger'].is_done():
                        pass

                    self.dut['tdc'].ENABLE = 0
            p_counter+=1


    def tdc_table(self,scanrange):
        h5_filename = self.output_filename +'.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]
            if (meta_data.shape[0]==0): return

            repeat_command=in_file_h5.root.meta_data.attrs.kwargs
            a=repeat_command.rfind("repeat_command: ")
            repeat_command=repeat_command[a+len("repeat_command: "):a+len("repeat_command: ")+7]
            a = repeat_command.rfind("\n")
            repeat_command = int(repeat_command[0:a])
            param, index = np.unique(meta_data['scan_param_id'], return_index=True)
            pxl_list = []
            for p in param:
                pix_no = int(p)/int(len(self.inj_charge))
                pxl_list.append(self.pixel_list[pix_no][0]*64+self.pixel_list[pix_no][1])
            index = index[1:]
            index = np.append(index, meta_data.shape[0])
            index = index - 1
            stops = meta_data['index_stop'][index]
            split = np.split(raw_data, stops)
            avg_tdc = []
            avg_tdc_err = []
            avg_del = []
            avg_del_err = []

            hits = []
            deletelist=()
            for i in range(len(split[:-1])): #loop on pulses
                rwa_data_param = split[i]
                tdc_data = rwa_data_param & 0xFFF  # take last 12 bit
                tdc_delay = (rwa_data_param & 0x0FF00000) >> 20
                counter = 0.0
                TOT_sum = 0.0
                DEL_sum = 0.0
                if (tdc_data.shape[0]==0 or tdc_data.shape[0]==1):
                    counter = 1.0
                for j in range(tdc_data.shape[0]): #loop on repeats
                    if (j>0):
                        counter += 1
                        TOT_sum += tdc_data[j]
                        DEL_sum += tdc_delay[j]
                if (counter > 1):
                    hits.append(counter)
                    avg_tdc.append((float(TOT_sum)/float(counter))*1.5625)
                    avg_tdc_err.append(1.5625/(np.sqrt(12.0*counter)))
                    avg_del.append((float(DEL_sum)/float(counter))*1.5625)
                    avg_del_err.append(1.5625/(np.sqrt(12.0*counter)))
                else:
                    deletelist=np.append(deletelist,i)
            pxl_list=np.delete(pxl_list,deletelist)
            newpix=[0]
            pix_no_old=pxl_list[0]
            runparam=0
            for p in pxl_list:
                if p!=pix_no_old:
                    newpix=np.append(newpix,runparam)
                pix_no_old=p
                runparam=runparam+1
                addedvalues=0
            for pixels in range(len(newpix)):
               missingvalues=0
               if newpix[pixels]==newpix[-1]:
                    missingvalues= scanrange - abs(newpix[pixels]+addedvalues-len(hits))
               else:
                if abs(newpix[pixels]-newpix[pixels+1]) < scanrange:
                    missingvalues= scanrange - abs(newpix[pixels]-newpix[pixels+1])
               if missingvalues!=0:
                    hits = np.insert(hits,newpix[pixels]+addedvalues,np.zeros(missingvalues))
                    avg_tdc = np.insert(avg_tdc, newpix[pixels] + addedvalues, np.zeros(missingvalues))
                    avg_tdc_err = np.insert(avg_tdc_err, newpix[pixels] + addedvalues, np.zeros(missingvalues))
                    avg_del = np.insert(avg_del, newpix[pixels] + addedvalues, np.zeros(missingvalues))
                    avg_del_err = np.insert(avg_del_err, newpix[pixels] + addedvalues, np.zeros(missingvalues))
                    pxl_list = np.insert(pxl_list, newpix[pixels] + addedvalues, (pxl_list[newpix[pixels]+ addedvalues])*np.ones(missingvalues))
               addedvalues = addedvalues + missingvalues
            injections=[]
            for pixels in range(int(len(pxl_list)/len(self.inj_charge))):
                for i in range(len(self.inj_charge)):
                    injections=np.append(injections,self.inj_charge[i])
            pix, stop = np.unique(pxl_list, return_index=True)
            stop = np.sort(stop)
            stop = list(stop)
            stop.append(len(avg_tdc))
            avg_tab = np.rec.fromarrays([injections, pxl_list, hits, avg_tdc, avg_tdc_err, avg_del, avg_del_err],
                                        dtype=[('charge', float), ('pixel_no', int), ('hits', int),
                                               ('tot_ns', float), ('err_tot_ns', float), ('delay_ns', float),
                                               ('err_delay_ns', float)])
            in_file_h5.createTable(in_file_h5.root, 'tdc_data', avg_tab, filters=self.filter_tables)
            thresholds = ()
            expfit0=()
            expfit1=()
            expfit2=()
            expfit3=()
            pixels=()
            for i in range(len(stop) - 1):
                s1 = int(stop[i])
                s2 = int(stop[i + 1])
                A, mu, sigma = analysis.fit_scurve(hits[s1:s2], injections[s1:s2])
                if np.max(hits[s1:s2]) > (repeat_command+100) or mu >2000:
                    thresholds = np.append(thresholds, 0)
                    expfit0 = np.append(expfit0, 0)
                    expfit1 = np.append(expfit1, 0)
                    expfit2 = np.append(expfit2, 0)
                    expfit3 = np.append(expfit3, 0)
                    pixels = np.append(pixels, pxl_list[s1])
                    continue
                for values in range(s1, s2):
                    if injections[values] >= 5 / 4 * mu:
                        s1 = values
                        break
                numberer = 0
                hitvaluesold = hits[-1]
                for hitvalues in hits[s1:s2]:
                    if abs(hitvalues - hitvaluesold) <= 1 and hitvalues != 0:
                        break
                    numberer = numberer + 1
                    hitvaluesold = hitvalues
                if numberer==len(avg_del[s1:s2]):
                    numberer=0
                expfit = analysis.fit_exp(injections[s1:s2], avg_del[s1:s2], mu, abs(numberer))
                startexp = -expfit[0] * np.log((25.0 + np.min(avg_del[s1:s2]) - expfit[3]) / expfit[2]) - expfit[1]
                if np.isnan(startexp) or startexp >= 2000:
                    startexp=0
                thresholds=np.append(thresholds,startexp)
                expfit0=np.append(expfit0,expfit[0])
                expfit1=np.append(expfit1,expfit[1])
                expfit2=np.append(expfit2,expfit[2])
                expfit3=np.append(expfit3,expfit[3])
                pixels=np.append(pixels,pxl_list[s1])
            thresh=np.rec.fromarrays([pixels,thresholds,expfit0,expfit1,expfit2,expfit3],dtype=[('pixel_no', int),('td_threshold', float),
                                    ('expfit0', float), ('expfit1', float), ('expfit2', float), ('expfit3', float)])
            in_file_h5.createTable(in_file_h5.root, 'td_threshold', thresh, filters=self.filter_tables)
        p1, p2, single_scan = plotting.plot_timewalk(h5_filename)
        output_file(self.output_filename + '.html', title=self.run_name)
        status=plotting.plot_status(h5_filename)
        save(hplot(vplot(p1,p2,status),single_scan))

if __name__ == "__main__":
     Timescan = TimewalkScan()
     Timescan.start(**local_configuration)
     scanrange=local_configuration['scan_range']
     Timescan.tdc_table(((scanrange[1]-scanrange[0])/scanrange[2])+3)
