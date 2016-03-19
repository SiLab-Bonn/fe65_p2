from fe65p2.scan_base import ScanBase
import time
import bitarray
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table

logging.basicConfig(level=logging.INFO,
					format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
local_configuration = {}

'''
	This scan writes a patterns of bits in the pixel registers
	and reads them back. The occuring errors are counted and a
	Shmoo plot is produced and printed on a .pdf
	The supply voltage and the .bit file loaded can be changed
	from here.
	Global registers are checked as well and a second Shmoo
	plot is printed in the same pdf.
'''

class Proofread_Scan(ScanBase):
	scan_id = "proof_read_scan"

	def __init__(self, dut_conf):
		super(Proofread_Scan, self).__init__(dut_conf)

	def scan(self, mask_steps=4, columns=[True] * 16, **kwargs):
#		bitfiles = ["fe65p2_mio_40MHz.bit"]
#		voltages = [2.0]
		path = "/home/carlo/fe65_p2/firmware/ise/"
		bitfiles = ["fe65p2_mio_1MHz.bit", "fe65p2_mio_2MHz.bit","fe65p2_mio_5MHz.bit", "fe65p2_mio_10MHz.bit", "fe65p2_mio_20MHz.bit","fe65p2_mio_40MHz.bit"]
		voltages = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.85]

		shmoo_errors = []
		shmoo_global_errors = []

		for bitfile in bitfiles:
			print "Loading " + bitfile
			setstatus = self.dut['intf']._sidev.DownloadXilinx(path+bitfile)
			try:
				setstatus == 0
			except:
				break

			for volt in voltages:

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
				self.dut['VAUX'].set_voltage(volt, unit='V')
				self.dut['VAUX'].set_enable(True)
				# global reg
				self.dut['global_conf']['OneSr'] = 1
				self.dut['global_conf']['PrmpVbpDac'] = 36
				self.dut['global_conf']['vthin1Dac'] = 255
				self.dut['global_conf']['vthin2Dac'] = 0
				self.dut['global_conf']['vffDac'] = 42  # added by me, default 42
				self.dut['global_conf']['SPARE'] = 0  # added by me, default 0
				self.dut['global_conf']['ColEn'] = 0  # added by me, default 0
				self.dut['global_conf']['ColSrEn'] = 15  # added by me, default 15
				self.dut['global_conf']['Latency'] = 400  # added by me, default 0
				self.dut['global_conf']['PrmpVbnFolDac'] = 0
				self.dut['global_conf']['vbnLccDac'] = 51
				self.dut['global_conf']['compVbnDac'] = 25
				self.dut['global_conf']['preCompVbnDac'] = 50
				self.dut['global_conf']['ColSrEn'].setall(True)  # enable programming of all columns
				self.dut.write_global()
				self.dut.write_global()  # need to write 2 times!

				print "*** POWER STATUS ***"
				print (scan.dut.power_status())		#prints power supply

				send = self.dut['global_conf'].tobytes()
				rec = self.dut['global_conf'].get_data(size=19)
				rec[18] = rec[18] & 0b1000000
				glob_errors = [i for i in range(len(send)) if send[i] != rec[i]]
				print "*** GLOBAL ERRORS ", len(glob_errors)
				shmoo_global_errors.append(len(glob_errors))
#				for j in range(len(glob_errors)):
#					print "in position ", j, "value ", glob_errors[j]	#if you want to know where is the error

				# pixel reg
				self.dut['pixel_conf'][0] = 1
				self.dut.write_pixel()
				self.dut['control']['RESET'] = 0b11
				self.dut['control'].write()

				lmask = ([0] * (mask_steps - 1)) + [1]  		# 1,0,0,0 pattern
				lmask = lmask * ((64 * 64) / mask_steps + 1)
				lmask = lmask[:64 * 64]  						# 1,0,0,0 pattern for a total of 4096 bits
				bv_mask = bitarray.bitarray(lmask) 				# convert in binary
				errors = []		#used for screen output - debug
				ERR = []		#pixel errors storage
				err_count = 0

				for i in range(0, 4):
					self.dut['pixel_conf'][:] = bv_mask
					self.dut.write_pixel()
					self.dut.write_pixel()
					time.sleep(0.5)
					returned_data = ''.join(format(x, '08b') for x in self.dut['pixel_conf'].get_data())
					returned_data_reversed = returned_data[::-1] #the readout comes upside down

					pix_send = bv_mask
					pix_rec = bitarray.bitarray(returned_data_reversed)

					print 's', str(pix_send[:8]) #on screen - debug
					print 'r', str(pix_rec[:8])  #on screen - debug

					errors.append([])
					for bit in xrange(len(pix_send)):
						if pix_send[bit] != pix_rec[bit]:
							errors[i].append(bit)
							ERR.append(bit)
							err_count += 1

					time.sleep(0.2)
					bv_mask = bv_mask[1:] + bv_mask[:1]	#shift the bit pattern

				shmoo_errors.append(err_count)

				print "*** PIXEL ERRORS  *** \n flagged iterations ", len([c for c in range(len(errors)) if len(errors[c]) > 0])
				for i in range(0, len(errors)):
					if len(errors[i]) > 0:
						print "no. ", i, " errors ", len(errors[i]) #, " at ", ' '.join([str(x) for x in errors[i]])


		''' pixel register shmoo plot '''
		shmoopdf = PdfPages('shmoo.pdf')
		shmoonp = np.array(shmoo_errors)
		data = shmoonp.reshape(len(voltages),-1,order='F')
		fig, ax = plt.subplots()
		plt.title('Pixel registers errors')
		ax.set_axis_off()
		tb = Table(ax, bbox=[0,0,1,1])
		ncols = len(bitfiles)
		nrows = len(voltages)
		width, height = 1.0 / ncols, 1.0 / nrows
			# Add cells
		for (i,j), val in np.ndenumerate(data):
			color = ''
			if val == 0: color = 'green'
			if (val > 0 & val < 10): color = 'yellow'
			if val > 10: color = 'red'
			tb.add_cell(i, j, width, height, text=str(val),
				loc='center', facecolor=color)
		# Row Labels...
		for i in range(len(voltages)):
			tb.add_cell(i, -1, width, height, text=str(voltages[i])+'V', loc='right',
						edgecolor='none', facecolor='none')
		# Column Labels...
		for j in range(len(bitfiles)):
			tb.add_cell(nrows+1, j, width, height/2, text=bitfiles[j][-9:-7]+' MHz', loc='center',
							   edgecolor='none', facecolor='none')
		ax.add_table(tb)
		shmoopdf.savefig()

		''' global register shmoo plot '''
		shmoo_glob_np = np.array(shmoo_global_errors)
		data_g = shmoo_glob_np.reshape(len(voltages),-1,order='F')
		fig_g, ax_g = plt.subplots()
		ax_g.set_axis_off()
		tb_g = Table(ax_g, bbox=[0,0,1,1])
		plt.title('Global registers errors')
		# Add cells
		for (i,j), val_g in np.ndenumerate(data_g):
			color = ''
			if val_g == 0: color = 'green'
			if val_g > 0: color = 'red'
			tb_g.add_cell(i, j, width, height, text=str(val_g),
				loc='center', facecolor=color)
		# Row Labels...
		for i in range(len(voltages)):
			tb_g.add_cell(i, -1, width, height, text=str(voltages[i])+'V', loc='right',
						edgecolor='none', facecolor='none')
		# Column Labels...
		for j in range(len(bitfiles)):
			tb_g.add_cell(nrows+1, j, width, height/2, text=bitfiles[j][-9:-7]+' MHz', loc='center',
							   edgecolor='none', facecolor='none')
		ax_g.add_table(tb_g)
		shmoopdf.savefig()
		shmoopdf.close()


if __name__ == "__main__":
	scan = Proofread_Scan("/home/carlo/fe65_p2/fe65p2/fe65p2.yaml")
	scan.start(**local_configuration)