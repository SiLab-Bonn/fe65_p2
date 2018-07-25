#===============================================================================
# Code to measure the IV curve with a given source meter
#
# must have source meter, chip does not need to be powered on
#
#===============================================================================


from basil.dut import Dut
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import NullFormatter
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import csv
from itertools import izip


def measure():
    dut = Dut('/home/daniel/MasterThesis/basil/examples/lab_devices/keithley2410_pyserial.yaml')
    dut.init()
    print dut['Sourcemeter'].get_name()

    meas_dut = Dut('/home/daniel/MasterThesis/basil/examples/lab_devices/keithley2400_pyserial.yaml')
    meas_dut.init()
    print meas_dut['Sourcemeter'].get_name()
#     dut['Sourcemeter'].beeper_off()
    curr_lim = 7 * 10**-8
    volt_step = 3
    finished = False
    curr_out = []
    curr_out_err = []
    volt_out = []
    dut['Sourcemeter'].set_current_limit(10**-6)
    curr_volt = float(dut['Sourcemeter'].get_current()[:13])
    dut['Sourcemeter'].set_voltage(0.)

    while not finished:
        curr_volt = float(dut['Sourcemeter'].get_current()[:13])
        vlt = float(dut['Sourcemeter'].get_current()[:13])
        time.sleep(5.0)
        stable = False
        curr_list = []
        start_time = time.time()
        for _ in range(4):
            curr_list.append(float(meas_dut['Sourcemeter'].get_current()[14:27]))
            time.sleep(0.5)
        while not stable:
            curr = float(meas_dut['Sourcemeter'].get_current()[14:27])
            curr_list.append(curr)
            mean = np.mean(curr_list[-4:])
            if abs(np.mean(curr_list) - mean) <= 1 * 10**-10 or (time.time() - start_time) > 20:
                stable = True
                print vlt, mean, time.time() - start_time
                curr_out.append(mean)
                curr_out_err.append(np.std(curr_list[-4:]))
                volt_out.append(float(dut['Sourcemeter'].get_current()[:13]))
            else:
                time.sleep(0.1)
        if abs(mean) >= 5e-06:
            finished = True
        else:
            dut['Sourcemeter'].set_voltage(curr_volt - volt_step)

    return curr_out, curr_out_err, volt_out


if __name__ == "__main__":
    curr_list, curr_err, volt_list = measure()
    print curr_list
    print curr_err
    print volt_list

    with open("/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/iv_curve_data_chip1_test.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(izip(curr_list, curr_err, volt_list))

#     pdfName = "/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/iv_curve.pdf"
#     pp = PdfPages(pdfName)
#
#     fig = Figure()
#     _ = FigureCanvas(fig)
#     fig.clear()
#     ax = fig.add_subplot(111)
#     ax.errorbar(volt_list * -1, curr_list * 10**9, yerr=curr_err)
#     ax.set_xlabel("Voltage [V]")
#     ax.set_ylabel("Current [A]")
#     ax.set_yscale("log")
#
#     pp.savefig(fig)
#     pp.close()
    ax = plt.subplot(111)
    plt.errorbar([abs(x) for x in volt_list], [abs(x) for x in curr_list], xerr=0., yerr=curr_err, fmt='o', markersize=2.)
    ax.set_yscale("log")
    plt.grid(True)
#     plt.ylim(min(plot_min) * 1e9 - 1e-8, 30)
    plt.xlabel("Bias Voltage [V]")
    plt.ylabel("Current [A]")
#     plt.gcf().set_size_inches(plt.gcf().get_size_inches()[1] * 1.618, plt.gcf().get_size_inches()[1])
    plt.show()


#     dut = Dut('/home/daniel/MasterThesis/basil/examples/lab_devices/keithley2400_pyserial.yaml')
#     dut.init()
#     for V in np.arange(-100, 0, 1):
#         dut['Sourcemeter'].set_voltage(V)
#         print V
#         dut['Sourcemeter'].on()
#         time.sleep(2.5)
#         print dut['Sourcemeter'].get_current()
