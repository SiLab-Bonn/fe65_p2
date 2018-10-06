from basil.dut import Dut
import time
import numpy as np


def ramp_up(vlt=None):
    dut = Dut('/home/daniel/MasterThesis/basil/examples/lab_devices/keithley2460_pyvisa.yaml')
    dut.init()
    print dut['Sourcemeter'].get_name()
#     dut['Sourcemeter'].set_current_limit(10**-6)
    if vlt:
        upper = vlt
    else:
        upper = -100
#     print float(dut['Sourcemeter'].get_current()[:13])
#     for V in np.arange(float(dut['Sourcemeter'].get_current()[:13]), -81, -2):
#         dut['Sourcemeter'].set_voltage(V)
#         time.sleep(2.)
#         print dut['Sourcemeter'].get_current()
    for V in np.arange(float(dut['Sourcemeter'].get_voltage()), -81, -2):
        dut['Sourcemeter'].set_voltage(V)
        time.sleep(2.)
        print "V:", dut['Sourcemeter'].get_voltage(), "I:", dut['Sourcemeter'].get_current()


if __name__ == "__main__":
    ramp_up()
