from basil.dut import Dut
import time
import numpy as np

dut = Dut('/home/daniel/MasterThesis/basil/examples/lab_devices/keithley2400_pyserial.yaml')
dut.init()
print dut['Sourcemeter'].get_name()
dut['Sourcemeter'].set_current_limit(10**-6)
for V in np.arange(float(dut['Sourcemeter'].get_current()[:13]), 0, 2):
    dut['Sourcemeter'].set_voltage(V)
    print V
    dut['Sourcemeter'].on()
    time.sleep(1.5)
    print dut['Sourcemeter'].get_current()

# dut['Sourcemeter'].off()
