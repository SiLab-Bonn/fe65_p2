import basil
from basil.dut import Dut
import logging
import time
import os

class power(object):

    def switch_on(self):
        try:
            pow = Dut(str(os.path.dirname(os.path.dirname(basil.__file__))) + '/examples/lab_devices/ttiql335tp_pyserial2.yaml')
            pow.init()
            logging.info('Connected to ' + str(pow['Power'].get_info()))
            pow['Power'].turn_on_ch1()

        except RuntimeError:
            logging.info('ERROR: connection to power supply failed')

    def switch_off(self):
        try:
            pow = Dut(str(os.path.dirname(os.path.dirname(basil.__file__))) + '/examples/lab_devices/ttiql335tp_pyserial2.yaml')
            pow.init()
            logging.info('Connected to ' + str(pow['Power'].get_info()))
            pow['Power'].turn_off_ch1()

        except RuntimeError:
            logging.info('ERROR: connection to power supply failed')

    def restart(self):
        try:
            logging.info('Trying to reach Power Supply...')
            pow = Dut(str(os.path.dirname(os.path.dirname(basil.__file__)))+'/examples/lab_devices/ttiql335tp_pyserial2.yaml')
            pow.init()
            logging.info('Connected to ' + str(pow['Power'].get_info()))
            logging.info('Powering off ch 1. Restart in 3 seconds')
            pow['Power'].turn_off_ch1()
            time.sleep(3)
            logging.info('Powering on ch1. Then holding on 10 seconds.')
            pow['Power'].turn_on_ch1()
            time.sleep(10)
        except RuntimeError:
            logging.info('ERROR: connection to power supply failed')


if __name__ == "__main__":
    supply = power()
    supply.switch_off()