from basil.dut import Dut

dut = Dut('/home/daniel/MasterThesis/basil/examples/lab_devices/tti_ql355tp_pyserial.yaml')
dut.init()
print dut['THURLBY-THANDAR,QL355TP,D,1.6'].get_info()
dut['THURLBY-THANDAR,QL355TP,D,1.6'].power_cycle(channel=3)