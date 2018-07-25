#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import yaml
import basil
from basil.dut import Dut
import logging
logging.getLogger().setLevel(logging.DEBUG)
import os
import numpy as np
import time
import bitarray
from numba import njit

TRIGGER_ID = 0x80000000
TRG_MASK = 0x7FFFFFFF

TDC_ID = 0x40000000
# TDC_ID = 0x10000000000
TDC_MASK = 0xFFF
TDC_DELAY_MASK = 0x0FF00000


@njit
def _interpret_raw_data(data, pix_data):
    irec = 0
    prev_bcid = 0
    bcid = 0
    lv1id = 0

    col = 0
    row = 0
    rowp = 0
    totB = 0
    totT = 0

    for inx in range(data.shape[0]):
        if (data[inx] & 0x800000):
            bcid = data[inx] & 0x7fffff
            if(prev_bcid + 1 != bcid):
                lv1id = 0
            else:
                lv1id += 1
            prev_bcid = bcid
        else:
            col = (data[inx] & 0b111100000000000000000) >> 17
            row = (data[inx] & 0b11111100000000000) >> 11
            rowp = (data[inx] & 0b10000000000) >> 10
            totB = (data[inx] & 0b11110000) >> 4
            totT = (data[inx] & 0b1111)

            # print col, row, rowp, totT, totB

            #| rowp = 1 | rowp = 0 |
            #| totL_T   | totR_T   |
            #| totL_B   | totR_B   |

            if rowp == 1:
                if(totT != 15):
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2 + 1
                    else:
                        pix_data[irec].row = 63 - ((row % 32) * 2)
                    pix_data[irec].col = col * 4 + (row / 32) * 2
                    pix_data[irec].tot = totT
                    irec += 1

                if(totB != 15):
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2
                    else:
                        pix_data[irec].row = 63 - ((row % 32) * 2 + 1)
                    pix_data[irec].col = col * 4 + (row / 32) * 2
                    pix_data[irec].tot = totB
                    irec += 1
            else:
                if(totT != 15):
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2 + 1
                    else:
                        pix_data[irec].row = 63 - (row % 32) * 2
                    pix_data[irec].col = col * 4 + (row / 32) * 2 + 1
                    pix_data[irec].tot = totT
                    irec += 1

                if(totB != 15):
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2
                    else:
                        pix_data[irec].row = 63 - ((row % 32) * 2 + 1)

                    pix_data[irec].col = col * 4 + (row / 32) * 2 + 1
                    pix_data[irec].tot = totB
                    irec += 1

    return pix_data[:irec]


@njit
def _interpret_raw_data_w_tdc(data, pix_data):
    irec = np.uint32(0)
    prev_bcid = np.uint8(0)
    bcid = np.uint32(0)
    lv1id = np.uint8(0)
    tdc = np.uint16(0)
    delay = np.uint8(0)
    trigg_id = np.uint32(0)

    col = np.uint8(0)
    row = np.uint8(0)
    rowp = np.uint8(0)
    totB = np.uint8(0)
    totT = np.uint8(0)

    for inx in range(data.shape[0]):
        if (data[inx] & TRIGGER_ID):
            trigg_id = (data[inx] & TRG_MASK)

        elif (data[inx] & TDC_ID):
            tdc = (data[inx] & TDC_MASK)
            delay = (data[inx] & TDC_DELAY_MASK)

        elif (data[inx] & 0x800000):
            bcid = data[inx] & 0x7FFFFF
            if(prev_bcid + 1 != bcid):
                lv1id = 0
            else:
                lv1id += 1
            prev_bcid = bcid
        else:
            col = (data[inx] & 0b111100000000000000000) >> 17
            row = (data[inx] & 0b11111100000000000) >> 11
            rowp = (data[inx] & 0b10000000000) >> 10
            totB = (data[inx] & 0b11110000) >> 4
            totT = (data[inx] & 0b1111)

            # print col, row, rowp, totT, totB

            #| rowp = 1 | rowp = 0 |
            #| totL_T   | totR_T   |
            #| totL_B   | totR_B   |

            if rowp == 1:
                if(totT != 15):
                    pix_data[irec].trig_id = trigg_id
                    pix_data[irec].tdc = tdc
                    pix_data[irec].tdc_delay = delay
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2 + 1
                    else:
                        pix_data[irec].row = 63 - ((row % 32) * 2)
                    pix_data[irec].col = col * 4 + (row / 32) * 2
                    pix_data[irec].tot = totT
                    irec += 1

                if(totB != 15):
                    pix_data[irec].trig_id = trigg_id
                    pix_data[irec].tdc = tdc
                    pix_data[irec].tdc_delay = delay
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2
                    else:
                        pix_data[irec].row = 63 - ((row % 32) * 2 + 1)

                    pix_data[irec].col = col * 4 + (row / 32) * 2

                    pix_data[irec].tot = totB
                    irec += 1
            else:
                if(totT != 15):
                    pix_data[irec].trig_id = trigg_id
                    pix_data[irec].tdc = tdc
                    pix_data[irec].tdc_delay = delay
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2 + 1
                    else:
                        pix_data[irec].row = 63 - (row % 32) * 2
                    pix_data[irec].col = col * 4 + (row / 32) * 2 + 1
                    pix_data[irec].tot = totT
                    irec += 1

                if(totB != 15):
                    pix_data[irec].trig_id = trigg_id
                    pix_data[irec].tdc = tdc
                    pix_data[irec].tdc_delay = delay
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2
                    else:
                        pix_data[irec].row = 63 - ((row % 32) * 2 + 1)

                    pix_data[irec].col = col * 4 + (row / 32) * 2 + 1
                    pix_data[irec].tot = totB
                    irec += 1
    return pix_data[:irec]


@njit
def _interpret_raw_data_tlu(data, pix_data):
    irec = np.uint32(0)
    prev_bcid = np.uint8(0)
    bcid = np.uint32(0)
    lv1id = np.uint8(0)
    trigg_id = np.uint32(0)

    col = np.uint8(0)
    row = np.uint8(0)
    rowp = np.uint8(0)
    totB = np.uint8(0)
    totT = np.uint8(0)

    for inx in range(data.shape[0]):
        if (data[inx] & TRIGGER_ID):
            trigg_id = (data[inx] & TRG_MASK)

        elif (data[inx] & 0x800000):
            bcid = data[inx] & 0x7FFFFF
            if(prev_bcid + 1 != bcid):
                lv1id = 0
            else:
                lv1id += 1
            prev_bcid = bcid
        else:
            col = (data[inx] & 0b111100000000000000000) >> 17
            row = (data[inx] & 0b11111100000000000) >> 11
            rowp = (data[inx] & 0b10000000000) >> 10
            totB = (data[inx] & 0b11110000) >> 4
            totT = (data[inx] & 0b1111)

            # print col, row, rowp, totT, totB

            #| rowp = 1 | rowp = 0 |
            #| totL_T   | totR_T   |
            #| totL_B   | totR_B   |

            if rowp == 1:
                if(totT != 15):
                    pix_data[irec].trig_id = trigg_id
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2 + 1
                    else:
                        pix_data[irec].row = 63 - ((row % 32) * 2)
                    pix_data[irec].col = col * 4 + (row / 32) * 2
                    pix_data[irec].tot = totT
                    irec += 1

                if(totB != 15):
                    pix_data[irec].trig_id = trigg_id
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2
                    else:
                        pix_data[irec].row = 63 - ((row % 32) * 2 + 1)

                    pix_data[irec].col = col * 4 + (row / 32) * 2

                    pix_data[irec].tot = totB
                    irec += 1
            else:
                if(totT != 15):
                    pix_data[irec].trig_id = trigg_id
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2 + 1
                    else:
                        pix_data[irec].row = 63 - (row % 32) * 2
                    pix_data[irec].col = col * 4 + (row / 32) * 2 + 1
                    pix_data[irec].tot = totT
                    irec += 1

                if(totB != 15):
                    pix_data[irec].trig_id = trigg_id
                    pix_data[irec].bcid = bcid
                    pix_data[irec].lv1id = lv1id
                    if row < 32:
                        pix_data[irec].row = (row % 32) * 2
                    else:
                        pix_data[irec].row = 63 - ((row % 32) * 2 + 1)

                    pix_data[irec].col = col * 4 + (row / 32) * 2 + 1
                    pix_data[irec].tot = totB
                    irec += 1

    return pix_data[:irec]


class fe65p2(Dut):

    def __init__(self, conf=None):

        if conf == None:
            conf = os.path.dirname(os.path.abspath(
                __file__)) + os.sep + "fe65p2.yaml"

        logging.info("Loading configuration file from %s" % conf)

        conf = self._preprocess_conf(conf)

#         if socket_address and not context:
#             logging.info('Creating ZMQ context')
#             context = zmq.Context()
#
#         if socket_address and context:
#             logging.info('Creating socket connection to server %s', socket_address)
#             self.socket = context.socket(zmq.PUB)  # publisher socket
#             self.socket.bind(socket_address)
#             send_meta_data(self.socket, None, name='Reset')  # send reset to indicate a new scan
#         else:
#             self.socket = None

        super(fe65p2, self).__init__(conf)

    def init(self):
        super(fe65p2, self).init()

    def _preprocess_conf(self, conf):
        return conf

    def write_global(self):

        # size of global register
        self['global_conf'].set_size(145)

        # write + start
        self['global_conf'].write()

        # wait for finish
        while not self['global_conf'].is_ready:
            pass
        time.sleep(0.1)

    def mask_sr(self, mask):
        """
        63 -> 124    63(127) -> 126  |  63(191) -> 0(128)     63(255) -> 2(130)
        62 -> 125    62(126) -> 127  |  62(190) -> 1(129)     62(254) -> 3(131)
                                        61(189) -> 4(132)     61(253) -> 6(134)
        3 -> 4       3(67)   -> 6    |  60(187) -> 5(133)     60(252) -> 7(135)
        2 -> 5       2(66)   -> 7    |
        1 -> 0       1(65)   -> 2    |  1(129) -> 124(252)       1(193) -> 126(254)
        0 -> 1       0(64)   -> 3    |  0(128) -> 125(253)       0(192) -> 127(255)

        """
        conf_array_mcol = np.reshape(mask, (16, 64 * 4))
        mask = np.empty([16, 64 * 4], dtype=np.bool)

        for mcol in range(16):
            for i in range(256):
                if(i < 64):
                    o = (i - 1) * 2 + 3 * ((i + 1) %
                                           2)  # (i - 1) * 2 #o = 1 + i/2
                elif (i < 128):
                    o = (i - 64) * 2 + 3 * ((i - 64 + 1) % 2)
                elif (i < 192):
                    o = 125 - (((i - 128) / 2) * 4 + (i - 128) % 2) + 128
                else:
                    o = 127 - (((i - 192) / 2) * 4 + (i - 192) % 2) + 128

                mask[mcol][o] = conf_array_mcol[mcol][i]

        mask_1d = np.ravel(mask)
        lmask = mask_1d.tolist()
        bv_mask = bitarray.bitarray(lmask)
        return bv_mask

    def write_pixel(self, mask=None, ld=False):

        if mask is not None:
            mask_gen = self.mask_sr(mask)
            self['pixel_conf'][:] = mask_gen

        # pixels in multi_column
        self['pixel_conf'].set_size(16 * 4 * 64)

        # enable writing pixels
        self['control']['GATE_EN_PIX_SR'] = 1
        self['control'].write()

        self['pixel_conf'].write()

        while not self['pixel_conf'].is_ready:
            pass

        self['control']['GATE_EN_PIX_SR'] = 0
        self['control'].write()

        if(ld):
            self['control']['LD'] = 1
            self['control'].write()
            self['control']['LD'] = 0
            self['control'].write()

    def write_pixel_col(self, ld=False):

        # pixels in multi_column
        self['pixel_conf'].set_size(4 * 64)

        # enable writing pixels
        self['control']['GATE_EN_PIX_SR'] = 1
        self['control'].write()

        self['pixel_conf'].write(4 * 64 / 8)

        while not self['pixel_conf'].is_ready:
            pass

        self['control']['GATE_EN_PIX_SR'] = 0
        self['control'].write()

        if(ld):
            self['control']['LD'] = 1
            self['control'].write()
            self['control']['LD'] = 0
            self['control'].write()

    def write_hitor_mask(self, mask=np.full([64, 64], False, dtype=np.bool)):
        # if false -> 0b01
        # if true -> 0b11

        self.write_pixel(mask)
        self['global_conf']['PixConfLd'] = 0b01
        self.write_global()
        self['global_conf']['PixConfLd'] = 0b00

    def write_en_mask(self, mask):
        self.write_pixel(mask)
        self['global_conf']['PixConfLd'] = 0b10
        self.write_global()
        self['global_conf']['PixConfLd'] = 0b00

    def write_inj_mask(self, mask):
        self.write_pixel(mask)
        self['global_conf']['InjEnLd'] = 0b1
        self.write_global()
        self['global_conf']['InjEnLd'] = 0b0

    def write_tune_mask_2(self, mask):
        mask_in = np.copy(mask)
        mask_above_15 = np.full((64, 64), 0, dtype=int)
        mask_below_15 = np.full((64, 64), 0, dtype=int)
        mask_above_15[mask_in > 15] = mask_in[mask_in > 15]
        mask_below_15[mask_in <= 15] = mask_in[mask_in <= 15]
        mask_above_15_2 = (mask_above_15 - 31) * -1
        mask_above_15_2[mask_above_15_2 == 31] = 0

        mask_out = mask_below_15 + mask_above_15_2
        bits_out = np.unpackbits(mask_out.astype(np.uint8))
        mask_bits_array = np.reshape(bits_out, (64, 64, 8))

        print mask_bits_array.shape
#         print mask_bits_array

        bits_out_list = [bin(int(x)) for x in mask_out.reshape(4096)]
        bin_full = np.asarray(bits_out_list)
        np.reshape(bin_full, (64, 64))
        print bin_full
        print bin_full.shape

        bits_out = np.full((64, 64), 0b0)
        # this makes a list with the order but it does not
        bin_above_15 = [bin(int(x)) for x in mask_above_15_2.reshape(4096)]
        bin_above_15 = np.asarray(bin_above_15)
#         print bin_above_15.shape
#         print bin_above_15
#         bin_below_15 = [bin(int(x)) for x in mask_below_15.reshape(4096)]

#         bits_out = bin_below_15 + bin_above_15
#         bits_out_arr = np.asarray(bits_out)
#         np.reshape(bits_out_arr, (64, 64))
#         print bits_out_arr
        self.write_pixel(mask_in)
        self['global_conf']['TDacLd'] = bin_full
        self.write_global()
        self['global_conf']['TDacLd'] = 0b1111

        sign_bit_out = [1 for x in bin_above_15 if x != 0b0]
        self.write_pixel(sign_bit_out)
        self['global_conf']['SignLd'] = 1
        self.write_global()
        self['global_conf']['SignLd'] = 0

    def write_tune_mask(self, mask):
            # 0  -> Sign = 1, TDac = 15 1111(lowest)
            # ...
            # 15 -> Sign = 1, TDac = 0  0000
            # 16 -> Sign = 0, TDac = 0  0000
            # ...
            # 31 -> Sign = 0, TDac = 15 1111

        mask_out = np.copy(mask)
        mask_bits = np.unpackbits(mask_out)
        mask_bits_array = np.reshape(mask_bits, (64, 64, 8))
        mask_out[mask_bits_array[:, :, 3] == 0] = 15 - \
            mask_out[mask_bits_array[:, :, 3] == 0]  # 15
        # investigate here how to set 0 to 0
        mask_bits = np.unpackbits(mask_out)
        mask_bits_array = np.reshape(mask_bits, (64, 64, 8)).astype(np.bool)
        mask_bits_array[:, :, 3] = ~mask_bits_array[:, :, 3]

        for bit in range(4):
            mask_bits_sel = mask_bits_array[:, :, 7 - bit]
            self.write_pixel(mask_bits_sel)
            self['global_conf']['TDacLd'][bit] = 1
            self.write_global()
            self['global_conf']['TDacLd'][bit] = 0

        mask_bits_sel = mask_bits_array[:, :, 3]
        self.write_pixel(mask_bits_sel)
        self['global_conf']['SignLd'] = 1
        self.write_global()
        self['global_conf']['SignLd'] = 0

    def interpret_raw_data(self, raw_data, meta_data=[]):
        data_type = {'names': ['bcid', 'col', 'row', 'tot', 'lv1id', 'scan_param_id'],
                     'formats': ['uint32', 'uint8', 'uint8', 'uint8', 'uint8', 'uint16']}
        ret = []
        if len(meta_data):
            param, index = np.unique(meta_data['scan_param_id'], return_index=True)
            index = index[1:]
            index = np.append(index, meta_data.shape[0])
            index = index - 1
            stops = meta_data['index_stop'][index]
            split = np.split(raw_data, stops)
            for i in range(len(split[:-1])):
                # print param[i], stops[i], len(split[i]), split[i]
                int_pix_data = self.interpret_raw_data(split[i])
                int_pix_data['scan_param_id'][:] = param[i]
                if len(ret):
                    ret = np.hstack((ret, int_pix_data))
                else:
                    ret = int_pix_data
        else:
            pix_data = np.recarray((raw_data.shape[0] * 2), dtype=data_type)
            ret = _interpret_raw_data(raw_data, pix_data)
        return ret

    def interpret_raw_data_w_tdc(self, raw_data, meta_data=[]):
        data_type = {'names': ['bcid', 'col', 'row', 'tot', 'lv1id', 'scan_param_id', 'tdc', 'tdc_delay', 'trig_id'],
                     'formats': ['uint32', 'uint8', 'uint8', 'uint8', 'uint8', 'uint16', 'uint16', 'uint8', 'uint32']}
        ret = []
        if len(meta_data):
            param, index = np.unique(meta_data['scan_param_id'], return_index=True)
            index = index[1:]
            index = np.append(index, meta_data.shape[0])
            index = index - 1
            starts = []
            stops = []
            for i in np.unique(meta_data['scan_param_id']):
                starts.append(meta_data[meta_data['scan_param_id'] == i]['index_start'][0])
                stops.append(meta_data[meta_data['scan_param_id'] == i]['index_stop'][-1])
            split = []
            for x, y in zip(starts, stops):
                split.append(raw_data[x:y])
            split.append(raw_data[stops[-1]:])

            for i in range(len(split[:-1])):
                # print param[i], stops[i], len(split[i]), split[i]
                int_pix_data = self.interpret_raw_data_w_tdc(split[i])
#                 print int_pix_data['tdc']
                int_pix_data['scan_param_id'][:] = param[i]
                if len(ret):
                    ret = np.hstack((ret, int_pix_data))
                else:
                    ret = int_pix_data

        else:
            pix_data = np.recarray((raw_data.shape[0] * 2), dtype=data_type)
            ret = _interpret_raw_data_w_tdc(raw_data, pix_data)
#             print "ret", ret['tdc']
        return ret

    def interpret_raw_data_tlu(self, raw_data, meta_data=[]):
        data_type = {'names': ['bcid', 'col', 'row', 'tot', 'lv1id', 'scan_param_id', 'trig_id'],
                     'formats': ['uint32', 'uint8', 'uint8', 'uint8', 'uint8', 'uint16', 'uint32']}
        ret = []
        if len(meta_data):
            param, index = np.unique(meta_data['scan_param_id'], return_index=True)
            index = index[1:]
            index = np.append(index, meta_data.shape[0])
            index = index - 1
            stops = meta_data['index_stop'][index]
            split = np.split(raw_data, stops)

            for i in range(len(split[:-1])):
                # print param[i], stops[i], len(split[i]), split[i]
                int_pix_data = self.interpret_raw_data_tlu(split[i])
                int_pix_data['scan_param_id'][:] = param[i]
                if len(ret):
                    ret = np.hstack((ret, int_pix_data))
                else:
                    ret = int_pix_data
        else:
            pix_data = np.recarray((raw_data.shape[0] * 2), dtype=data_type)
            ret = _interpret_raw_data_tlu(raw_data, pix_data)

        return ret

    def power_up(self):

        self['VDDA'].set_current_limit(200, unit='mA')
        self['VDDA'].set_voltage(1.2, unit='V')
        self['VDDA'].set_enable(True)

        self['VDDD'].set_voltage(1.2, unit='V')
        self['VDDD'].set_enable(True)

        self['VAUX'].set_voltage(1.2, unit='V')
        self['VAUX'].set_enable(True)

    def power_down(self):

        self['VDDA'].set_enable(False)
        self['VDDD'].set_enable(False)
        self['VAUX'].set_enable(False)

    def power_status(self):
        staus = {}

        staus['VDDD[V]'] = self['VDDD'].get_voltage(unit='V')
        staus['VDDD[mA]'] = self['VDDD'].get_current(unit='mA')
        staus['VDDA[V]'] = self['VDDA'].get_voltage(unit='V')
        staus['VDDA[mA]'] = self['VDDA'].get_current(unit='mA')
        staus['VAUX[V]'] = self['VAUX'].get_voltage(unit='V')
        staus['VAUX[mA]'] = self['VAUX'].get_current(unit='mA')

        return staus

    def dac_status(self):
        staus = {}

        dac_names = ['PrmpVbpDac', 'vthin1Dac', 'vthin2Dac', 'vffDac',
                     'PrmpVbnFolDac', 'vbnLccDac', 'compVbnDac', 'preCompVbnDac']
        for dac in dac_names:
            staus[dac] = int(str(self['global_conf'][dac]), 2)

        return staus

    def set_for_configuration(self):
        self['global_conf']['vthin1Dac'] = 255
        self['global_conf']['vthin2Dac'] = 0
        self['global_conf']['preCompVbnDac'] = 50
        self['global_conf']['PrmpVbpDac'] = 80
        self.write_global()
        time.sleep(0.2)

    def scan_setup(self):
        self['control']['RESET'] = 0b01
        self['control']['DISABLE_LD'] = 0
        self['control']['PIX_D_CONF'] = 0
        self['control'].write()

        self['control']['CLK_OUT_GATE'] = 1
        self['control']['CLK_BX_GATE'] = 1
        self['control'].write()
        time.sleep(0.1)

        self['control']['RESET'] = 0b11
        self['control'].write()

        self.dut.start_up()

        self['global_conf']['OneSr'] = 1

        self['global_conf']['TestHit'] = 0
        self['global_conf']['SignLd'] = 0
        self['global_conf']['InjEnLd'] = 0
        self['global_conf']['TDacLd'] = 0
        self['global_conf']['PixConfLd'] = 0
        self.write_global()

    def start_up(self):
        self['control']['RESET'] = 0b01
        self['control']['DISABLE_LD'] = 0
        self['control'].write()

        self['control']['CLK_OUT_GATE'] = 1
        self['control']['CLK_BX_GATE'] = 1
        self['control'].write()
        time.sleep(0.1)

        self['control']['RESET'] = 0b11
        self['control'].write()

    def scan_loop(self, scanType="none", repeat_command=100, use_delay=True, additional_delay=0, mask_steps=4,
                  enable_mask_steps=None, same_mask_for_all_qc=False, bol_function=None, eol_function=None, digital_injection=False,
                  enable_shift_masks=None, disable_shift_masks=None,
                  restore_shift_masks=True, mask=None):
        '''Parameters:
        scanType (string):                  switch for the desired scan
            options -> noise_tune, analog, threshold, timewalk, digital, schmoo
        repeat_command (bool):              number of reps for a command per mask step
        use_delay (bool):                   add additional delay to the command
        additional_delay (int):             additional delay to inc the command-to-command delay [clock cycles / ### ns] TODO: number???
        mask_steps (int):                   number of mask steps (have found 4 to work well for all scans as of 22/11/17)
        enable_mask_steps (list,tuple):     list of mask steps to be applied, default is all. From 0 to (mask-1). A value equal None or empty list will select all mask steps.
        same_mask_for_all_qc (bool):        use same mask for all quad columns. only effects shift masks (enable_shift_masks). Enabling is a good call since all quad columns will have the same configuration and the scan speed can increased by an order of magnitude.
        bol_function (function):            beginning of loop function called each time before sending a command. Argument is a function pointer (without braces) or functor. TODO: make prewrite function to reset the chip to high threshold values
        eol_function (function):            end of loop function called each time after sending a command. Argument is a function pointer (without braces) or functor.
        digitial_injection (bool):          enables digital injection TODO: what is disabled/enabled
        enable_shift_masks (list,tuple):    list of enabled pixel masks which will be shifted during scan, mask set to 1 for selected pixels else 0. None will select TODO: which will be selected here
        disable_shift_masks (list,tuple):   list of disabled pixel makss which will be shifted during scna, mask set to 0 for selected pixels, else 0. None will disable no maks TODO: ?????
        restore_shift_masks (bool):         writing the initial (restored) pixel config into FE after finishing the scan loop
        mask (array-like):                  additional mask. must be convertible to an array of bools with the same shape as mask array, true indictes a masked pixel, masked pixels will be diabled during shifting of the enable shift masks, and enabled during shifting diable shift masks
        '''


if __name__ == "__main__":
    chip = fe65p2()
    chip.init()
    chip.power_up()
