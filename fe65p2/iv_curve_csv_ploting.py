#===============================================================================
#
# Plotting code for iv curves from csv file
# meas curr, err, voltage
#
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import logging
from h5py.h5a import delete
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import tables as tb
import yaml
import matplotlib as mpl
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colors, cm
from scipy.optimize import curve_fit
from math import ceil
import csv
import os


def read_file(filename_csv):
    # may need the 'rb' option to read the file in binary
    with open('%s' % filename_csv, 'rb') as in_file_csv:
        reader = csv.reader(in_file_csv, delimiter=',')
        meas_curr = []
        curr_err = []
        meas_volt = []
        for row in reader:
            meas_curr = np.append(meas_curr, float(row[0]))
            curr_err = np.append(curr_err, float(row[1]))
            meas_volt = np.append(meas_volt, float(row[2]))
        in_file_csv.close()
    return meas_curr, curr_err, meas_volt


def combine_and_plot():
    csv_name = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/iv_curve_data.csv'
    plot_min = []
    ax = plt.subplot(111)
    # for i in [1, 3, 4, 5]:
    for i in [2]:
        csv_name = '/home/daniel/MasterThesis/fe65_p2/fe65p2/scans/output_data/iv_curves/good_iv/iv_curve_data_chip' + str(i) + '.csv'
        meas_curr, curr_err, meas_volt = read_file(csv_name)

        # want to read in all of the files here
        plt.errorbar(abs(meas_volt), abs(meas_curr), xerr=0., yerr=curr_err, fmt='o', markersize=2.5, label='Chip' + str(i))
    ax.set_yscale("log")
    plt.grid(True)
#     plt.ylim(min(plot_min) * 1e9 - 1e-8, 30)
    plt.xlabel("Bias Voltage [V]")
    plt.ylabel("Current [A]")
    plt.legend()
#     plt.gcf().set_size_inches(plt.gcf().get_size_inches()[1] * 1.618, plt.gcf().get_size_inches()[1])
    plt.show()


if __name__ == "__main__":
    combine_and_plot()
