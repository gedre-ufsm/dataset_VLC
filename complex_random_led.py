#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: complex_random_led
# GNU Radio version: 3.10.1.1

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
import sip
from gnuradio import blocks
import numpy
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
import complex_random_led_epy_block_0 as epy_block_0  # embedded python block

from scipy.io import savemat

import os

from gnuradio import qtgui

class complex_random_led(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "complex_random_led", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("complex_random_led")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "complex_random_led")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.vector_size = vector_size = 1024
        self.tuning = tuning = 10e6
        self.samp_rate = samp_rate = 500e3
        self.rf_gain = rf_gain = 76

        ##################################################
        # Blocks
        ##################################################
        self._tuning_range = Range(1e6, 50e6, 100e3, 10e6, 200)
        self._tuning_win = RangeWidget(self._tuning_range, self.set_tuning, "Frequency", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tuning_win)
        self._rf_gain_range = Range(0, 76, 1, 76, 200)
        self._rf_gain_win = RangeWidget(self._rf_gain_range, self.set_rf_gain, "RF Gain", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._rf_gain_win)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("addr0=192.168.136.136", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=[0],
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        _last_pps_time = self.uhd_usrp_source_0.get_time_last_pps().get_real_secs()
        # Poll get_time_last_pps() every 50 ms until a change is seen
        while(self.uhd_usrp_source_0.get_time_last_pps().get_real_secs() == _last_pps_time):
            time.sleep(0.05)
        # Set the time to PC time on next PPS
        self.uhd_usrp_source_0.set_time_next_pps(uhd.time_spec(int(time.time()) + 1.0))
        # Sleep 1 second to ensure next PPS has come
        time.sleep(1)

        self.uhd_usrp_source_0.set_center_freq(tuning, 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0.set_gain(rf_gain, 0)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("addr0=192.168.136.139", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=[0],
            ),
            "",
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        _last_pps_time = self.uhd_usrp_sink_0.get_time_last_pps().get_real_secs()
        # Poll get_time_last_pps() every 50 ms until a change is seen
        while(self.uhd_usrp_sink_0.get_time_last_pps().get_real_secs() == _last_pps_time):
            time.sleep(0.05)
        # Set the time to PC time on next PPS
        self.uhd_usrp_sink_0.set_time_next_pps(uhd.time_spec(int(time.time()) + 1.0))
        # Sleep 1 second to ensure next PPS has come
        time.sleep(1)

        self.uhd_usrp_sink_0.set_center_freq(tuning, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(rf_gain, 0)
        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            vector_size,
            0,
            1.0,
            "x-Axis",
            "y-Axis",
            "",
            1, # Number of inputs
            None # parent
        )
        self.qtgui_vector_sink_f_0.set_update_time(0.10)
        self.qtgui_vector_sink_f_0.set_y_axis(-140, 10)
        self.qtgui_vector_sink_f_0.enable_autoscale(True)
        self.qtgui_vector_sink_f_0.enable_grid(True)
        self.qtgui_vector_sink_f_0.set_x_axis_units("")
        self.qtgui_vector_sink_f_0.set_y_axis_units("")
        self.qtgui_vector_sink_f_0.set_ref_level(0)

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_vector_sink_f_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_vector_sink_f_0.set_line_label(i, labels[i])
            self.qtgui_vector_sink_f_0.set_line_width(i, widths[i])
            self.qtgui_vector_sink_f_0.set_line_color(i, colors[i])
            self.qtgui_vector_sink_f_0.set_line_alpha(i, alphas[i])

        self._qtgui_vector_sink_f_0_win = sip.wrapinstance(self.qtgui_vector_sink_f_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_vector_sink_f_0_win)
        self.qtgui_number_sink_0 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_NONE,
            1,
            None # parent
        )
        self.qtgui_number_sink_0.set_update_time(0.10)
        self.qtgui_number_sink_0.set_title("Maximo")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_0.set_min(i, -1)
            self.qtgui_number_sink_0.set_max(i, 1)
            self.qtgui_number_sink_0.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_0.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_0.set_label(i, labels[i])
            self.qtgui_number_sink_0.set_unit(i, units[i])
            self.qtgui_number_sink_0.set_factor(i, factor[i])

        self.qtgui_number_sink_0.enable_autoscale(False)
        self._qtgui_number_sink_0_win = sip.wrapinstance(self.qtgui_number_sink_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_0_win)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
            1024, #size
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0.enable_grid(False)
        self.qtgui_const_sink_x_0_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_0_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "enviado", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_win)
        self.fft_vxx_0_1 = fft.fft_vcc(vector_size, False, window.blackmanharris(vector_size), True, 1)
        self.fft_vxx_0_0 = fft.fft_vcc(vector_size, True, window.blackmanharris(vector_size), True, 1)
        self.fft_vxx_0 = fft.fft_vcc(vector_size, True, window.blackmanharris(vector_size), True, 1)
        self.epy_block_0 = epy_block_0.blk(vectorSize=vector_size)
        self.epy_block_0.set_block_alias("vector_size=vector_size")
        self.blocks_stream_to_vector_0_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vector_size)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vector_size)
        self.blocks_multiply_conjugate_cc_0 = blocks.multiply_conjugate_cc(vector_size)
        self.blocks_interleaved_short_to_complex_0 = blocks.interleaved_short_to_complex(False, False,32768)
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/workstation/Documentos/Rodrigo/GNU_UTEC/complex_random_led (GUI)/recebido.mat', False)
        self.blocks_file_sink_0_0.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/workstation/Documentos/Rodrigo/GNU_UTEC/complex_random_led (GUI)/enviado.mat', False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_complex_to_mag_0_0 = blocks.complex_to_mag(vector_size)
        self.analog_random_source_x_1 = blocks.vector_source_s(list(map(int, numpy.random.randint(-32768, 32768, 500000))), False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_1, 0), (self.blocks_interleaved_short_to_complex_0, 0))
        self.connect((self.blocks_complex_to_mag_0_0, 0), (self.epy_block_0, 0))
        self.connect((self.blocks_complex_to_mag_0_0, 0), (self.qtgui_vector_sink_f_0, 0))
        self.connect((self.blocks_interleaved_short_to_complex_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_interleaved_short_to_complex_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_interleaved_short_to_complex_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.blocks_interleaved_short_to_complex_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self.fft_vxx_0_1, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.blocks_stream_to_vector_0_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.epy_block_0, 0), (self.qtgui_number_sink_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_multiply_conjugate_cc_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.blocks_multiply_conjugate_cc_0, 1))
        self.connect((self.fft_vxx_0_1, 0), (self.blocks_complex_to_mag_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_file_sink_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_stream_to_vector_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.qtgui_const_sink_x_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "complex_random_led")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_vector_size(self):
        return self.vector_size

    def set_vector_size(self, vector_size):
        self.vector_size = vector_size

    def get_tuning(self):
        return self.tuning

    def set_tuning(self, tuning):
        self.tuning = tuning
        self.uhd_usrp_sink_0.set_center_freq(self.tuning, 0)
        self.uhd_usrp_sink_0.set_center_freq(self.tuning, 1)
        self.uhd_usrp_source_0.set_center_freq(self.tuning, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_bandwidth(self.samp_rate, 1)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rf_gain(self):
        return self.rf_gain

    def set_rf_gain(self, rf_gain):
        self.rf_gain = rf_gain
        self.uhd_usrp_sink_0.set_gain(self.rf_gain, 0)
        self.uhd_usrp_sink_0.set_gain(self.rf_gain, 1)
        self.uhd_usrp_source_0.set_gain(self.rf_gain, 0)




def main(top_block_cls=complex_random_led, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    auto_close_timer = QtCore.QTimer()
    auto_close_duration = 5000  # Duration in milliseconds (5 seconds)
    auto_close_timer.singleShot(auto_close_duration, qapp.quit)

    # Start the application event loop
    qapp.exec_()

if __name__ == '__main__':
    main()
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Read I/Q data from files
with open('enviado.mat', 'rb') as f:
    x = np.fromfile(f, dtype=np.float32)

with open('recebido.mat', 'rb') as f:
    y = np.fromfile(f, dtype=np.float32)
    # Trim 'y' to match the length of 'x'
    
y_trimmed = y[:len(x)]

# Perform cross-correlation to find the delay between 'x' and 'y'
c = correlate(y_trimmed, x, mode='full')
lag = np.arange(-len(y_trimmed) + 1, len(x))
I = np.argmax(c)
maxC = c[I]
lagAtMaxC = lag[I]

# Align 'y' with 'x' by suppressing initial values according to the delay
y_aligned = np.concatenate([y_trimmed[lagAtMaxC:], np.zeros(lagAtMaxC)])

# Correlation check (lagAtMaxC2 must be '0')
c2 = correlate(y_aligned, x, mode='full')
I2 = np.argmax(c2)
maxC2 = c2[I2]
lagAtMaxC2 = lag[I2]
print(lagAtMaxC2)

# Scale adjustment to match the amplitude of 'x'
scale_factor = np.max(np.abs(x)) / np.max(np.abs(y_aligned))
y_aligned_scaled = y_aligned * scale_factor

# Normalize the signals to a range of -1 to 1
x_normalized = 2 * ((x - np.min(x)) / (np.max(x) - np.min(x))) - 1
y_normalized = 2 * ((y_aligned_scaled - np.min(y_aligned_scaled)) / (np.max(y_aligned_scaled) - np.min(y_aligned_scaled))) - 1

# Separate I and Q components for the normalized signals
I_y = y_normalized[::2]
Q_y = y_normalized[1::2]
IQ_y = np.column_stack((I_y, Q_y))

I_x = x_normalized[::2]
Q_x = x_normalized[1::2]
IQ_x = np.column_stack((I_x, Q_x))

# Plot and save the constellation diagram for received data
max_val_y = np.max(np.abs(IQ_y))
plt.figure(figsize=(14, 12))
plt.scatter(IQ_y[:, 0], IQ_y[:, 1], color='b', s=0.1)
plt.title('Constellation Diagram Received')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.axis('equal')
plt.xlim([-max_val_y-0.1, max_val_y+0.1])
plt.ylim([-max_val_y-0.1, max_val_y+0.1])
plt.savefig('constellation_received.png')
plt.close()

# Plot and save the constellation diagram for sent data
max_val_x = np.max(np.abs(IQ_x))
plt.figure(figsize=(14, 12))
plt.scatter(IQ_x[:, 0], IQ_x[:, 1], color='b', s=0.1)
plt.title('Constellation Diagram Sent')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.axis('equal')
plt.xlim([-max_val_x-0.1, max_val_x+0.1])
plt.ylim([-max_val_x-0.1, max_val_x+0.1])
plt.savefig('constellation_sent.png')
plt.close()

# Create a new directory for the files
directory = "IQ_data"
if not os.path.exists(directory):
    os.makedirs(directory)

# Function to save data in both .npy and .mat formats
def save_data(data, filename):
    np.save(os.path.join(directory, filename + '.npy'), data)
   # savemat(os.path.join(directory, filename + '.mat'), {filename: data})

# 1. Combine I and Q into complex-valued arrays
IQ_x_complex = I_x + 1j * Q_x
IQ_y_complex = I_y + 1j * Q_y
save_data(IQ_x_complex, 'sent_data_complex')
save_data(IQ_y_complex, 'received_data_complex')

# 2. Save as a tuple
save_data(IQ_x, 'sent_data_tuple')
save_data(IQ_y, 'received_data_tuple')

# 3. Interleave I and Q components
IQ_x_interleaved = np.empty((I_x.size + Q_x.size,), dtype=I_x.dtype)
IQ_x_interleaved[0::2] = I_x
IQ_x_interleaved[1::2] = Q_x

IQ_y_interleaved = np.empty((I_y.size + Q_y.size,), dtype=I_y.dtype)
IQ_y_interleaved[0::2] = I_y
IQ_y_interleaved[1::2] = Q_y

save_data(IQ_x_interleaved, 'sent_data_interleaved')
save_data(IQ_y_interleaved, 'received_data_interleaved')
