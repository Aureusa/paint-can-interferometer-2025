#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.9.2

# Standard library imports
import os
import sys
import json
import yaml
import signal

# Third-party imports
from PyQt5 import Qt

# GNU Radio imports
from gnuradio import gr
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import fft
from gnuradio import analog
from gnuradio.fft import window
import osmosdr

# Local imports
from utils import print_box

# File dir
file_dir = os.path.dirname(os.path.abspath(__file__))

# Open Airspy info JSON
airspy_info_folder = os.path.join(file_dir, "airspy_info.json")
with open(airspy_info_folder, 'r') as f:
    AIRSPY_INFO = json.load(f)

# Open observation config file YAML
config_folder = os.path.join(file_dir, "observation_conf.yaml")
with open(config_folder, 'r') as f:
    OBSERVATION_CONFIG = yaml.safe_load(f)

# Extract device list from config
DEVICE_LIST = OBSERVATION_CONFIG['device_list']

# Extract airspy parameters from config
SAMPLING_RATE = int(OBSERVATION_CONFIG['sampling_rate'])
CENTER_FREQUENCY = int(OBSERVATION_CONFIG['center_frequency'])
FFT_SIZE = int(OBSERVATION_CONFIG['fft_size'])
INTEGRATION_TIME = int(OBSERVATION_CONFIG['integration_time'])

# Obs duration
OBSERVATION_DURATION = int(OBSERVATION_CONFIG['observation_duration'])

# Data storage path
DATA_STORAGE_PATH = OBSERVATION_CONFIG['data_storage_path']

    
class Interferometer(gr.top_block, Qt.QWidget):
    """
    Interferometer schema for processing visibility data from multiple antennas.
    The scheme includes:
    - Airspy SDR sources for each antenna
    - FFT blocks to convert time-domain data to frequency domain
    - Visibility correlator to compute visibilities between antenna pairs and integrate over time

    The flow is as follows:
    Airspy Source ->
    Stream to Vector ->
    FFT ->
    Visibility Correlator and Integrator ->
    File Sink    
    """
    def __init__(
            self,
            sampling_rate=10e6,
            integration_time=1,
            frequency=1.42e9,
            fft_size=1024,
            num_antennas=9,
            folder_path="mock_data",
            test=False
        ):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
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

        self.settings = Qt.QSettings("GNU Radio", "Interferometer")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.sampling_rate = sampling_rate
        self.integration_time = integration_time
        self.frequency = frequency
        self.fft_size = fft_size
        self.num_antennas = num_antennas

        ##################################################
        # Blocks
        ##################################################

        # Create multiple Airspy devices
        # All Airspy devices configured identically assuming airspy=0, airspy=1, ..., airspy=8
        # TODO: This most likely needs to change based on actual device naming conventions
        if not test:
            self.airspy_devices = [
                self.def_airspy_device(device, sampling_rate, frequency)
                for device in DEVICE_LIST
            ]
        # ##################################################################
        # Define gaussian noise sources as placeholders for Airspy devices #
        ####################################################################
        if test:
        # ️               This is for testing purposes only
            self.num_antennas = 9
            self.airspy_devices = [
                analog.noise_source_c(analog.GR_GAUSSIAN, 0.1, seed=i*42)
                for i in range(self.num_antennas)
            ]
            # Add throttle blocks to control sample rate
            self.throttle_blocks = [
                blocks.throttle(gr.sizeof_gr_complex*1, self.sampling_rate, True)
                for _ in range(self.num_antennas)
            ]
        ###################################################################
        ###################################################################
        ###################################################################

        # Create FFT blocks
        self.fft_blocks = [
            fft.fft_vcc(fft_size, True, window.blackmanharris(fft_size), True, 1)
            for _ in range(self.num_antennas)
        ]

        # Create stream to vector blocks
        self.stream_to_vector_blocks = [
            blocks.stream_to_vector(gr.sizeof_gr_complex*1, fft_size)
            for _ in range(self.num_antennas)
        ]

        self.file_sinks = [
            blocks.file_sink(
                gr.sizeof_gr_complex*self.fft_size,
                os.path.join(folder_path, f'antenna_{device.split("=")[1]}.dat'),
                False
            )
            for device in DEVICE_LIST
        ]

        ##################################################
        # Connections
        ##################################################

        # Connect each airspy -> stream_to_vector -> fft -> filesink
        for i in range(self.num_antennas):
            self.connect(
                (self.airspy_devices[i], 0),
                (self.stream_to_vector_blocks[i], 0)
            )
            self.connect(
                (self.stream_to_vector_blocks[i], 0),
                (self.fft_blocks[i], 0)
            )
            self.connect(
                (self.fft_blocks[i], 0),
                (self.file_sinks[i], 0)
            )

        self._print_interferometer_info()

    def _print_interferometer_info(self):
        info = "Interferometer Configuration:"
        info += f"\n  Number of Antennas: {self.num_antennas}"
        info += f"\n  Sampling Rate: {self.sampling_rate/1e6} MHz"
        info += f"\n  Frequency: {self.frequency/1e6} MHz"
        info += f"\n  FFT Size: {self.fft_size}"
        print_box(info)

    def def_airspy_device(self, device, sampling_rate, frequency):
        """Define and configure an Airspy source block"""
        serial_nr = AIRSPY_INFO[device]['serial_nr']
        device = f"airspy={serial_nr}"

        osmosdr_source = osmosdr.source(
            args="numchan=" + str(1) + " " + device
        )
        osmosdr_source.set_sample_rate(sampling_rate)
        osmosdr_source.set_center_freq(frequency, 0)
        osmosdr_source.set_freq_corr(0, 0)
        osmosdr_source.set_dc_offset_mode(0, 0)
        osmosdr_source.set_iq_balance_mode(0, 0)
        osmosdr_source.set_gain_mode(False, 0)
        osmosdr_source.set_gain(21, 0)
        osmosdr_source.set_if_gain(0, 0)
        osmosdr_source.set_bb_gain(0, 0)
        osmosdr_source.set_antenna('', 0)
        osmosdr_source.set_bandwidth(0, 0)
        return osmosdr_source

    def connect_airspy_to_fft(self, airspy_devices, stream_to_vec_blocks, fft_blocks, test=False):
        ##########################################################
        # TODO: WHEN USING ACTUAL AIRSPY DEVICES, UNCOMMENT THIS #
        ##########################################################
        if not test:
            for i in range(self.num_antennas):
                self.connect(
                    (airspy_devices[i], 0),
                    (stream_to_vec_blocks[i], 0)
                )
                self.connect(
                    (stream_to_vec_blocks[i], 0),
                    (fft_blocks[i], 0)
                )
                self.connect(
                    (fft_blocks[i], 0),
                    (self.file_sinks[i], 0)
                )
        ##########################################################
        ##########################################################

        ##########################################################
        # This is for testing purposes only - using throttle     #
        # block to simulate Airspy device rate control           #
        # ️         REMOVE WHEN USING ACTUAL AIRSPY DEVICES ️      #
        ##########################################################
        if test:
            for i in range(self.num_antennas):
                # Connect: Noise → Throttle → Stream-to-Vec → FFT
                self.connect(
                    (airspy_devices[i], 0),
                    (self.throttle_blocks[i], 0)
                )
                self.connect(
                    (self.throttle_blocks[i], 0),
                    (stream_to_vec_blocks[i], 0)
                )
                self.connect(
                    (stream_to_vec_blocks[i], 0),
                    (fft_blocks[i], 0)
                )
        ##########################################################
        ##########################################################

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "Interferometer")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_sampling_rate(self):
        return self.sampling_rate

    def set_sampling_rate(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def get_integration_time(self):
        return self.integration_time

    def set_integration_time(self, integration_time):
        self.integration_time = integration_time

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size


def main(top_block_cls=Interferometer, options=None, test: bool = False):
    qapp = Qt.QApplication(sys.argv)

    if not os.path.exists(DATA_STORAGE_PATH):
        os.makedirs(DATA_STORAGE_PATH)

    tb = top_block_cls(
        sampling_rate=SAMPLING_RATE,
        integration_time=INTEGRATION_TIME,
        frequency=CENTER_FREQUENCY,
        fft_size=FFT_SIZE,
        num_antennas=len(DEVICE_LIST),
        folder_path=DATA_STORAGE_PATH,
        test=test
    ) 
    
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        print("\nShutting down...")
        # Flush all file sinks
        for fs in tb.file_sinks:
            fs.stop()
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    def observation_timeout():
        """Called when observation duration is reached"""
        print(f"\n🎯 Observation complete! Ran for {OBSERVATION_DURATION} seconds")
        print("Stopping interferometer...")
        sig_handler()
        

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    # Timer for automatic observation stop
    observation_timer = Qt.QTimer()
    observation_timer.setSingleShot(True)  # Only trigger once
    observation_timer.timeout.connect(observation_timeout)
    observation_timer.start(OBSERVATION_DURATION * 1000)  # Convert seconds to milliseconds

    print(f"🚀 Starting {OBSERVATION_DURATION}-second observation...")
    print(f"📊 Will automatically stop after {OBSERVATION_DURATION} seconds")

    qapp.exec_()

if __name__ == '__main__':
    main()