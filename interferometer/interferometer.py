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
import time

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
            frequency=1.42e9,
            num_antennas=9,
            folder_path="mock_data",
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
        self.frequency = frequency
        self.sampling_rate = sampling_rate
        self.num_antennas = num_antennas

        ##################################################
        # Blocks
        ##################################################

        # Create multiple Airspy devices
        self.airspy_devices = [
            self.def_airspy_device(device, sampling_rate, frequency)
            for device in DEVICE_LIST
        ]
        
        # Create file sinks for each Airspy
        self.file_sinks = [
            blocks.file_sink(
                gr.sizeof_gr_complex,
                os.path.join(folder_path, f'antenna_{device.split("=")[1]}.dat'),
                False
            )
            for device in DEVICE_LIST
        ]

        ##################################################
        # Connections
        ##################################################

        # Connect each airspy -> filesink
        for i in range(self.num_antennas):
            self.connect(
                (self.airspy_devices[i], 0),
                (self.file_sinks[i], 0)
            )

        self._print_interferometer_info()

    def _print_interferometer_info(self):
        info = "Interferometer Configuration:"
        info += f"\n  Number of Antennas: {self.num_antennas}"
        info += f"\n  Sampling Rate: {self.sampling_rate/1e6} MHz"
        info += f"\n  Frequency: {self.frequency/1e6} MHz"
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

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency

def _wait_for_start_time():
    """Wait until the configured start time before proceeding"""
    # Get the configured start time
    start_time_config = OBSERVATION_CONFIG.get('start_time', None)
    
    # Handle both string and datetime objects from YAML
    if start_time_config is None:
        print("No start_time configured - starting immediately")
        start_time_str = "Not configured"
    elif isinstance(start_time_config, str):
        # It's already a string (like "2025-11-14T15:25:00Z")
        start_time_str = start_time_config
    else:
        # It's a datetime object from YAML parsing
        # Check if it has timezone info to avoid adding Z incorrectly
        if hasattr(start_time_config, 'tzinfo') and start_time_config.tzinfo is not None:
            # Convert to UTC and format properly
            import datetime
            start_time_utc = start_time_config.astimezone(datetime.timezone.utc)
            start_time_str = start_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            # Assume it's already UTC
            start_time_str = start_time_config.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Get current time
    curr_time_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print(f"Observation start time (UTC): {start_time_str}")
    print(f"Current time (UTC): {curr_time_str}")

    # Calculate time difference if we have a valid start time
    if start_time_config is not None:
        try:
            import datetime
            if isinstance(start_time_config, str):
                # Handle Z suffix properly
                if start_time_config.endswith('Z'):
                    start_time = datetime.datetime.fromisoformat(start_time_config[:-1] + '+00:00')
                else:
                    start_time = datetime.datetime.fromisoformat(start_time_config)
            else:
                # Handle datetime object
                if hasattr(start_time_config, 'tzinfo') and start_time_config.tzinfo is not None:
                    start_time = start_time_config.astimezone(datetime.timezone.utc)
                else:
                    start_time = start_time_config.replace(tzinfo=datetime.timezone.utc)
            
            curr_time = datetime.datetime.now(datetime.timezone.utc)
            time_difference = (start_time - curr_time).total_seconds()
            print(f"Time difference: {time_difference:.1f} seconds")
            
            if time_difference > 0:
                print(f"Would wait {time_difference:.1f} seconds...")
            else:
                print("Start time has already passed")
        except Exception as e:
            print(f"Error calculating time difference: {e}")

    if time_difference < 0:
        raise SystemExit("The time difference is not positive - please double check the start_time configuration!")

    print(f"Waiting for scheduled start time...")
    time.sleep(time_difference)

    # Convert start_time_str to Amsterdam time
    start_time_amst_time = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    print(f"Starting interferometer at {start_time_amst_time}...")

    # Create folder like session_20231015_153000
    create_session_folder = f"session_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"

    data_folder = os.path.join(DATA_STORAGE_PATH, create_session_folder)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    return data_folder


def main(top_block_cls=Interferometer, options=None):
    # Wait for starting time
    data_folder = _wait_for_start_time()

    qapp = Qt.QApplication(sys.argv)

    if not os.path.exists(DATA_STORAGE_PATH):
        os.makedirs(DATA_STORAGE_PATH)

    tb = top_block_cls(
        sampling_rate=SAMPLING_RATE,
        frequency=CENTER_FREQUENCY,
        num_antennas=len(DEVICE_LIST),
        folder_path=data_folder,
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
