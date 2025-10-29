#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import osmosdr
from datetime import datetime, timezone
import os
import yaml
import json
import threading
import multiprocessing
import time
import os
from multiprocessing.pool import Pool

from data.obs_md import SharedMetadataManager
from utils import print_box


# Read configuration from YAML file
with open("observation_conf.yaml", 'r') as file:
    config = yaml.safe_load(file)

SAMPLING_RATE = config.get("sampling_rate", 10e6)
INTEGRATION_TIME = config.get("integration_time", 1)
FREQUENCY = config.get("center_frequency", 1.42e9)
FFT_SIZE = config.get("fft_size", 1024)

DEVICE_LIST = config.get("device_list", ["1", "2", "3", "4", "5", "6", "7", "8", "9"])
OBSERVATION_DURATION = config.get("observation_duration", 3600)


def print_config(session_folder_name: str = ""):
    msg = "Current Configuration:"
    msg += f"\nSampling Rate: {SAMPLING_RATE} Hz"
    msg += f"\nIntegration Time: {INTEGRATION_TIME} s"
    msg += f"\nCenter Frequency: {FREQUENCY} Hz"
    msg += f"\nFFT Size: {FFT_SIZE}"
    msg += f"\nObservation Duration: {OBSERVATION_DURATION} s"
    msg += f"\nDevice List"
    for device in DEVICE_LIST:
        msg += f"\n - {device}"

    if session_folder_name:
        msg += f"\nSession Folder: {session_folder_name}"
    print_box(msg)


class AntennaGNU(gr.top_block, Qt.QWidget):

    def __init__(self, device: str = "example_device", filename: str = "output.npy"):
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

        self.settings = Qt.QSettings("GNU Radio", "mock")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        
        # Calculate exact number of integrations for the observation duration
        total_integrations = OBSERVATION_DURATION  # Since integration_time = 1 second
        self.blocks_head_0 = blocks.head(gr.sizeof_float * FFT_SIZE, total_integrations)


        ##################################################
        # Variables
        ##################################################
        self.sampling_rate = sampling_rate = SAMPLING_RATE
        self.integration_time = integration_time = INTEGRATION_TIME
        self.frequency = frequency = FREQUENCY
        self.fft_size = fft_size = FFT_SIZE

        ##################################################
        # Blocks
        ##################################################

        # TODO: Replace osmosdr source with actual SDR source when testing with real hardware
        # self.osmosdr_source_0 = osmosdr.source(
        #     args="numchan=" + str(1) + " " + device
        # )
        # self.osmosdr_source_0.set_sample_rate(sampling_rate)
        # self.osmosdr_source_0.set_center_freq(frequency, 0)
        # self.osmosdr_source_0.set_freq_corr(0, 0)
        # self.osmosdr_source_0.set_dc_offset_mode(0, 0)
        # self.osmosdr_source_0.set_iq_balance_mode(0, 0)
        # self.osmosdr_source_0.set_gain_mode(False, 0)
        # self.osmosdr_source_0.set_gain(21, 0)
        # self.osmosdr_source_0.set_if_gain(0, 0)
        # self.osmosdr_source_0.set_bb_gain(0, 0)
        # self.osmosdr_source_0.set_antenna('', 0)
        # self.osmosdr_source_0.set_bandwidth(0, 0)

        # WTF IS THIS?: Basically using a noise source as a placeholder for the SDR source for testing
        # TODO: Remove this block when testing with real hardware
        from gnuradio import analog
        self.osmosdr_source_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 1, 0)


        self.fft_vxx_0 = fft.fft_vcc(fft_size, True, window.blackmanharris(1024), True, 1)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 1024)
        self.blocks_multiply_conjugate_cc_0 = blocks.multiply_conjugate_cc(fft_size)
        self.blocks_integrate_xx_0 = blocks.integrate_ff(integration_time, fft_size)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_float*1024, filename, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_complex_to_float_0 = blocks.complex_to_float(fft_size)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_complex_to_float_0, 0), (self.blocks_integrate_xx_0, 0))

        # Now using head block to limit number of integrations written to file
        # I am not deleting the old line in case we want to revert
        # I am also not sure if this is correct approach - ask in class
        # self.connect((self.blocks_integrate_xx_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_integrate_xx_0, 0), (self.blocks_head_0, 0)) # NEW
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0)) # NEW

        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self.blocks_complex_to_float_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_multiply_conjugate_cc_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_multiply_conjugate_cc_0, 1))
        self.connect((self.osmosdr_source_0, 0), (self.blocks_stream_to_vector_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "mock")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_sampling_rate(self):
        return self.sampling_rate

    def set_sampling_rate(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.osmosdr_source_0.set_sample_rate(self.sampling_rate)

    def get_integration_time(self):
        return self.integration_time

    def set_integration_time(self, integration_time):
        self.integration_time = integration_time

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.osmosdr_source_0.set_center_freq(self.frequency, 0)

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size


def main(top_block_cls=AntennaGNU, device: str = "example_device", filename: str = "output", metadata_manager=None):
    # Record start time with high precision
    start_time = datetime.now(timezone.utc)
    start_timestamp = start_time.timestamp()
    
    # Update process metadata with start time
    if metadata_manager:
        metadata_manager.update_process(
            device,
            filename=filename,
            start_time_utc=start_time.isoformat(),
            start_timestamp=start_timestamp,
            status="started",
            process_id=os.getpid()
        )

    qapp = Qt.QApplication(sys.argv)
    tb = top_block_cls(device=device, filename=filename)

    def finalize_metadata(reason="completed"):
        end_time = datetime.now(timezone.utc)
        end_timestamp = end_time.timestamp()
        duration = end_timestamp - start_timestamp
        
        process_data = {
            "end_time_utc": end_time.isoformat(),
            "end_timestamp": end_timestamp,
            "duration_seconds": duration,
            "status": reason
        }
        
        # Check final file size if it exists
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            process_data["file_size_bytes"] = file_size
            process_data["total_samples"] = file_size // 4
        
        # Update metadata
        if metadata_manager:
            metadata_manager.update_process(device, **process_data)
        
        print_box(f"Process for device {device} ended ({reason}): {duration:.3f}s duration")

    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        finalize_metadata("interupted")
        print_box(f"Process for device {device} ended.")
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # TODO: ASK IN CLASS
    # I am not sure if we need this to stop after observation duration or if head block is sufficient
    # Add observation timer
    observation_timer = Qt.QTimer()
    observation_timer.setSingleShot(True)
    observation_timer.timeout.connect(lambda: (
        tb.stop(), 
        tb.wait(), 
        finalize_metadata("observation_complete"),
        Qt.QApplication.quit()
    ))
    observation_timer.start(OBSERVATION_DURATION * 1000)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

def multithreaded_main():
    multiprocessing.set_start_method('spawn')

    current_date = time.strftime("%Y%m%d")
    current_time = time.strftime("%H%M%S")
    session_folder_name = "session_" + current_date + "_" + current_time

    print_config(session_folder_name)

    # Create session folder
    os.makedirs(session_folder_name, exist_ok=True)
    

    with multiprocessing.Manager() as manager:
        metadata_manager = SharedMetadataManager(
            manager,
            sampling_rate=SAMPLING_RATE,
            integration_time=INTEGRATION_TIME,
            frequency=FREQUENCY,
            fft_size=FFT_SIZE,
            observation_duration=OBSERVATION_DURATION
        )
        metadata_manager.initialize_session(session_folder_name, DEVICE_LIST)

        with Pool(processes=len(DEVICE_LIST)) as pool:
            # Generate unique filenames for each device
            args = [(AntennaGNU, device, os.path.join(session_folder_name, f"antenna_{i+1}_date_{current_date}_time_{current_time}"), metadata_manager) for i, device in enumerate(DEVICE_LIST)]
            
            # TODO: ASK IN CLASS
            # I am not sure if we should use starmap or starmap_async here.
            # Also not sure how to properly integrate stop_event into main
            # function and even if I should have it in the first place.
            # pool.starmap(main, args)

            # Start all processes
            result = pool.starmap_async(main, args)
            
            # Wait for all processes to complete
            result.wait()

        # Finalize session metadata
        metadata_manager.finalize_session()
        
        # Get completion stats
        completed, total = metadata_manager.get_completed_count()
        
        print_box("All processes completed successfully.")
        
        # Print final metadata summary
        session_duration = metadata_manager.shared_dict.get("duration_seconds", 0)
        info = f"Session duration: {session_duration:.2f} seconds"
        info += f"Processes completed: {completed}/{total}"
        print_box(info)


if __name__ == '__main__':
    multithreaded_main()
