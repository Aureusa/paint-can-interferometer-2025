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
import time
from gnuradio import analog

import os
import torch
import numpy as np

class VisibilityCorrelator(gr.sync_block):
    """
    Custom GNU Radio block that computes visibility matrix from multiple FFT inputs
    """
    def __init__(self, fft_size=1024, num_antennas=9, use_gpu=True):
        # Define input signature: 9 complex vector inputs
        in_sig = [(np.complex64, fft_size)] * num_antennas
        # out_sig = (num_antennas, num_antennas, fft_size)

        # Define output signature: flattened visibility matrix
        # We'll output the upper triangle of the visibility matrix (including diagonal)
        num_baselines = num_antennas * (num_antennas + 1) // 2
        out_sig = [(np.complex64, fft_size)] * num_baselines
        
        
        gr.sync_block.__init__(
            self,
            name="visibility_correlator",
            in_sig=in_sig,
            out_sig=out_sig,
        )
        
        self.fft_size = fft_size
        self.num_antennas = num_antennas
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        print(f"VisibilityCorrelator: {num_antennas} antennas, GPU: {self.use_gpu}")

        # Pre-compute indices for extracting upper triangle
        self.num_baselines = num_baselines
        self.baseline_indices = []
        for i in range(num_antennas):
            for j in range(i, num_antennas):
                self.baseline_indices.append((i, j))
    
    def work(self, input_items, output_items):
        """Process each set of FFT vectors using matrix operations"""
        num_samples = len(input_items[0])  # Number of time samples to process

        # Stack all antenna FFT data into matrix: (num_antennas, num_samples, fft_size)
        fft_matrix = torch.stack([
            torch.tensor(input_items[ant_idx], device=self.device) 
            for ant_idx in range(self.num_antennas)
        ])

        # Compute visibility matrix for all samples at once
        # Shape: (num_antennas, num_antennas, num_samples, fft_size)
        visibility_matrix = torch.einsum(
            'iaf,jaf->ijaf',
            fft_matrix,
            torch.conj(fft_matrix)
        )

        # Extract upper triangle baselines and assign to outputs
        for baseline_idx, (i, j) in enumerate(self.baseline_indices):
            # Shape: (num_samples, fft_size)
            baseline_data = visibility_matrix[i, j, :, :].cpu().numpy()
            output_items[baseline_idx][:] = baseline_data
        
        return num_samples  # Return number of samples processed


class VisibilityIntegrator(gr.sync_block):
    """
    Simple integrator for visibility data - just integrates and outputs
    """
    def __init__(self, fft_size=1024, num_baselines=45, integration_samples=1000, use_gpu=True):
        
        # Input signature: visibility data from correlator
        in_sig = [(np.complex64, fft_size)] * num_baselines
        # Output signature: same as input but at reduced rate
        out_sig = [(np.complex64, fft_size)] * num_baselines
        
        gr.sync_block.__init__(
            self,
            name="visibility_integrator",
            in_sig=in_sig,
            out_sig=out_sig,
        )
        
        self.fft_size = fft_size
        self.num_baselines = num_baselines
        self.integration_samples = integration_samples
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Integration buffers
        self.reset_buffers()
        
        print(f"VisibilityIntegrator: {integration_samples} samples, GPU: {self.use_gpu}")

    def reset_buffers(self):
        """Reset integration buffers"""
        if self.use_gpu:
            self.buffer = torch.zeros(
                (self.num_baselines, self.fft_size), 
                dtype=torch.complex64, 
                device=self.device
            )
        else:
            self.buffer = np.zeros(
                (self.num_baselines, self.fft_size), 
                dtype=np.complex64
            )
        self.count = 0

    def work(self, input_items, output_items):
        """Integrate visibility data"""
        num_input_samples = len(input_items[0])
        num_output_samples = 0
        
        for sample_idx in range(num_input_samples):
            # Add current sample to buffer
            if self.use_gpu:
                current_sample = torch.stack([
                    torch.tensor(input_items[baseline][sample_idx], device=self.device)
                    for baseline in range(self.num_baselines)
                ])
                self.buffer += current_sample
            else:
                for baseline in range(self.num_baselines):
                    self.buffer[baseline] += input_items[baseline][sample_idx]
            
            self.count += 1
            
            # Output when integration is complete
            if self.count >= self.integration_samples:
                # Average and output
                if self.use_gpu:
                    averaged = (self.buffer / self.count).cpu().numpy()
                else:
                    averaged = self.buffer / self.count
                
                for baseline in range(self.num_baselines):
                    output_items[baseline][num_output_samples] = averaged[baseline]
                
                num_output_samples += 1
                self.reset_buffers()
        
        return num_output_samples
    

class VisibilityFileSink(gr.sync_block):
    """
    Simple file sink for visibility data
    """
    def __init__(self, fft_size=1024, num_baselines=45, filename_prefix='visibility_', use_gpu=True, flush_interval=10):
        
        # Input signature: visibility data from integrator
        in_sig = [(np.complex64, fft_size)] * num_baselines
        out_sig = []
        
        gr.sync_block.__init__(
            self,
            name="visibility_file_sink",
            in_sig=in_sig,
            out_sig=out_sig,
        )
        
        self.fft_size = fft_size
        self.num_baselines = num_baselines
        self.filename_prefix = filename_prefix
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Open file handles
        self.file_handles = [
            open(f"{self.filename_prefix}baseline_{i}_{j}.bin", 'ab')
            for i in range(9) for j in range(i, 9)
        ]

        # Monitor flush interval
        self.flush_interval = flush_interval
        self.write_count = 0
        
    def work(self, input_items, output_items):
        """Write visibility data to files"""
        num_samples = len(input_items[0])
        
        for sample_idx in range(num_samples):
            for baseline_idx in range(self.num_baselines):
                data = input_items[baseline_idx][sample_idx]
                self.file_handles[baseline_idx].write(data.tobytes())

        self.write_count += num_samples
        
        # Flush periodically, not every time
        if self.write_count >= self.flush_interval:
            for fh in self.file_handles:
                fh.flush()
            self.write_count = 0
        
        return num_samples
    
    def stop(self):
        """Called when flowgraph stops - flush remaining data"""
        for fh in self.file_handles:
            fh.flush()
            fh.close()
        return True

    def close(self):
        """Close all file handles"""
        for fh in self.file_handles:
            fh.close()


class Interferometer(gr.top_block, Qt.QWidget):
    """
    Interferometer schema for processing visibility data from multiple antennas.
    The scheme includes:
    - Airspy SDR sources for each antenna
    - FFT blocks to convert time-domain data to frequency domain
    - Visibility correlator to compute visibilities between antenna pairs
    - Visibility integrator to average visibilities over time
    - File sink to store visibility data to disk

    The flow is as follows:
    Airspy Source ->
    Stream to Vector ->
    FFT ->
    Visibility Correlator ->
    Visibility Integrator ->
    File Sink    
    """
    def __init__(
            self,
            sampling_rate=10e6,
            integration_time=1,
            frequency=1.42e9,
            fft_size=1024,
            num_antennas=9,
            folder_path="test_folder"
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
        # self.airspy_devices = [
        #     self.def_airspy_device(f"airspy={i}", sampling_rate, frequency)
        #     for i in range(self.num_antennas)
        # ]

        # Define gaussian noise sources as placeholders for Airspy devices
        self.airspy_devices = [
            analog.noise_source_c(analog.GR_GAUSSIAN, 0.1, seed=i*42)
            for i in range(self.num_antennas)
        ]

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

        # Create Visibility Correlator Block
        self.cross_correlator = VisibilityCorrelator(
            fft_size=fft_size, 
            num_antennas=self.num_antennas, 
            use_gpu=True
        )

        # Create Visibility Integrator Block
        self.visibility_integrator = VisibilityIntegrator(
            fft_size=fft_size,
            num_baselines=self.num_antennas * (self.num_antennas + 1) // 2,
            integration_samples=int(integration_time * (sampling_rate / fft_size)),
            use_gpu=True
        )

        # Create Visibility File Sink Block
        self.visibility_file_sink = VisibilityFileSink(
            fft_size=fft_size,
            num_baselines=self.num_antennas * (self.num_antennas + 1) // 2,
            filename_prefix=os.path.join(folder_path, 'visibility_'),
            use_gpu=True,
            flush_interval=10
        )

        ##################################################
        # Connections
        ##################################################

        # Connect each airspy -> stream_to_vector -> fft
        self.connect_airspy_to_fft(self.airspy_devices, self.stream_to_vector_blocks, self.fft_blocks)

        # Connect all FFT outputs to correlator inputs
        self.connect_fft_to_correlator(self.fft_blocks, self.cross_correlator)

        # Connect all correlator outputs to integrator inputs
        self.connect_correlator_to_integrator(self.cross_correlator, self.visibility_integrator)

        # Connect all integrator outputs to file sink inputs
        self.connect_integrator_to_file_sink(self.visibility_integrator, self.visibility_file_sink)

    def def_airspy_device(self, device, sampling_rate, frequency):
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

    def connect_airspy_to_fft(self, airspy_devices, stream_to_vec_blocks, fft_blocks):
        for i in range(self.num_antennas):
            self.connect(
                (airspy_devices[i], 0),
                (stream_to_vec_blocks[i], 0)
            )
            self.connect(
                (stream_to_vec_blocks[i], 0),
                (fft_blocks[i], 0)
            )

    def connect_fft_to_correlator(self, fft_blocks, correlator_block):
        for i in range(self.num_antennas):
            self.connect(
                (fft_blocks[i], 0),
                (correlator_block, i)
            )

    def connect_correlator_to_integrator(self, correlator_block, integrator_block):
        num_baselines = self.num_antennas * (self.num_antennas + 1) // 2
        for i in range(num_baselines):
            self.connect(
                (correlator_block, i),
                (integrator_block, i)
            )

    def connect_integrator_to_file_sink(self, integrator_block, file_sink_block):
        num_baselines = self.num_antennas * (self.num_antennas + 1) // 2
        for i in range(num_baselines):
            self.connect(
                (integrator_block, i),
                (file_sink_block, i)
            )

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




def main(top_block_cls=Interferometer, options=None):

    qapp = Qt.QApplication(sys.argv)
    folder_name = "test_folder"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    tb = top_block_cls(
        sampling_rate=10e6,
        integration_time=0.1,
        frequency=1.42e9,
        fft_size=1024,
        num_antennas=9,
        folder_path=folder_name
    )

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
