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

import time
from datetime import datetime
from collections import defaultdict
import threading

from utils import print_box

# Global verbose control - set to False to reduce output
VERBOSE = False

class VisibilityCorrelator(gr.sync_block):
    """
    Custom GNU Radio block that computes visibility matrix from multiple FFT inputs
    """
    def __init__(self, fft_size=1024, num_antennas=9, use_gpu=True):
        # Define input signature: complex vector inputs from FFTs
        in_sig = [(np.complex64, fft_size)] * num_antennas

        # Define output signature: flattened visibility matrix
        self.num_baselines = num_antennas * (num_antennas + 1) // 2
        out_sig = [(np.complex64, fft_size)] * self.num_baselines
        
        # Initialize GNU Radio sync block - THIS WAS MISSING!
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
        
        # Pre-compute indices for extracting upper triangle
        self.baseline_indices = []
        for i in range(num_antennas):
            for j in range(i, num_antennas):
                self.baseline_indices.append((i, j))
        
        # Add monitoring variables
        self.sample_count = 0
        self.start_time = None
        self.last_report_time = 0
        self.samples_per_antenna = [0] * num_antennas
        self.data_fingerprints = [[] for _ in range(num_antennas)]
        self.first_data_time = [None] * num_antennas

        self._print_correlator_info()

    def _print_correlator_info(self):
        info = "VisibilityCorrelator Configuration:"
        info += f"\n  Number of Antennas: {self.num_antennas}"
        info += f"\n  Number of Baselines: {self.num_baselines}"
        info += f"\n  Using GPU: {self.use_gpu}"
        print_box(info)

    def work(self, input_items, output_items):
        """Process each set of FFT vectors using matrix operations"""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
            if VERBOSE:
                print(f"VisibilityCorrelator: Data flow started at {time.strftime('%H:%M:%S', time.localtime())}")
        
        num_samples = len(input_items[0])
        
        # Monitor each antenna for data arrival and content
        for ant_idx in range(self.num_antennas):
            if len(input_items[ant_idx]) != num_samples:
                print(f"WARNING: Antenna {ant_idx} has {len(input_items[ant_idx])} samples, expected {num_samples}")
            
            self.samples_per_antenna[ant_idx] += len(input_items[ant_idx])
            
            # Record first data arrival time for each antenna
            if self.first_data_time[ant_idx] is None and len(input_items[ant_idx]) > 0:
                self.first_data_time[ant_idx] = current_time - self.start_time
                if VERBOSE:
                    print(f"Antenna {ant_idx}: First data arrived at +{self.first_data_time[ant_idx]:.3f}s")
            
            # Create data fingerprint (mean of first few samples for comparison)
            if len(input_items[ant_idx]) > 0:
                fingerprint = np.mean(np.abs(input_items[ant_idx][:min(10, len(input_items[ant_idx]))]))
                self.data_fingerprints[ant_idx].append(fingerprint)
                # Keep only last 100 fingerprints
                if len(self.data_fingerprints[ant_idx]) > 100:
                    self.data_fingerprints[ant_idx].pop(0)

        self.sample_count += num_samples
        
        # Report statistics every 5 seconds
        if VERBOSE and current_time - self.last_report_time > 5.0:
            self.report_statistics(current_time)
            self.last_report_time = current_time

        fft_matrix = torch.stack([
            torch.tensor(input_items[ant_idx], device=self.device) 
            for ant_idx in range(self.num_antennas)
        ])

        visibility_matrix = torch.einsum(
            'iaf,jaf->ijaf',
            fft_matrix,
            torch.conj(fft_matrix)
        )

        for baseline_idx, (i, j) in enumerate(self.baseline_indices):
            baseline_data = visibility_matrix[i, j, :, :].cpu().numpy()
            output_items[baseline_idx][:] = baseline_data
        
        return num_samples

    def report_statistics(self, current_time):
        """Report comprehensive statistics about data flow"""
        elapsed = current_time - self.start_time
        overall_rate = self.sample_count / elapsed if elapsed > 0 else 0
        
        # Check for sample count differences (indicates dropped samples)
        min_samples = min(self.samples_per_antenna)
        max_samples = max(self.samples_per_antenna)
        
        # Always show warnings about sample loss
        if max_samples - min_samples > 0:
            print(f"WARNING: Sample count mismatch detected! Difference: {max_samples - min_samples}")
            for i, count in enumerate(self.samples_per_antenna):
                if count != max_samples:
                    print(f"  Antenna {i}: {count} samples (missing {max_samples - count})")
        
        # Check data similarity and warn about anomalies
        if all(len(fp) > 10 for fp in self.data_fingerprints):
            recent_means = [np.mean(fp[-10:]) for fp in self.data_fingerprints]
            overall_mean = np.mean(recent_means)
            for i, level in enumerate(recent_means):
                deviation = abs(level - overall_mean) / overall_mean * 100 if overall_mean > 0 else 0
                if deviation > 50:  # Warn about major deviations
                    print(f"WARNING: Antenna {i} data level anomaly: {level:.4f} ({deviation:.1f}% from mean)")
        
        # Only show full statistics if verbose
        if VERBOSE:
            print(f"\n=== VisibilityCorrelator Statistics (t={elapsed:.1f}s) ===")
            print(f"Overall sample rate: {overall_rate:.1f} samples/sec")
            print(f"Total samples processed: {self.sample_count}")
            print(f"Samples per antenna: min={min_samples}, max={max_samples}")
            if max_samples - min_samples == 0:
                print("✓ All antennas have equal sample counts")
            
            # Check data similarity (are all antennas sending similar noise levels?)
            if all(len(fp) > 10 for fp in self.data_fingerprints):
                recent_means = [np.mean(fp[-10:]) for fp in self.data_fingerprints]
                overall_mean = np.mean(recent_means)
                print(f"Recent data levels (should be similar for noise sources):")
                for i, level in enumerate(recent_means):
                    deviation = abs(level - overall_mean) / overall_mean * 100 if overall_mean > 0 else 0
                    status = "✓" if deviation < 20 else "⚠️"
                    print(f"  Antenna {i}: {level:.4f} ({deviation:.1f}% from mean) {status}")
            
            print("=" * 50)

class VisibilityIntegrator(gr.sync_block):
    """
    Simple integrator for visibility data - just integrates and outputs
    """
    def __init__(self, fft_size=1024, num_baselines=45, integration_samples=1000, use_gpu=True):
        # Input signature: visibility data from correlator
        in_sig = [(np.complex64, fft_size)] * num_baselines
        # Output signature: same as input but at reduced rate
        out_sig = [(np.complex64, fft_size)] * num_baselines
        
        # Initialize GNU Radio sync block - THIS WAS MISSING!
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
        
        # Add monitoring
        self.input_sample_count = 0
        self.output_sample_count = 0
        self.start_time = None
        self.last_report_time = 0

        self._print_integrator_info()

    def _print_integrator_info(self):
        info = "VisibilityIntegrator Configuration:"
        info += f"\n  Integration Samples: {self.integration_samples}"
        info += f"\n  Using GPU: {self.use_gpu}"
        print_box(info)

    def work(self, input_items, output_items):
        """Integrate visibility data"""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
            if VERBOSE:
                print(f"VisibilityIntegrator: Data flow started at {time.strftime('%H:%M:%S', time.localtime())}")

        num_input_samples = len(input_items[0])
        num_output_samples = 0
        self.input_sample_count += num_input_samples
        
        if VERBOSE:
            print(f"VisibilityIntegrator: Processing {num_input_samples} samples, buffer: {self.count}/{self.integration_samples}")
        
        for sample_idx in range(num_input_samples):
            # ... existing integration code ...
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
            
            if self.count >= self.integration_samples:
                if VERBOSE:
                    print(f"VisibilityIntegrator: Integration complete! Outputting integrated sample {num_output_samples}")
                
                if self.use_gpu:
                    averaged = (self.buffer / self.count).cpu().numpy()
                else:
                    averaged = self.buffer / self.count
                
                for baseline in range(self.num_baselines):
                    output_items[baseline][num_output_samples] = averaged[baseline]
                
                num_output_samples += 1
                self.output_sample_count += 1
                self.reset_buffers()
        
        # Report every 10 seconds
        if VERBOSE and current_time - self.last_report_time > 10.0:
            elapsed = current_time - self.start_time
            input_rate = self.input_sample_count / elapsed if elapsed > 0 else 0
            print(f"VisibilityIntegrator: Input rate: {input_rate:.1f} samples/sec, "
                  f"Integrated {self.output_sample_count} samples total")
            self.last_report_time = current_time
        
        return num_output_samples
    
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

class VisibilityFileSink(gr.sync_block):
    """
    Simple file sink for visibility data
    """
    def __init__(self, fft_size=1024, num_baselines=45, data_folder="test_folder", use_gpu=True, flush_interval=10):
        # Input signature: visibility data from integrator
        in_sig = [(np.complex64, fft_size)] * num_baselines
        out_sig = []  # No outputs, this is a sink
        
        # Initialize GNU Radio sync block - THIS WAS MISSING!
        gr.sync_block.__init__(
            self,
            name="visibility_file_sink",
            in_sig=in_sig,
            out_sig=out_sig,
        )
        
        self.fft_size = fft_size
        self.num_baselines = num_baselines
        self.data_folder = data_folder
        self.flush_interval = flush_interval
        self.write_count = 0
        
        # Open file handles in append mode
        self.file_handles = []
        baseline_idx = 0
        for i in range(int(np.sqrt(2 * num_baselines))):  # Calculate num_antennas from num_baselines
            for j in range(i, int(np.sqrt(2 * num_baselines))):
                if baseline_idx < num_baselines:
                    filename = os.path.join(self.data_folder, f"baseline_{i}_{j}.bin")
                    self.file_handles.append(open(filename, 'ab'))
                    baseline_idx += 1
        
        # Add monitoring
        self.samples_written = 0
        self.bytes_written = 0
        self.start_time = None
        self.last_report_time = 0

        self._print_visibility_info()

    def _print_visibility_info(self):
        info = "VisibilityFileSink Configuration:"
        info += f"\n  FFT Size: {self.fft_size}"
        info += f"\n  Number of Baselines: {self.num_baselines}"
        info += f"\n  Data Folder: {self.data_folder}"
        info += f"\n  Flush Interval: {self.flush_interval} samples"
        print_box(info)

    def work(self, input_items, output_items):
        """Write visibility data to files"""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
            if VERBOSE:
                print(f"VisibilityFileSink: Data flow started at {time.strftime('%H:%M:%S', time.localtime())}")
                print("VisibilityFileSink: ✓ Writing to disk has begun!")

        num_samples = len(input_items[0])
        
        if VERBOSE and num_samples > 0:
            print(f"VisibilityFileSink: Writing {num_samples} samples to {self.num_baselines} files")
        
        for sample_idx in range(num_samples):
            for baseline_idx in range(self.num_baselines):
                data = input_items[baseline_idx][sample_idx]
                bytes_to_write = data.tobytes()
                self.file_handles[baseline_idx].write(bytes_to_write)
                self.bytes_written += len(bytes_to_write)

        self.samples_written += num_samples
        self.write_count += num_samples
        
        if self.write_count >= self.flush_interval:
            if VERBOSE:
                print(f"VisibilityFileSink: Flushing {self.write_count} samples to disk")
            for fh in self.file_handles:
                fh.flush()
            self.write_count = 0
        
        # Report every 15 seconds
        if VERBOSE and current_time - self.last_report_time > 15.0:
            elapsed = current_time - self.start_time
            sample_rate = self.samples_written / elapsed if elapsed > 0 else 0
            data_rate_mb = (self.bytes_written / elapsed) / (1024*1024) if elapsed > 0 else 0
            
            print(f"\n=== VisibilityFileSink Statistics (t={elapsed:.1f}s) ===")
            print(f"Samples written: {self.samples_written}")
            print(f"Sample rate: {sample_rate:.2f} samples/sec")
            print(f"Data rate: {data_rate_mb:.2f} MB/sec")
            print(f"Total data written: {self.bytes_written / (1024*1024):.2f} MB")
            print("=" * 50)
            self.last_report_time = current_time
        
        return num_samples
    
    def stop(self):
        """Called when flowgraph stops - close files"""
        for fh in self.file_handles:
            if not fh.closed:
                fh.close()
        return True


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
            data_folder=folder_path,
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

        self._print_interferometer_info()

    def _print_interferometer_info(self):
        info = "Interferometer Configuration:"
        info += f"\n  Number of Antennas: {self.num_antennas}"
        info += f"\n  Sampling Rate: {self.sampling_rate/1e6} MHz"
        info += f"\n  Frequency: {self.frequency/1e6} MHz"
        info += f"\n  FFT Size: {self.fft_size}"
        info += f"\n  Integration Time: {self.integration_time} seconds"
        print_box(info)

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


# Add this test function to check file contents
def analyze_output_files(folder_path):
    """Analyze the output files to check for data consistency"""
    import glob
    
    print(f"\n=== Analyzing output files in {folder_path} ===")
    
    files = glob.glob(os.path.join(folder_path, "visibility_*.bin"))
    if not files:
        print("No output files found!")
        return
    
    print(f"Found {len(files)} output files:")
    
    file_sizes = []
    for file_path in sorted(files):
        size = os.path.getsize(file_path)
        file_sizes.append(size)
        print(f"  {os.path.basename(file_path)}: {size} bytes ({size/(8*1024):.1f} KB)")
    
    # Check if all files have similar sizes (they should for noise sources)
    if file_sizes:
        min_size, max_size = min(file_sizes), max(file_sizes)
        if max_size - min_size == 0:
            print("✓ All files have identical sizes")
        else:
            print(f"⚠️ File size variation: {min_size} to {max_size} bytes (diff: {max_size-min_size})")

# Modify the main function to include monitoring
def main(top_block_cls=Interferometer, options=None):
    qapp = Qt.QApplication(sys.argv)
    folder_name = "test_folder"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Start with shorter integration time for faster feedback
    tb = top_block_cls(
        sampling_rate=10e6,
        integration_time=0.01,  # Very short for quick testing
        frequency=1.42e9,
        fft_size=1024,
        num_antennas=9,
        folder_path=folder_name
    )

    print("Starting interferometer test...")
    if VERBOSE:
        print("Monitor console output for:")
        print("  - First data arrival times from each antenna")
        print("  - Sample count consistency")
        print("  - Data flow rates")
        print("  - File writing progress")
    else:
        print("Running in quiet mode. Only warnings and errors will be displayed.")
        print(f"Set VERBOSE = True (line 32) for detailed monitoring.")
    
    tb.start()
    tb.show()

    # Schedule file analysis after 30 seconds
    def analyze_files():
        time.sleep(30)
        analyze_output_files(folder_name)
    
    analysis_thread = threading.Thread(target=analyze_files, daemon=True)
    analysis_thread.start()

    def sig_handler(sig=None, frame=None):
        print("\nShutting down...")
        analyze_output_files(folder_name)  # Final analysis
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