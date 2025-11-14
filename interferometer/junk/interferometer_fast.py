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
import time
import signal
import threading
from datetime import datetime
from collections import defaultdict
from argparse import ArgumentParser

# Third-party imports
import numpy as np
import torch
from PyQt5 import Qt

# GNU Radio imports
from gnuradio import gr
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import fft
from gnuradio import analog
from gnuradio.fft import window
from gnuradio.filter import firdes
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
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

# Extract folder
DATA_FOLDER = OBSERVATION_CONFIG['data_storage_path']


class VisibilityCorrelator(gr.sync_block):
    """
    Custom GNU Radio block that computes visibility matrix AND integrates over time
    """
    def __init__(self, fft_size=1024, num_antennas=9, integration_samples=100000, use_gpu=True):
        # Define input signature: complex vector inputs from FFTs
        in_sig = [(np.complex64, fft_size)] * num_antennas

        # Define output signature: integrated visibility matrix
        self.num_baselines = num_antennas * (num_antennas + 1) // 2
        out_sig = [(np.complex64, fft_size)] * self.num_baselines
        
        # Initialize GNU Radio sync block
        gr.sync_block.__init__(
            self,
            name="visibility_correlator",
            in_sig=in_sig,
            out_sig=out_sig,
        )
        
        # Store parameters
        self.fft_size = fft_size
        self.num_antennas = num_antennas
        self.integration_samples = integration_samples

        # Device setup (GPU/CPU) - Always check for CUDA availability
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Pre-compute indices for extracting upper triangle - N(N+1)/2 baselines
        self.baseline_indices = []
        baseline_i_indices = []
        baseline_j_indices = []
        
        for i in range(num_antennas):
            for j in range(i, num_antennas):
                self.baseline_indices.append((i, j))
                baseline_i_indices.append(i)
                baseline_j_indices.append(j)

        # Store as tensors/arrays for vectorized access
        if self.use_gpu:
            self.baseline_i_indices = torch.tensor(baseline_i_indices, device=self.device)
            self.baseline_j_indices = torch.tensor(baseline_j_indices, device=self.device)
        else:
            self.baseline_i_indices = np.array(baseline_i_indices)
            self.baseline_j_indices = np.array(baseline_j_indices)

        # Initialize buffers
        self.reset_integration_buffers()
        
        # Add monitoring variables
        self.sample_count = 0
        self.output_count = 0
        self.start_time = None
        self.last_report_time = 0

        self.total_samples_processed = 0

        self._print_correlator_info()

    def _print_correlator_info(self):
        """Print configuration info"""
        info = "VisibilityCorrelator Configuration:"
        info += f"\n  Number of Antennas: {self.num_antennas}"
        info += f"\n  Number of Baselines: {self.num_baselines}"
        info += f"\n  Integration Samples: {self.integration_samples:,}"
        info += f"\n  Using GPU: {self.use_gpu}"
        print_box(info)

    def reset_integration_buffers(self):
        """Reset integration buffers"""
        self.integration_buffer = np.zeros(
            (self.num_antennas, 1, self.fft_size), 
            dtype=np.complex64
        )
        self.integration_count = 0

    def work(self, input_items, output_items):
        """
        Process FFT vectors from multiple antennas, compute cross-correlations, and integrate over time.
        
        This is the core processing method of the VisibilityCorrelator block that implements a radio 
        astronomy correlator with built-in integration. The method performs batch correlation of FFT 
        vectors from all antenna pairs and accumulates the results over time to improve signal-to-noise 
        ratio through coherent integration.
        
        Processing Pipeline:
        1. **Input Reception**: Receives FFT vectors from all antennas simultaneously
        2. **GPU Transfer**: Moves input data to GPU device for accelerated processing
        3. **Cross-Correlation**: Computes visibility matrix V_ij = F_i * conj(F_j) for all antenna pairs
        4. **Time Integration**: Sums correlations over time samples within the current batch
        5. **Accumulation**: Adds batch result to integration buffer for long-term averaging
        6. **Output Generation**: When integration period completes, outputs time-averaged visibilities
        
        Mathematical Operations:
        - Visibility computation uses Einstein summation: torch.einsum('iaf,jaf->ijaf', F, F*)
        where i,j = antenna indices, a = time samples, f = frequency bins
        - Only upper triangle baselines are stored (N(N+1)/2 instead of N²) for memory efficiency
        - Integration averaging: V_avg = (∑V_batch) / N_samples for noise reduction
        
        Performance Optimizations:
        - Vectorized operations using pre-computed baseline indices for GPU acceleration
        - Batch processing of multiple time samples reduces GPU transfer overhead
        - Single-precision complex64 arithmetic for optimal GPU performance
        
        GNU Radio Integration:
        - Implements decimation pattern: processes multiple input samples, outputs one integrated sample
        - Returns 0 during accumulation phase, returns 1 when integration completes
        - Automatically manages buffer reset for continuous operation
        
        Args:
            input_items (list): List of numpy arrays, one per antenna
                            Shape: [num_antennas][time_samples, fft_size] 
                            dtype: complex64 FFT vectors from each antenna
            output_items (list): List of output arrays, one per baseline  
                            Shape: [num_baselines][output_samples, fft_size]
                            dtype: complex64 integrated visibility data
        
        Returns:
            int: Number of output samples produced this call
                - 0: Still accumulating, no output ready
                - 1: Integration complete, one averaged sample output per baseline
        
        Data Flow Example:
            Batch 1-(n-1): Process m samples each → accumulate → return 0
            Batch n:   Process m samples → complete integration → output 1 sample → return 1
            Batch n+1:   Start new integration cycle → return 0
            Variables:
            n = integration_samples / m
            m = number of time samples per work call - depends on GNU Radio scheduler
        
        Integration Logic:
            - integration_samples: Target number of FFT vectors to average
            - integration_count: Current number of FFT vectors accumulated
            - Trigger output when we reached desired number of integration samples:
              integration_count >= integration_samples
            - Integration time ≈ integration_samples * fft_size / sampling_rate
        
        Memory Management:
            - integration_buffer: Accumulates correlation sums [num_baselines, fft_size]
            - Automatic GPU memory transfers minimized for performance
            - Buffer reset after each integration cycle for continuous operation
        
        Monitoring & Statistics:
            - Tracks sample processing count and timing
            - Reports statistics every 30 seconds for performance monitoring
            - Measures total samples processed and integration cycles completed
        
        Error Handling:
            - Graceful fallback to CPU if GPU operations fail
            - Consistent data types between GPU and CPU processing paths
            - Automatic device management
        
        Radio Astronomy Context:
            This correlator implements the fundamental operation of interferometry:
            measuring complex visibilities between antenna pairs to reconstruct
            sky brightness distributions. The integration process improves sensitivity
            by averaging out thermal noise while preserving coherent astronomical signals.
        """
        # Monitoring - start time
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        # Number of input samples available and output samples to produce (initially zero)
        # The input items have a shape like: [num_antennas, time_samples, fft_size]
        # Thus, we have that len(input_items[0]) = time_samples
        inp_arr = np.array(input_items) # Shape: [num_antennas, time_samples, fft_size]
        arr_shape = inp_arr.shape
        num_input_samples = arr_shape[1] * arr_shape[2]
        
        # Add to self.integration_buffer
        np.append(inp_arr, self.integration_buffer, axis=1)
        self.integration_count += num_input_samples

        # Check if we have reached the integration sample count
        if self.integration_count >= self.integration_samples:
            # At this point, we effectively have an integration buffer with shape [num_baselines, fft_size]
            # that is the sum over all samples processed so far.
            # The averaged visibility is simply this buffer divided by the number of samples we added to the buffer.
            # averaged is now shape: [num_baselines, fft_size]
            integration_buffer = self.integration_buffer[:,1:,:] # Remove the initial zero column

            if self.use_gpu:
                fft_matrix = torch.from_numpy(self.integration_buffer).to(
                    self.device, non_blocking=True
                ) # Shape: [num_antennas, num_samples, fft_size]

                visibilities = torch.einsum(
                    'iaf,jaf->ijaf',
                    fft_matrix,
                    torch.conj(fft_matrix)
                ) # Shape: [num_antennas, num_antennas, time_samples, fft_size]

                averaged = visibilities.mean(dim=2)  # Average over time samples
            else:
                visibilities = np.einsum(
                    'iaf,jaf->ijaf',
                    integration_buffer,
                    np.conj(integration_buffer)
                ) # Shape: [num_antennas, num_antennas, time_samples, fft_size]

                averaged = visibilities.mean(axis=2)  # Average over time samples

            # Copy to output_items - output_items is a list of arrays, one per baseline
            # This is where we write the final averaged visibility data to the output
            # Each output_items[baseline_idx] is an array of shape [max_output_samples, fft_size]
            # In the documentation of GNU Radio, output_items is structured such that max_output_samples
            # is preassigned by the scheduler. Here we only produce one output sample per baseline
            # per work call when integration is complete - thus filling output_items[baseline_idx][0].
            # I know this is a bit confusing, but that's how GNU Radio works apparently.
            # Note: Because of this I don't think vectorizing this copy is not worth it.
            for baseline_idx in range(self.num_baselines):
                output_items[baseline_idx][num_output_samples] = averaged[baseline_idx]
                #                                  ↑
                #                    This is 0, so we're filling slot [0]

            # Increment output sample count
            # Set num_output_samples to 1 since we produced one integrated output sample
            # This tells GNU Radio that we have produced one output sample for each baseline.
            # Reset integration buffers for next round
            num_output_samples += 1
            self.output_count += 1
            self.reset_integration_buffers()

        # Report every 30 seconds
        if current_time - self.last_report_time > 30:
            self.print_stats()

        return num_output_samples
    
    def print_stats(self):
        """
        Print statistics about processing rates and counts
        """
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        self.last_report_time = current_time
        if elapsed > 0:
            # processed_samples = self.sample_count * self.fft_size * self.num_antennas
            total_output_samples = self.output_count * self.fft_size * self.num_baselines
            print(f"\n=== VisibilityCorrelator Statistics (t={elapsed:.1f}s) ===")
            print(f"Total input samples processed: {self.total_samples_processed}")
            print(f"Total output samples produced: {total_output_samples}")
            print("=" * 50)

class VisibilityFileSink(gr.sync_block):
    """
    Simple file sink for visibility data
    """
    def __init__(self, fft_size=1024, num_antennas=9, data_folder="mock_data", use_gpu=True, flush_interval=10):
        # Calculate number of baselines
        num_baselines = num_antennas * (num_antennas + 1) // 2

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
        
        # Store parameters
        self.fft_size = fft_size
        self.num_baselines = num_baselines
        self.num_antennas = num_antennas
        self.data_folder = data_folder
        self.flush_interval = flush_interval
        
        # Create files with the naming convention baseline_i_j.bin
        # Make sure that the way we open files is compatible with appending binary data
        # VERY IMPORTANT: This for loop needs to be the same as in
        # the correlator to ensure correct baseline ordering
        self.file_handles = []
        for i in range(num_antennas):
            for j in range(i, num_antennas):
                filename = os.path.join(data_folder, f"baseline_{i}_{j}.bin")
                self.file_handles.append(open(filename, 'ab'))
        
        # Add monitoring
        self.samples_written = 0
        self.bytes_written = 0
        self.write_count = 0
        self.start_time = None
        self.last_report_time = 0

        self._print_visibility_info()

    def _print_visibility_info(self):
        """Print configuration info"""
        info = "VisibilityFileSink Configuration:"
        info += f"\n  FFT Size: {self.fft_size}"
        info += f"\n  Number of Baselines: {self.num_baselines}"
        info += f"\n  Data Folder: {self.data_folder}"
        info += f"\n  Flush Interval: {self.flush_interval} samples"
        print_box(info)

    def work(self, input_items, output_items):
        """Write visibility data to files"""
        # Monitoring - start time
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        # Number of input samples available - all baselines have the same number of samples
        # The shape of input_items is like: [num_baselines][num_samples, fft_size]
        num_samples = len(input_items[0])

        # We assume that num_samples is usually 1 because the correlator outputs
        # one integrated sample at a time. However, we handle the general case.
        # If the Correlator outputs multiple samples at once this is very unexpected behavior
        # as it will mean that the Correlator executed multiple times before the sink.
        if num_samples == 1:
            # Fast path for single sample
            # Loop over baselines and write the sample to the corresponding file
            for baseline_idx in range(self.num_baselines):
                data = input_items[baseline_idx][0]  # Get the single sample
                bytes_to_write = data.tobytes()
                self.file_handles[baseline_idx].write(bytes_to_write)
                self.bytes_written += len(bytes_to_write)
        else:
            # General path for multiple samples
            # This should not happen but I have included it for completeness
            print("⚠️ VisibilityFileSink: Warning - multiple samples received in one work call ⚠️")
            print("This is unexpected behavior. Proceeding to write all samples. Maybe the Correlator executed twice before the sink?")
            for sample_idx in range(num_samples):
                for baseline_idx in range(self.num_baselines):
                    data = input_items[baseline_idx][sample_idx]
                    bytes_to_write = data.tobytes()
                    self.file_handles[baseline_idx].write(bytes_to_write)
                    self.bytes_written += len(bytes_to_write)

        # Update samples written count
        self.samples_written += num_samples * self.fft_size * self.num_baselines
        self.write_count += num_samples
        
        # Flush files periodically
        self._flush()

        # Report every 30 seconds
        if current_time - self.last_report_time > 30.0:
            self.print_stats()
            self.last_report_time = current_time
        
        return num_samples
    
    def print_stats(self):
        """Print final statistics"""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        if elapsed > 0:
            print(f"\n=== VisibilityFileSink Last Statistics (t={elapsed:.1f}s) ===")
            print(f"Total samples written: {self.samples_written:.2e}")
            print(f"Average sample rate: {self.samples_written / elapsed:.2f} samples/sec")
            print(f"Average data rate: {(self.bytes_written / elapsed) / (1024*1024):.2f} MB/sec")
            print(f"Total data written: {self.bytes_written / (1024*1024):.2f} MB")
            print("=" * 50)
    
    def stop(self):
        """Called when flowgraph stops - close files"""
        for fh in self.file_handles:
            fh.flush() # Clear Python's internal buffer - pushes data to OS
            os.fsync(fh.fileno()) # Clear OS's internal buffer - pushes data to disk
            if not fh.closed:
                fh.close()
        return True
    
    def _flush(self):
        """
        Periodically flushes buffered data to ensure data integrity.

        This method performs two levels of flushing:
        1. Every `flush_interval` writes, it calls `flush()` on all open file handles
        to push Python's internal I/O buffers to the operating system.
        2. Every `flush_interval * 10` writes, it additionally calls `os.fsync()`
        to force the operating system to write its cached data to the physical disk.

        This strategy provides a balance between performance and data safety:
        frequent lightweight flushes minimize data loss on crashes, while
        less frequent `fsync()` calls ensure persistence to disk without
        excessive I/O overhead.
        """
        if self.write_count % self.flush_interval == 0:
            for fh in self.file_handles:
                fh.flush() # Clear Python's internal buffer - pushes data to OS
                
        if self.write_count % (self.flush_interval * 10) == 0:
            for fh in self.file_handles:
                os.fsync(fh.fileno()) # Clear OS's internal buffer - pushes data to disk


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

        # Create Visibility Correlator Block with built-in integration
        integration_samples = int(integration_time * (sampling_rate / fft_size)) # T_I * (Fs / N)
        self.cross_correlator = VisibilityCorrelator(
            fft_size=fft_size, 
            num_antennas=self.num_antennas,
            integration_samples=integration_samples,
            use_gpu=True
        )

        # Create Visibility File Sink Block (connect directly to correlator)
        self.visibility_file_sink = VisibilityFileSink(
            fft_size=fft_size,
            num_antennas=self.num_antennas,
            data_folder=folder_path,
            use_gpu=True,
            flush_interval=100
        )

        ##################################################
        # Connections
        ##################################################

        # Connect each airspy -> stream_to_vector -> fft
        self.connect_airspy_to_fft(self.airspy_devices, self.stream_to_vector_blocks, self.fft_blocks, test=test)

        # Connect all FFT outputs to correlator inputs
        self.connect_fft_to_correlator(self.fft_blocks, self.cross_correlator)

        # Connect correlator directly to file sink (skip separate integrator)
        self.connect_correlator_to_file_sink(self.cross_correlator, self.visibility_file_sink)

        self._print_interferometer_info()

    def connect_correlator_to_file_sink(self, correlator_block, file_sink_block):
        """Connect correlator directly to file sink"""
        num_baselines = self.num_antennas * (self.num_antennas + 1) // 2
        for i in range(num_baselines):
            self.connect(
                (correlator_block, i),
                (file_sink_block, i)
            )

    def _print_interferometer_info(self):
        info = "Interferometer Configuration:"
        info += f"\n  Number of Antennas: {self.num_antennas}"
        info += f"\n  Sampling Rate: {self.sampling_rate/1e6} MHz"
        info += f"\n  Frequency: {self.frequency/1e6} MHz"
        info += f"\n  FFT Size: {self.fft_size}"
        info += f"\n  Integration Time: {self.integration_time} seconds"
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

    def connect_fft_to_correlator(self, fft_blocks, correlator_block):
        for i in range(self.num_antennas):
            self.connect(
                (fft_blocks[i], 0),
                (correlator_block, i),
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


def main(top_block_cls=Interferometer, options=None, test=False):
    qapp = Qt.QApplication(sys.argv)
    folder_name = "mock_data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    tb = top_block_cls(
        sampling_rate=SAMPLING_RATE,
        integration_time=INTEGRATION_TIME,
        frequency=CENTER_FREQUENCY,
        fft_size=FFT_SIZE,
        num_antennas=len(DEVICE_LIST),
        folder_path=folder_name,
        test=test
    ) 
    
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        print("\nShutting down...")
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