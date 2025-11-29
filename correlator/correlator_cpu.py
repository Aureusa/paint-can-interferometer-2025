import time

import numpy as np
from tqdm import tqdm

from utils import print_box
from data import read_file, reshape_data


class CPUCorrelator:
    """
    Custom GNU Radio block that computes visibility matrix AND integrates over time
    """
    def __init__(self, files: str, nr_chunks: int, time_samples: int, fft_size: int, num_antennas: int):      
        self.fft_size = fft_size
        self.num_antennas = num_antennas

        # Data Loading Params
        self.nr_chunks = nr_chunks  # Number of chunks to read
        self.time_samples = time_samples
        self.files = files

        # Calculate number of baselines
        self.num_baselines = self.num_antennas * (self.num_antennas + 1) // 2
        
        # Pre-compute indices for extracting upper triangle - N(N+1)/2 baselines
        # Add to self.baseline_i_indices and self.baseline_j_indices for vectorized access
        self.baseline_indices = []
        baseline_i_indices = []
        baseline_j_indices = []
        for i in range(self.num_antennas):
            for j in range(i, self.num_antennas):
                self.baseline_indices.append((i, j))
                baseline_i_indices.append(i)
                baseline_j_indices.append(j)
        self.baseline_i_indices = np.array(baseline_i_indices)
        self.baseline_j_indices = np.array(baseline_j_indices)

        # Initialize buffers
        self._reset_integration_buffer()
        
        self.estimated_total_samples = self._estimate_total_samples()

        self._print_correlator_info()

    def run(self):
        self.start_time = time.time()
        # Initialize progress bar
        self.progress_bar = tqdm(
            total=self.estimated_total_samples,
            desc="Processing samples",
            unit="samples",
            unit_scale=True
        )

        keep_going = True
        start_idx = 0
        while keep_going:
            input_data = []
            ##################################################################
            # This definately needs a revist. It does not handle the size    #
            # of different files properly. Depends on the correlating schema #
            ##################################################################
            for ant in range(self.num_antennas):
                file_path = self.files[ant]
                data_chunk, start_idx = read_file(
                    filename=file_path,
                    chunks=1,
                    chunk_size=self.time_samples * self.fft_size, # Number of samples per chunk (multiple of fft_size)
                    start_idx=start_idx,
                    return_last_idx=True,
                    verbose=False # We want to avoid cluttering output
                )
                input_data.append(data_chunk)

            # Assuming we have collected the data properly for all antennas
            # Process data
            proccessed_data, time_samples = self._preprocess_data(input_data)
            
            # Populate buffer
            self._populate_buffer(proccessed_data, time_samples)

            # Check if we should continue processing
            keep_going = self._should_continue()

        # Once done, average the buffer
        self._average_buffer()

        # Compute visibilities
        visibilities = self._compute_visibilities()

        return visibilities

        # Save visibilities to disk
        # self._save_visibilities(visibilities)
    
    def _preprocess_data(self, input_items):
        # Shape: [num_antennas, time_samples, fft_size]
        inputs = []
        for inp in input_items:
            reshaped_inp = reshape_data(inp, self.fft_size)
            inputs.append(reshaped_inp)
        input_items = np.stack(inputs, axis=0)  # Shape: [num_antennas, time_samples, fft_size]

        # FFT each time sample to convert to frequency domain
        input_items = np.fft.fft(input_items, axis=2).astype(np.complex64)  # FFT along fft_size dimension

        arr_shape = input_items.shape
        time_samples = arr_shape[1]
        return input_items, time_samples

    def _populate_buffer(self, proccessed_data, time_samples):
        # Add to self.integration_buffer
        self.integration_buffer += proccessed_data.sum(axis=1, keepdims=True)  # Sum over time samples
        self.integration_count += time_samples
        self.num_samples_processed += time_samples * self.fft_size * self.num_antennas

        # Update progress bar
        if self.progress_bar:
            self.progress_bar.update(self.num_samples_processed - self.progress_bar.n)
            self.progress_bar.set_postfix({
                'Integration Count': f'{self.integration_count:,}',
                'Processed': f'{self.num_samples_processed:,}/{self.estimated_total_samples:,}'
            })

    def _should_continue(self):
        # Placeholder for actual logic to determine if processing should continue
        if self.num_samples_processed == self.estimated_total_samples:
            if self.progress_bar:
                self.progress_bar.close()
            return False
        return True
        #return self.num_samples_processed < self.estimated_total_samples

    def _average_buffer(self):
        self.integration_buffer /= self.integration_count

    def _compute_visibilities(self):
        fft_matrix = self.integration_buffer  # Shape: [num_antennas, 1, fft_size]

        # Get rid of the time dimension for visibility computation
        fft_matrix = fft_matrix.squeeze(1)  # Shape: [num_antennas, fft_size]

        # IT IS POSSIBLE TO VECTORIZE THIS FURTHER IN THIS WAY:
        # # Only compute upper triangle baselines (much faster than full matrix)
        # # Extract baseline pairs using precomputed indices
        # i_data = fft_size[self.baseline_i_indices]  # Shape: [num_baselines, fft_size]
        # j_data = fft_size[self.baseline_j_indices]  # Shape: [num_baselines, fft_size]
        
        # # Vectorized visibility computation
        # visibilities = i_data * np.conj(j_data)  # Shape: [num_baselines, fft_size]

        # Compute visibilities V_ij = conj(F_i) * F_j for all antenna pairs
        visibilities = np.einsum(
            'if,jf->ijf',
            np.conj(fft_matrix),  # Conjugate first signal (like scipy)
            fft_matrix
        ) # Shape: [num_antennas, num_antennas, fft_size]

        # Convert to time domain
        time_domain_corr = np.fft.ifft(visibilities, axis=2)
        
        # Shift to center zero-lag
        time_domain_corr = np.fft.fftshift(time_domain_corr, axes=2)
        return visibilities, time_domain_corr
    
    def _save_visibilities(self, visibilities):
        output_folder = self.session_folder / "visibilities"
        output_folder.mkdir(exist_ok=True)

        for _, (i, j) in enumerate(self.baseline_indices):
            baseline_vis = visibilities[i, j, :].cpu().numpy()  # Shape: [fft_size]
            output_path = output_folder / f"baseline_{i}_{j}.npy"
            np.save(output_path, baseline_vis)
    
    def _print_correlator_info(self):
        """Print configuration info"""
        info = "VisibilityCorrelator Configuration:"
        info += f"\n  Number of Antennas: {self.num_antennas}"
        info += f"\n  Number of Baselines: {self.num_baselines}"
        info += f"\n  FFT Size: {self.fft_size}"
        info += f"\n  Number of Chunks: {self.nr_chunks}"
        info += f"\n  Time Samples per Chunk: {self.time_samples}"
        info += f"\n  Total samples per Chunk: {self.time_samples * self.fft_size}"
        info += f"\n  Total samples per Read: {self.nr_chunks * self.time_samples * self.fft_size}"
        info += f"\n  Estimated Total Samples: {self.estimated_total_samples}"
        print_box(info)

    def _reset_integration_buffer(self):
        """Reset integration buffers"""
        self.integration_buffer = np.zeros(
            (self.num_antennas, 1, self.fft_size), 
            dtype=np.complex64
        )
        self.integration_count = 0
        self.num_samples_processed = 0

    def _estimate_total_samples(self):
        """Estimate total samples based on first antenna file size"""
        total_samples = self.num_antennas * self.nr_chunks * self.time_samples * self.fft_size
        return total_samples
    