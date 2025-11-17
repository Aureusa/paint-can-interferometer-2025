# Standard library imports
import time
import os
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
import numpy as np
import torch
from tqdm import tqdm

# Local imports
from utils import print_box
from data import read_file


load_dotenv()


class GPUCorrelator:
    """
    Custom GNU Radio block that computes visibility matrix AND integrates over time
    """
    def __init__(self, session_folder: str):
        # Device setup
        self.device = torch.device('cuda')
        if self.device.type != 'cuda':
            raise RuntimeError("GPUCorrelator requires a CUDA-capable GPU, but none was found.")
        
        # Get the variable from the environment variable
        self.base_data_folder = Path(os.getenv("DATA_FOLDER"))
        self.session_folder = self.base_data_folder / session_folder
        self.fft_size = int(os.getenv("FFT_SIZE", "1024"))  # Default to 1024 if not set
        self.num_antennas = int(os.getenv("NUM_ANTENNAS", "9"))  # Default to 9 if not set

        # Data Loading Params
        self.nr_chunks = int(os.getenv("NUM_CHUNKS", "5"))  # Number of chunks to read
        self.time_samples = int(os.getenv("TIME_SAMPLES", "10000"))

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
        self.baseline_i_indices = torch.tensor(baseline_i_indices, device=self.device)
        self.baseline_j_indices = torch.tensor(baseline_j_indices, device=self.device)

        # Initialize buffers
        self._reset_integration_buffer()
        
        self.estimated_total_samples = self._estimate_total_samples()

        self._print_correlator_info()

    def run(self):
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
                file_path = self.session_folder / f"antenna_{ant}.dat"
                data_chunk, start_idx = read_file(
                    filename=file_path,
                    chunks=self.nr_chunks,
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

        # Save visibilities to disk
        self._save_visibilities(visibilities)
    
    def _preprocess_data(self, input_items):
        # Shape: [num_antennas, time_samples, fft_size]
        inp_tensor = torch.tensor(input_items, dtype=torch.complex64, device=self.device)
        arr_shape = inp_tensor.shape
        time_samples = arr_shape[1]
        return inp_tensor, time_samples

    def _populate_buffer(self, proccessed_data, time_samples):
        # Add to self.integration_buffer
        self.integration_buffer += proccessed_data.sum(dim=1, keepdim=True)  # Sum over time samples
        self.integration_count += time_samples

        # Update progress bar
        if self.progress_bar:
            self.progress_bar.update(time_samples)
            self.progress_bar.set_postfix({
                'Integration Count': f'{self.integration_count:,}',
                'Rate': f'{time_samples/(time.time() - self.start_time):.1f} samples/s'
            })

    def _should_continue(self):
        # Placeholder for actual logic to determine if processing should continue
        return True

    def _average_buffer(self):
        self.integration_buffer /= self.integration_count

    def _compute_visibilities(self):
        fft_matrix = self.integration_buffer  # Shape: [num_antennas, 1, fft_size]

        # Get rid of the time dimension for visibility computation
        fft_matrix = fft_matrix.squeeze(1)  # Shape: [num_antennas, fft_size]

        # Compute visibilities V_ij = F_i * conj(F_j) for all antenna pairs
        visibilities = torch.einsum(
            'if,jf->ijf',
            fft_matrix,
            torch.conj(fft_matrix)
        ) # Shape: [num_antennas, num_antennas, fft_size]
        return visibilities
    
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
        self.integration_buffer = torch.zeros(
            (self.num_antennas, 1, self.fft_size), 
            dtype=torch.complex64,
            device=self.device
        )
        self.integration_count = 0

    def _estimate_total_samples(self):
        """Estimate total samples based on first antenna file size"""
        try:
            first_file = self.session_folder / "antenna_0.dat"
            if first_file.exists():
                file_size = first_file.stat().st_size
                samples_per_file = file_size // (np.dtype(np.complex64).itemsize)
                return samples_per_file
        except:
            pass
        return 1000000  # Default fallback
    