import numpy as np
from tqdm import tqdm
import gc

from utils import print_box


def read_file(
        filename: str,
        chunks: int,
        chunk_size: int = int(10e6),
        start_idx: int = 0,
        return_last_idx: bool = False,
        verbose: bool = True
    ) -> np.ndarray:
    """
    Read chunks of data from a binary file using memory mapping.
    
    :param filename: Path to the binary data file.
    :type filename: str
    :param chunks: Number of chunks to read.
    :type chunks: int
    :param chunk_size: Number of samples per chunk. Default is 10 million.
    :type chunk_size: int
    :param start_idx: Starting index in the file to begin reading.
    :type start_idx: int
    :param return_last_idx: Whether to return the last index read.
    :type return_last_idx: bool
    :param verbose: Whether to print detailed information.
    :type verbose: bool
    :return: Numpy array of the read data. If return_last_idx is True,
    returns a tuple (np.ndarray, int) with the data and the last index read.
    :rtype: np.ndarray or (np.ndarray, int)
    """
    # Memory map the file
    data = np.memmap(filename, dtype=np.complex64, mode='r')
    total_samples = len(data)

    # Check if start_idx is valid
    if start_idx >= total_samples:
        if verbose:
            print(f"⚠️  start_idx ({start_idx:,}) is at or beyond file end ({total_samples:,})")
        if return_last_idx:
            return np.array([]), start_idx
        else:
            return np.array([])
    
    # Calculate available samples from start_idx
    available_samples = total_samples - start_idx
    requested_samples = chunk_size * chunks
    
    if verbose:
        info = f"File: {filename}"
        info += f"\nTotal samples: {total_samples:,}"
        info += f"\nStarting at index: {start_idx:,}"
        info += f"\nAvailable samples: {available_samples:,}"
        info += f"\nRequested samples: {requested_samples:,}"
        
        if available_samples < requested_samples:
            info += f"\n - Will read to EOF ({available_samples:,} samples) -"
        else:
            info += f"\nReading {chunks} full chunks of size {chunk_size:,} samples each"
        print_box(info)
    
    chunk_start = start_idx
    collected_data = []
    chunks_read = 0
    for _ in tqdm(range(chunks), disable=not verbose, desc="Reading chunks"):
        chunk_end = min(chunk_start + chunk_size, total_samples) # Ensure we don't go past EOF
            
        # Read the chunk
        data_chunk = data[chunk_start:chunk_end]
        collected_data.append(data_chunk)
        chunks_read += 1
        
        chunk_start = chunk_end # Move to next chunk start
        
        # If we've reached the end of file, break
        if chunk_start >= total_samples:
            break

    # Concatenate all collected data
    collected_data_arr = np.concatenate(collected_data, dtype=np.complex64)
    
    # Clean up and garbage collect
    del collected_data
    gc.collect()

    if verbose:
        info = "Finished reading."
        info += f"\nChunks requested: {chunks}"
        info += f"\nChunks actually read: {chunks_read}"
        info += f"\nTotal samples read: {len(collected_data_arr):,}"
        info += f"\nLast index: {chunk_start:,}"
        if chunk_start >= total_samples:
            info += " (EOF)"
        print_box(info)

    if return_last_idx:
        return collected_data_arr, chunk_start
    else:
        return collected_data_arr
    
def save_file(
    filename: str,
    data: np.ndarray,
    verbose: bool = True
    ) -> None:
    """
    Save data to a binary .dat file.
    
    :param filename: Path to the output binary file.
    :type filename: str
    :param data: Numpy array of data to save.
    :type data: np.ndarray
    :param verbose: Whether to print detailed information.
    :type verbose: bool
    """
    # Ensure filename has .dat extension
    if not filename.endswith('.dat'):
        filename += '.dat'
    
    data.astype(np.complex64).tofile(filename)
    
    if verbose:
        info = f"Data saved to {filename}"
        info += f"\nTotal samples saved: {data.size:,}"
        print_box(info)

def create_smaller_copy(
    input_filename: str,
    output_filename: str,
    num_samples: int,
    verbose: bool = True
    ) -> None:
    """
    Create a smaller copy of a binary data file by reading a specified number of samples.
    Start from the beginning of the file.
    
    :param input_filename: Path to the input binary data file.
    :type input_filename: str
    :param output_filename: Path to the output binary data file.
    :type output_filename: str
    :param num_samples: Number of samples to read and save to the new file.
    :type num_samples: int
    :param verbose: Whether to print detailed information.
    :type verbose: bool
    """
    # Memory map the input file
    data = np.memmap(input_filename, dtype=np.complex64, mode='r')
    total_samples = len(data)
    
    if num_samples > total_samples:
        num_samples = total_samples
        if verbose:
            print(f"⚠️  Requested num_samples exceeds total samples. Adjusted to {total_samples:,}.")
    
    # Read the specified number of samples
    data_to_save = data[:num_samples]
    
    # Save to the output file
    save_file(output_filename, data_to_save, verbose=verbose)
