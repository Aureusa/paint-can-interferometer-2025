import numpy as np
from typing import Union

from validation import validate_dtype

def observation_time(
        data: np.ndarray,
        sample_rate: Union[float, int],
        integration_time: Union[float, int]
) -> float:
    """
    Calculate the total observation time in seconds.
    
    :data: numpy array of the data points the data is assumed to be a real valued time series
    :sample_rate: sampling rate in Hz (s^-1)
    :integration_time: integration time in seconds
    """
    validate_dtype(data, np.complex64, "data")
    validate_dtype(data, np.float32, "data")
    return len(data) / sample_rate * integration_time

def reshape_fft_data(data: np.ndarray, fft_size: int = 1024) -> np.ndarray:
    """
    Reshape the data into a 2D array where each row is an FFT segment.
    
    :data: numpy array of the data points the data is assumed to be a real valued time series
    :fft_size: size of each FFT segment
    """
    n_segments = len(data) // fft_size

    assert len(data) % fft_size == 0, f"Data length {len(data)} is not a multiple of fft_size {fft_size}. Are you sure that this is the correct fft_size?"
    
    reshaped_data = data.reshape(n_segments, fft_size)
    return reshaped_data
