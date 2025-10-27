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
