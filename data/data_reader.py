'''
The DataReader class handles reading data for the Paint Can Interferometer project.

It assumes that data files are stored in a directory specified by the DATA_FOLDER environment variable.
The way the data is organized within this folder is as follows:
    - Each observation session has its own subfolder named with a timestamp session_yyyymmdd_hhmmss (e.g., session_20231015_153000).
    - Within each session folder, data files are named according to the following convention:
    antenna_<antenna_number>_date_<YYYYMMDD>_time_<HHMMSS>
    - Data files are stored in raw binary format.

    Schematically:
    DATA_FOLDER/
        session_20231015_153000/
            antenna_1_date_20231015_time_153000
            antenna_2_date_20231015_time_153000
        session_20231016_101500/
            antenna_1_date_20231016_time_101500
            antenna_2_date_20231016_time_101500

The DataReader class provides methods to:
    - Load data from a specified session folder.
    - List all available session folders in the DATA_FOLDER.
'''
import os
from typing import Union
import numpy as np
from pathlib import Path
from logging import log

from dotenv import load_dotenv
import warnings

from .obs_md import ObservationMetadata
from .utils import reshape_fft_data

from utils import print_box

class DataReader:
    """
    Class to read data for the Paint Can Interferometer project.
    Assumes data files are stored in a directory specified by the DATA_FOLDER
    environment variable. For each observation session, there is a subfolder
    named with a timestamp (e.g., session_20231015_153000). Within each session
    folder, data files are named according to the convention:
    antenna_<antenna_number>_channels_<number_of_channels>_date_<YYYYMMDD>_time_<HHMMSS>.npy
    """
    def __init__(self):
        """
        Initialize the DataReader by loading environment variables and setting up paths.
        Expects the following environment variables to be set:
            - NUMBER_OF_ANTENNAS: Total number of antennas in the interferometer array.
            - DATA_FOLDER: Path to the folder containing data files.
        """
        load_dotenv()
        self.num_antennas = int(os.getenv("NUMBER_OF_ANTENNAS"))
        self.base_data_folder = Path(os.getenv("DATA_FOLDER"))
        if not self.base_data_folder.exists():
            raise FileNotFoundError(f"Data folder {self.base_data_folder} does not exist.")

        self.session_folders = self._get_session_folders()

    def get_data(self, session_folder: str, fft_size: Union[None, int] = None) -> dict[int, np.ndarray]:
        """
        Read data files from a specified session folder.

        :param session_folder: Name of the session folder to read data from.
        Must be one of the folders listed by the `list_sessions` method.
        :type session_folder: str
        :return: Dictionary mapping antenna numbers to their corresponding data arrays.
        :rtype: dict[int, np.ndarray]
        """
        session_path = self.base_data_folder / session_folder
        if not session_path.exists() or not session_path.is_dir():
            raise FileNotFoundError(
                f"Session folder {session_path} does not exist or is not a directory."
                f" Available sessions: {self.list_sessions()}"
            )

        data = {}
        for antenna_num in range(1, self.num_antennas + 1):
            file_pattern = f"antenna_{antenna_num}_*"
            files = list(session_path.glob(file_pattern))
            if not files:
                warnings.warn(
                    f"No data file found for antenna {antenna_num} in session {session_folder}.",
                    UserWarning
                )
                continue
            if len(files) > 1:
                warnings.warn(
                    f"Multiple data files found for antenna {antenna_num} in session {session_folder}."
                    f" Using the first one.",
                    UserWarning
                )

            # Assume data is stored in raw binary format as float32
            warnings.warn(
                f"Assuming data file {files[0]} is in raw binary format with float32 data type.",
                UserWarning
            )
            data_array = np.fromfile(files[0], dtype=np.float32)
            data[antenna_num] = data_array

            # Reshape data if fft_size is provided or can be obtained from metadata
            if fft_size is not None:
                data[antenna_num] = reshape_fft_data(data_array, fft_size=fft_size)
                print_box(f"Reshaped data for antenna {antenna_num} using the provided fft_size argument into segments of size {fft_size}.")
            else:
                try:
                    metadata = self.get_metadata(session_folder)
                    fft_size = metadata.fft_size

                    data[antenna_num] = reshape_fft_data(data_array, fft_size=fft_size)
                    print_box(f"Reshaped data for antenna {antenna_num} using fft_size from metadata into segments of size {fft_size}.")
                except:
                    info = f"Failed to reshape data for antenna {antenna_num} using fft_size from metadata."
                    info += "\nSince no fft_size argument was provided, returning raw data array."
                    print_box(info)

        return data
    
    def get_metadata(self, session_folder: str) -> ObservationMetadata:
        """
        Read metadata from a specified session folder.

        :param session_folder: Name of the session folder to read metadata from.
        Must be one of the folders listed by the `list_sessions` method.
        :type session_folder: str
        :return: Dictionary containing metadata information.
        :rtype: dict
        """
        session_path = self.base_data_folder / session_folder
        metadata_file = session_path / "observation_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file {metadata_file} does not exist in session {session_folder}."
            )
        
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        metadata = ObservationMetadata.from_dict(metadata)
        
        return metadata

    def list_sessions(self) -> list[str]:
        """
        List all available session folders in the base data folder.
        
        :return: List of session folder names.
        :rtype: list[str]
        """
        return [folder.name for folder in self.session_folders]
        
    def _get_session_folders(self) -> list[Path]:
        """
        Get all the session subfolders in the base data folder.
        Each session folder is expected to be named with the convention:
        session_YYYYMMDD_HHMMSS
        
        :return: List of Path objects representing valid session subfolders.
        :rtype: list[Path]
        """
        # Check if the base data folder is a directory
        if not self.base_data_folder.is_dir():
            raise NotADirectoryError(f"{self.base_data_folder} is not a directory.")

        # Get all the subfolders in the base data folder
        subfolders = []

        # Iterate through the items in the base data folder
        # Make sure they are all directories
        # Warn if any folder does not match the expected naming convention: session_YYYYMMDD_HHMMSS
        # Then check if they contain the expected number of .npy files
        # If not, warn and skip those folders
        for f in self.base_data_folder.iterdir():
            if f.is_dir():
                if f.name.startswith("session_") and len(f.name) == 23:  # e.g., session_20231015_153000
                    files = list(f.glob("*")) # Get all the files in the subfolder

                    if not files: # No files found
                        warnings.warn(
                            f"Session folder {f} does not contain any files. "
                            f"Skipping.",
                            UserWarning
                        )
                        continue
                    if len(files) < self.num_antennas: # Fewer files than expected antennas
                        warnings.warn(
                            f"Session folder {f} contains fewer files ({len(files)})"
                            f" than expected antennas ({self.num_antennas}). "
                            f"Skipping.",
                            UserWarning
                        )
                        continue
                    
                    # If all checks passed, add the folder to the list
                    subfolders.append(f)
                else: # Warn about unexpected folder naming convention
                    warnings.warn(
                        f"Folder {f} does not match the expected session naming convention. "
                        f"Expected format: session_YYYYMMDD_HHMMSS. Got: {f.name}. "
                        f"Skipping.",
                        UserWarning
                    )
            else:
                warnings.warn(
                    f"Item {f} in data folder is not a directory. Skipping.",
                    UserWarning
                )

        # If no valid subfolders found, raise an error
        if not subfolders:
            raise FileNotFoundError(
                f"No valid session subfolders found in data folder {self.base_data_folder}."
            )
        
        return subfolders
    