# This folder contains scripts for running the interferometer using GNU Radio.

# Configuring GNU Radio


# For Linux:
1. Install GNU Radio and the osmosdr module using:
```bash
sudo apt-get install gnuradio
sudo apt-get install gr-osmosdr
```
Now we hope that the base python GNU Radio bindings are installed. To check, run:
```bash
python3 -c "import gnuradio"
```
If you get no errors, the installation was successful. If you do get errors, we are screwed...

2. Create a virtual environment and activate it:
```bash
python3 -m venv gnuradio_env
source gnuradio_env/bin/activate
```
3. Veryify the installation:
```bash
python3 -c "import gnuradio; print('GNU Radio version:', gnuradio.version())"
```

# For Windows:
1. Download and install **radioconda** from [this link](https://github.com/ryanvolz/radioconda/releases/latest)
   - Choose the Windows x86_64 installer
   - Follow the installation wizard

2. Open **Anaconda Prompt (radioconda)** from the Start menu

3. Create a new environment with GNU Radio:
```bash
conda create -n gnuradio_env gnuradio gr-osmosdr
```

4. Activate the environment:
```bash
conda activate gnuradio_env
```

5. Verify the installation:
```bash
python -c "import gnuradio; print('GNU Radio version:', gnuradio.version())"
```
If you get no errors, the installation was successful. If you do get errors, we are screwed...

# For macOS:
1. Install Homebrew if you haven't already:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install GNU Radio using Homebrew:
```bash
brew install gnuradio
```

3. Install gr-osmosdr:
```bash
brew install gr-osmosdr
```

4. Create and activate a virtual environment:
```bash
python3 -m venv gnuradio_env
source gnuradio_env/bin/activate
```

5. Verify the installation:
```bash
python3 -c "import gnuradio; print('GNU Radio version:', gnuradio.version())"
```
If you get no errors, the installation was successful. If you do get errors, we are screwed...

# Running the Interferometer

1. Make sure to update the `observation_config.yaml` file with the correct parameters for your setup, such as frequency, sample rate, and gain settings. Also uncoment the airspys connected to your laptop.
```bash
# Change this to the time you want to start observing. For example if it is 2025-11-14T12:00:00+01:00, set it at least to 1 minute later 2025-11-14T12:01:00+01:00. This is done so that we have the same start time accross all laptops
start_time: 2025-11-14T16:44:15+01:00 # YYYY-MM-DDTHH:MM:SS+TZ

# Observation duration in seconds - Currently set to 10 minutes
observation_duration: 600  # in seconds

# List of AirSpy devices
# Make sure to uncoment the Airspys connected to your computer
device_list: [
  # airspy=1,
  # airspy=2,
  # airspy=3,
  # airspy=4,
  # airspy=5,
  # airspy=6,
  # airspy=7,
  # airspy=8,
  # airspy=9,
]

# A folder to store the recorded data
data_storage_path: mock_data

###########################################
#   THIS IS GENERALLY NOT TO BE CHANGED   #
###########################################

# Sampling rate for the observation
sampling_rate: 10000000 #10e6 in Hz

# Center frequency for the observation
center_frequency: 1420000000 #1.42e9 in Hz
```

2. Navigate to the main directory `cd` command and run the `main.py` file.
```bash
cd paint-can-interferometer-2025
python3 main.py
```
