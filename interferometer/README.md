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

3. Check the CUDA drivers of your system to ensure GPU support:
```bash
nvidia-smi
```
You should get a readout of your GPU status which looks something like this:
```+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.6     |
|-------------------------------+----------------------+----------------------+
```

4. Install PyTorch with CUDA support. Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the appropriate version for your system. For example for CUDA 12.6, you might run:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
```
verify that you use the correct index url for your CUDA version!

5. Verify the PyTorch installation with GPU support:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
If it prints `True`, PyTorch is correctly set up with GPU support.

6. Navigate to the directory where `interferometer_fast.py` is located using the `cd` command.
7. Run the script using:
```bash
python3 interferometer_fast.py
```

# For Windows:
1. Download and install the `radioconda` from this [link](https://glare-sable.vercel.app/radioconda/radioconda-installer/radioconda-.*-Windows-x86_64.exe).
2. Open the `Anaconda Prompt` from the start menu by searchin `radioconda`.
3. Create a new environment with GNU Radio:
   ```bash
   conda create -n gnuradio_env gnuradio
   ```
4. Activate the environment:
   ```bash
   conda activate gnuradio_env
   ```
5. Verify the installation by running:
   ```bash
   python -c "import gnuradio"
   ```
If you get no errors, the installation was successful. If you do get errors, we are screwed...

6. Check the CUDA drivers of your system to ensure GPU support:
```bash
nvidia-smi
```
You should get a readout of your GPU status which looks something like this:
```+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.6     |
|-------------------------------+----------------------+----------------------+
```

7. Install PyTorch with CUDA support. Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the appropriate version for your system. For example for CUDA 12.6, you might run:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
```
verify that you use the correct index url for your CUDA version!

8. Verify the PyTorch installation with GPU support:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
If it prints `True`, PyTorch is correctly set up with GPU support.

9. Navigate to the directory where `interferometer_fast.py` is located using the `cd` command.
10. Run the script using:
```bash
python3 interferometer_fast.py
```
