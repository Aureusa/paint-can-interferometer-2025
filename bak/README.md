# This folder contains scripts for running the interferometer using GNU Radio. This serves as a backup if the designated laptop fails.

This is the backup of the original antennagnu.py file before refactoring into modules.

 - `antennagnu_no_metadata.py` is the version without metadata management for simpler testing and debugging.
 - `antennagnu_with_metadata.py` includes metadata management for the observation sessions.

In practice both versions should function identically, with the only difference being the handling of metadata. This means that both scripts should produce the same output data files when run under the same conditions.

Both scripts are intended to be run in an environment where GNU Radio is installed and properly configured. They rely on external configuration files (e.g., `observation_conf.yaml`) to set up the parameters for the interferometer. Make sure to run these scripts in the same dir as the configuration files for proper operation.

# Configuring GNU Radio

---

# For Linux:
```bash
sudo apt-get install gnuradio
sudo apt-get install gr-osmosdr
```
Now we hope that the base python GNU Radio bindings are installed. To check, run:
```bash
python3 -c "import gnuradio"
```
If you get no errors, the installation was successful. If you do get errors, we are screwed...

---

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

6. Let's assume everything is working fine. Now, navigate to the directory where `antennagnu_file.py` is located using the `cd` command in the Anaconda Prompt.
7. Run the script using:
   ```bash
   python antennagnu_file.py
   ```

# For WSL:
1. Use the supported Linux distro:
    ```bash
    wsl -d Ubuntu-20.04
    ```
2. Then install the required packages:
    ```bash
    sudo apt-get install gnuradio
    sudo apt-get install gr-osmosdr
    ```
3. Verify the installation by running:
    ```bash
    python3 -c "import gnuradio"
    ```
If you get no errors, the installation was successful. If you do get errors, we are fked...

4. Go to the directory where `antennagnu_file.py` is located using the `cd` command.

5. Run the script using:
    ```bash
    python3 antennagnu_file.py
    ```

# Note:
Make sure to run the script in the same directory as your configuration files (e.g., `observation_conf.yaml`) for proper operation. And make sure that everything is set correctly in the config files before running the script!