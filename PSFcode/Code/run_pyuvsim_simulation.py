#!/usr/bin/env python3
"""
Run pyuvsim simulation for SCALAR small array
Quick test using chatGPT (I know), not used in the final product
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    """Main function to run the pyuvsim configuration test."""
    from pyuvsim import simsetup
    from pyuvdata import UVData
    
    # Path to the observation parameter file
    obsparam_file = 'pyuvsim_obsparam.yaml'

    print("=" * 60)
    print("PyUVSim Simulation Runner for SCALAR Array")
    print("=" * 60)

    # Initialize the simulation from parameter file
    # This reads the obsparam file and creates UVData object
    try:
        # In newer versions, it may return just UVData or a tuple
        result = simsetup.initialize_uvdata_from_params(obsparam_file)
        
        # Check if it's a tuple or single object
        if isinstance(result, tuple):
            uv_obj = result[0]
        else:
            uv_obj = result
            
        print("✓ Configuration loaded successfully")
        print(f"  - Telescope: {uv_obj.telescope.name if hasattr(uv_obj, 'telescope') else 'SCALAR'}")
        print(f"  - Number of antennas: {uv_obj.Nants_telescope}")
        print(f"  - Number of baselines: {uv_obj.Nbls}")
        print(f"  - Number of times: {uv_obj.Ntimes}")
        print(f"  - Number of frequencies: {uv_obj.Nfreqs}")
        print(f"  - Frequency range: {uv_obj.freq_array.min()/1e6:.1f} - {uv_obj.freq_array.max()/1e6:.1f} MHz")
        print(f"  - Integration time: {uv_obj.integration_time[0]:.1f} seconds")
        
    except Exception as e:
        print(f"ERROR loading configuration: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "=" * 60)
    print("To run the full simulation with MPI (recommended):")
    print("=" * 60)
    print(f"mpiexec -n 4 run_pyuvsim --param {obsparam_file}")
    print("\nOr for single-core (slower):")
    print(f"run_pyuvsim --param {obsparam_file}")
    print("\n" + "=" * 60)

    # Optionally, visualize the array layout
    print("\nVisualizing array layout...")
    # try:
    #     # Read antenna positions directly from the CSV file to get local ENU coordinates
    #     # The antenna_positions from UVData are in ECEF frame, not local ENU
    #     import csv
    #     antpos_enu = []
    #     with open('pyuvsim_array_layout.csv', 'r') as f:
    #         reader = csv.DictReader(f, delimiter=' ', skipinitialspace=True)
    #         for row in reader:
    #             e = float(row['E'])
    #             n = float(row['N'])
    #             antpos_enu.append([e, n])
    #     antpos_enu = np.array(antpos_enu)
        
    #     print(f"\nAntenna positions (Local ENU frame):")
    #     for i in range(len(antpos_enu)):
    #         r = np.sqrt(antpos_enu[i, 0]**2 + antpos_enu[i, 1]**2)
    #         print(f"  ANT{i}: E={antpos_enu[i, 0]:7.3f}, N={antpos_enu[i, 1]:7.3f}, r={r:.3f} m")
        
    #     plt.figure(figsize=(8, 8))
    #     plt.scatter(antpos_enu[:, 0], antpos_enu[:, 1], s=100, alpha=0.7, color='steelblue', edgecolors='black', linewidth=1.5)
        
    #     # Label antennas
    #     for i in range(len(antpos_enu)):
    #         plt.annotate(f'A{i}', 
    #                     (antpos_enu[i, 0], antpos_enu[i, 1]),
    #                     xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    #     # Add circle showing the ~0.66m radius
    #     circle = plt.Circle((0, 0), 0.66, fill=False, color='red', linestyle='--', linewidth=2, alpha=0.5, label='0.66m radius')
    #     plt.gca().add_patch(circle)
        
    #     plt.xlabel('East (m)', fontsize=12)
    #     plt.ylabel('North (m)', fontsize=12)
    #     plt.title(f'SCALAR Array Layout\n{uv_obj.Nants_telescope} antennas', fontsize=14, fontweight='bold')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.axis('equal')
    #     plt.tight_layout()
    #     plt.savefig('pyuvsim_array_layout.png', dpi=150)
    #     print("✓ Array layout saved to: pyuvsim_array_layout.png")
    #     plt.show()
        
    # except Exception as e:
    #     print(f"Warning: Could not create array layout plot: {e}")

    print("\nSetup complete! Ready to run simulation.")


if __name__ == '__main__':
    main()
    os.system('mpiexec -n 4 run_pyuvsim --param pyuvsim_obsparam.yaml')
