import matplotlib.pyplot as plt
import numpy as np

def read_array_file(filename):
    #Reads an array file and returns the telescope info and antenna coordinates.
    telescope = None
    config = None
    latitude_deg = None
    diameter_m = None
    coords = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                if key == 'telescope':
                    telescope = value
                elif key == 'config':
                    config = value
                elif key == 'latitude_deg':
                    latitude_deg = float(value)
                elif key == 'diameter_m':
                    diameter_m = float(value)
            else:
                # Antenna coordinates
                parts = line.split(',')
                if len(parts) == 2:
                    e, n = float(parts[0].strip()), float(parts[1].strip())
                    coords.append((e, n))
    return telescope, config, latitude_deg, diameter_m, coords


def plot_baseline_matrix_and_histogram(coords, config_name, freq_hz=1.420e9, save=True):
    """
    Plots:
      (1) 2D matrix of baseline lengths between antenna pairs.
      (2) Histogram of all unique baseline lengths.
      Adds angular resolution range at given observing frequency.

    Parameters
    ----------
    coords : ndarray (N, 2)
        East and North coordinates of each antenna [m].
    config_name : str
        Configuration name for titles and filenames.
    freq_hz : float
        Observing frequency [Hz]. Default = 1.420 GHz (21 cm line).
    save : bool
        Save figure as PDF.
    """
    c = 2.99792458e8  # m/s
    lam = c / freq_hz

    n = len(coords)
    baseline_matrix = np.zeros((n, n))

    # Compute baseline lengths
    for i in range(n):
        for j in range(n):
            e1, n1 = coords[i]
            e2, n2 = coords[j]
            baseline_matrix[i, j] = np.sqrt((e2 - e1)**2 + (n2 - n1)**2)

    # Unique baseline lengths (i<j)
    baseline_lengths = baseline_matrix[np.triu_indices(n, k=1)]

    # --- Figure layout ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (1) Baseline length matrix
    im = axes[0].imshow(baseline_matrix, origin='lower', cmap='viridis')
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Baseline Length (m)')
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(range(1, n+1))
    axes[0].set_yticklabels(range(1, n+1))
    axes[0].set_xlabel('Antenna Number')
    axes[0].set_ylabel('Antenna Number')
    axes[0].invert_yaxis()
    axes[0].set_title(f'{config_name}: Baseline Matrix')

    # (2) Histogram of baseline lengths
    bins = np.linspace(0, baseline_lengths.max(), 20)
    counts, edges, patches = axes[1].hist(baseline_lengths, bins=bins, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Baseline Length (m)')
    axes[1].set_ylabel('Number of Baselines')
    axes[1].set_title(f'{config_name}: Baseline Distribution')
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()

    if save:
        filename = f"baseline_diagnostics_{config_name.replace(' ', '_')}.pdf"
        plt.savefig(filename)
        print(f"Saved figure as {filename}")

    plt.show()

    # --- Summary statistics ---
    Bmin, Bmax = baseline_lengths.min(), baseline_lengths.max()
    theta_max = (lam / Bmin) * (180/np.pi)  
    theta_min = (lam / Bmax) * (180/np.pi) 

    print(f"\n{config_name} summary:")
    print(f"  Number of antennas: {n}")
    print(f"  Unique baselines: {len(baseline_lengths)}")
    print(f"  Shortest baseline: {Bmin:.2f} m")
    print(f"  Longest baseline: {Bmax:.2f} m")
    print(f"  Median baseline: {np.median(baseline_lengths):.2f} m")
    print(f"  Resolution range: {theta_max:.1f}' – {theta_min:.1f}' (deg)")

def plot_build_guide(coords, config, diameter_m=None, lofar_box=(0,0)):
    """
    Build guide plot showing antennas, LOFAR box, dotted lines,
    with a legend inside the plot area listing cable lengths and angles,
    with values centered.
    """
    x, y = zip(*coords)

    # Compute cable lengths and angles
    cable_info = []
    for i, (xi, yi) in enumerate(coords, start=1):
        dx = xi - lofar_box[0]
        dy = yi - lofar_box[1]
        length = np.sqrt(dx**2 + dy**2)
        angle = np.degrees(np.arctan2(dy, dx))
        cable_info.append((i, length, angle))

    # Create plot
    plt.figure(figsize=(10,8))
    plt.scatter(x, y, s=120, c='blue', label='Antennas')

    # Draw antenna circles
    if diameter_m:
        for xi, yi in coords:
            circle = plt.Circle((xi, yi), diameter_m/2, color='blue', fill=False, linestyle='--')
            plt.gca().add_patch(circle)

    # Draw LOFAR box
    plt.scatter([lofar_box[0]], [lofar_box[1]], marker='s', c='orange', s=200, label='LOFAR Box')



    # Draw dotted lines from antennas to LOFAR box
    for xi, yi in coords:
        plt.plot([xi, lofar_box[0]], [yi, lofar_box[1]], 'k--', linewidth=1)
    
    # Add a circle (radius 0.66 m) around the LOFAR box
    circle = plt.Circle((lofar_box[0], lofar_box[1]), 0.46,
                              edgecolor='green', facecolor='none',
                              linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
    
    plt.gca().add_patch(circle)

    # Label antennas
    for i, (xi, yi) in enumerate(coords, start=1):
        plt.text(xi, yi - 0.07, f'A{i}', ha='center', va='bottom', fontsize=14, color='black')

    # Embedded centered legend inside plot
    header = f"{'Ant':^7} | {'Cable (m)':^7} | {'Angle (°)':^10}"
    legend_lines = [f"{i:^8} | {length:^12.2f} | {angle:^10.1f}" 
                    for i, length, angle in cable_info]
    legend_text = header + "\n" + "\n".join(legend_lines)

    plt.gca().text(0.02, 0.98, legend_text, transform=plt.gca().transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Axes and styling
    plt.xlabel('Offset East (m)')
    plt.ylabel('Offset North (m)')
    plt.title(f'{config}')
    plt.grid(True, linestyle=':')
    plt.axis('equal')
    plt.savefig(f"build_guide_{config}.pdf")
    plt.show()

if __name__ == "__main__":
    filename = 'antenna_layout.config'
    telescope, config, latitude_deg, diameter_m, coords = read_array_file(filename)
    
    print(f"Telescope: {telescope}")
    print(f"Config: {config}")
    print(f"Latitude (deg): {latitude_deg}")
    print(f"Antenna Diameter (m): {diameter_m}")
    print(f"Number of Antennas: {len(coords)}")

    plot_baseline_matrix_and_histogram(coords, config)
    plot_build_guide(coords, config, diameter_m )
