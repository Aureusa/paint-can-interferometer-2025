"""
Optimize a small interferometer array (9 dishes) for uniform UV coverage.

Science goal:
    Mapping diffuse 21 cm emission from the Milky Way.
Constraints:
    - 9 single dishes, each 17 cm diameter (~70deg primary beam)
    - All antennas connect to a central LOFAR box with coax <= 10 m --> antennas must lie within a 10 m radius.
    - Max baseline <= 20 m (automatically satisfied).
    - Snapshot observation (10 min) --> no Earth-rotation synthesis.
    - Optimize for uniform UV coverage and minimal baseline redundancy.
"""

import math, random
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
n_ant = 9
max_baseline = 0.9     # max baseline (m)
max_radius = max_baseline / 2          # coax length limit (m)
iterations = 30000          # optimization iterations
report_every = 2000

dish_radius = 0.183 / 2
min_physical_separation = 2 * dish_radius  # 0.183 m

# simulated annealing parameters
t0 = 1.0
t_final = 0.0005
step0 = 1.5     # the step size it starts with
step_min = 0.1     # to avoid stepsizes getting too small as the iterations increase

# histogram bins for UV-plane uniformity
nbins_r = 6
nbins_theta = 6

# objective weights
w_uniform = 1.0
w_redundancy = 1.0
duplicate_tol = 0.1        # meters

config_name = "maxbl2000cm"

# ---------------- Helper functions ----------------
def random_point_in_disk(max_r):
    """Returns a uniformly sampled point [x,y] within a disc of radius max_r, 
    However to sample a disc uniformly it is not simply r = max_r * U[0,1], theta = 2 pi U[0,1] 
    as that would lead to a bias towards the inner disc at smaller radii as there is less area there for the same angles.
    A common mathematical trick is to use sqrt() to give higher radii a fairer chance of being selected,
    distributing the points evenly by area."""
    r = max_r * np.sqrt(random.random())
    theta = random.random() * 2 * np.pi
    return np.array([r * np.cos(theta), r * np.sin(theta)])

def enforce_disk(pos_ant, max_r):
    """A function that ensures that a new configuration still remains within the constrained radius of the center (10m coaxial cable length)"""
    r = np.linalg.norm(pos_ant)
    return pos_ant if r <= max_r else pos_ant * (max_r / r)

def pairwise_baselines(ants):
    """Return all plus/minus baseline vectors between antenna pairs."""
    baselines = []
    # loop over each antenna i: [x,y]
    for i in range(len(ants)):
        # loop over each other antenna j = i+1 such that we don't count any pair twice
        for j in range(i+1, len(ants)):
            # vector v = [delta_x, delta_y]
            v = ants[j] - ants[i]
            baselines.append(v)
            baselines.append(-v)
    return np.array(baselines)

def redundancy_score(baselines, tol=duplicate_tol):
    """Calcualtes the fraction of baseline pairs closer than tol. 
    To essentially create a measure of how many (near)duplicate baselines there are """

    diffs = baselines[:, None, :] - baselines[None, :, :]       # Differences each baseline and each other baseline
    diffs_squared = np.sum(diffs**2, axis=2)   # This calculates the squared difference between each pair of baselines             
    
    # The following function finds the indices for the upper triangle of the distance matrix (d2). 
    # We only want to consider each pair once (e.g., baseline A to B, not also B to A)
    # and ignore the distance of a baseline to itself (which is always zero). k=1 excludes the main diagonal.
    indices_unique_baselines = np.triu_indices_from(diffs_squared, k=1)  
    unique_diffs_squared = diffs_squared[indices_unique_baselines]

    # now we count how many differences are below the tolerated distance such that they are considered a dublicate
    count = np.sum(unique_diffs_squared < tol*tol)
    maxpairs = len(unique_diffs_squared)

    return float(count)/maxpairs if maxpairs > 0 else 0.0


def uv_uniformity_chi2(baselines, r_max=max_baseline, n_r=nbins_r, n_theta=nbins_theta):
    """
    Calculates the Chi-squared metric for UV-plane uniformity.
    """
    # 1. Convert Cartesian baselines (X, Y) to Polar Coordinates (Radius, Angle)
    radii = np.linalg.norm(baselines, axis=1)
    angles_rad = np.arctan2(baselines[:,1], baselines[:,0]) % (2*np.pi)

    # 2. Filter out baselines outside the maximum radius of interest
    within_bounds_mask = (radii <= r_max)
    radii = radii[within_bounds_mask]
    angles_rad = angles_rad[within_bounds_mask]

    # 3. Define the grid boundaries (bins)
    radius_bins = np.linspace(0.0, r_max, n_r + 1)
    angle_bins = np.linspace(0.0, 2*np.pi, n_theta + 1)

    # 4. Count observations per bin using a 2D histogram
    # H_observed is the matrix of observed counts
    H_observed, _, _ = np.histogram2d(radii, angles_rad, bins=[radius_bins, angle_bins])

    # 5. Calculate the EXPECTED counts based on bin areas
    # Cells further out (larger radius) are physically larger and should contain more points, this scales linearly with r
    r_mids = 0.5 * (radius_bins[:-1] + radius_bins[1:])
    dr = radius_bins[1] - radius_bins[0]
    dtheta = angle_bins[1] - angle_bins[0]
    
    # Calculate the relative area weights of each (polar) bin
    area_weights = np.outer(r_mids * dr, np.ones(n_theta)) * dtheta
    
    # Scale the weights so they sum up to the total number of observed points
    expected_counts = area_weights / area_weights.sum() * H_observed.sum()

    # 6. Calculate the Chi-squared metric
    # Avoid division by zero in empty bins
    denominator = np.where(expected_counts > 0, expected_counts, 1.0)
    
    # Apply the Chi-squared formula: SUM((Observed - Expected)^2 / Expected)
    chi2_per_bin = ((H_observed - expected_counts)**2 / denominator)
    total_chi2_sum = chi2_per_bin.sum()
    
    # Normalize by the total number of bins to get an average value
    normalized_chi2 = total_chi2_sum / (n_r * n_theta)
    
    return normalized_chi2

def propose_move(ants, idx, step_scale=1.0):
    """Move antenna idx randomly, but respect the min physical separation."""
    max_attempts = 20
    for attempt in range(max_attempts):
        move = np.random.normal(scale=step_scale, size=2)
        new_pos = ants[idx] + move
        new_pos = enforce_disk(new_pos, max_radius)
        
        # check distance to all other antennas
        diffs = ants - new_pos
        dists = np.linalg.norm(diffs[np.arange(len(ants)) != idx], axis=1)
        if np.all(dists >= min_physical_separation):
            return new_pos
    # if all attempts fail, return original position
    return ants[idx]

def objective(ants):
    """Total cost function combining uniformity, redundancy, and spacing."""
    # calculate the baselines for input into the three different components of the objective function
    baselines = pairwise_baselines(ants)

    # Calculate Chi2, redundancy_score, minimal seperation and its corresponding penalty
    chi2 = uv_uniformity_chi2(baselines)
    red = redundancy_score(baselines)

    # Calculates the cost 
    cost = w_uniform * chi2 + w_redundancy * red
    return cost, dict(chi2=chi2, red=red)

# ---------------- Multi-seed Simulated Annealing ----------------

def run_annealing(seed):
    random.seed(seed)
    np.random.seed(seed)

    # --- Initialization ---
    ants = np.array([random_point_in_disk(max_radius) for _ in range(n_ant)])
    best_ants = ants.copy()
    best_cost, best_metrics = objective(best_ants)
    current_ants = ants.copy()
    current_cost = best_cost

    acceptance_history = []
    acceptance_window = 500
    accepted_moves = 0

    # --- Optimization loop ---
    for it in range(iterations):
        # temperature schedule
        T = t0 * (t_final / t0) ** (it / float(iterations))

        # pick random antenna and propose move
        idx = random.randrange(n_ant)
        step_scale = step0 * (T / t0) ** 0.5 + step_min
        proposal = current_ants.copy()
        proposal[idx] = propose_move(current_ants, idx, step_scale)

        # evaluate proposal
        cost_prop, metrics_prop = objective(proposal)
        delta = cost_prop - current_cost

        # acceptance check
        accepted = False
        if delta <= 0 or random.random() < math.exp(-delta / max(1e-12, T)):
            current_ants = proposal
            current_cost = cost_prop
            accepted = True

        # update best solution
        if current_cost < best_cost:
            best_cost = current_cost
            best_ants = current_ants.copy()
            best_metrics = metrics_prop

        # acceptance tracking
        if accepted:
            accepted_moves += 1
        if (it + 1) % acceptance_window == 0:
            rate = accepted_moves / acceptance_window
            acceptance_history.append(rate)
            accepted_moves = 0

        # progress printing
        if (it + 1) % report_every == 0 or it == 0 or it == iterations - 1:
            print(f"[seed {seed:03d}] iter {it+1}/{iterations}  cost={current_cost:.4f}  best={best_cost:.4f}  metrics={best_metrics}")

    # final evaluation
    ants_final = best_ants
    cost_final, metrics_final = objective(ants_final)
    print(f"[seed {seed:03d}] Final cost: {cost_final:.4f}, Metrics: {metrics_final}")

    return {
        'seed': seed,
        'cost': cost_final,
        'metrics': metrics_final,
        'ants': ants_final,
        'acceptance': acceptance_history
    }


# ---------------- Multi-seed loop ----------------
n_runs = 1
results = []

for s in range(n_runs):
    print(f"\n=== Starting run {s+1}/{n_runs} (seed={s}) ===")
    result = run_annealing(seed=s)
    results.append(result)

# ---------------- Select best result ----------------
best_run = min(results, key=lambda r: r['cost'])
print("\n=== Best configuration found ===")
print(f"Seed: {best_run['seed']}, Cost: {best_run['cost']:.5f}, Metrics: {best_run['metrics']}")

ants_final = best_run['ants']

# # ---------------- Optimization ----------------
# # seed
# seed = 41
# random.seed(seed); np.random.seed(seed)

# # Simulated Annealing algorithm:

# # 1. start with a random antenna configuration

# # 2. In each iteration, pick a random antenna and move it to a slightly new spot which then becomes the proposal array

# # 3. Check if the new configuration is better or worse (lower or higher cost according to the cost function)

# # 4. If better: We always accept the new configuration, If worse: we might still accept it, depending on the temperature T

# # 5. Cool down: The temperature starts high and gradually drops 
# # (When hot in the beginning you accept bad moves quite often so you enable the array to explore the configuration space)
# # (When cold towards the end, you rarely accept bad moves so you can start to refine within the local minimum of the cost function)

# # 6. Update the current absolute best arrangement that has been found "best_ants"

# # 1.
# # initial array
# ants = np.array([random_point_in_disk(max_radius) for _ in range(n_ant)])

# # Save the first configuration as the best one so far and calculate the corresponding cost
# best_ants = ants.copy()
# best_cost, best_metrics = objective(best_ants)
# current_ants = ants.copy()
# current_cost = best_cost

# # tracking to see if the temperature method is good or if the system cools down to fast
# acceptance_history = []
# acceptance_window = 500   # compute acceptance rate over this many iterations
# accepted_moves = 0

# # start the loop
# for it in range(iterations):

#     # calculates the temprature for the current interation. The temperature decreases exponentially from t0 down to t_final as iterations proceed
#     T = t0 * (t_final/t0)**(it/float(iterations))

#     # 2.
#     # Randomly selects which antenna to move 
#     idx = random.randrange(n_ant)

#     # determines by how much we can move the antenna, 
#     # this scale decreases as you proceed in the loop, such that you can explore more efficiently in the beginning
#     #step_scale = 1.2 * (1 - it/iterations) + 0.2    # basic linear decrease in step_scale
#     step_scale = step0 * (T / t0)**0.5 + step_min   # dependent on temperature

#     # Stores the current configuration to propose a change
#     proposal = current_ants.copy()

#     # creates the proposed array by moving the antenna and enforces the configuration to abide by our constraints
#     proposal[idx] = propose_move(current_ants, idx, step_scale)

#     # 3.
#     # calculate the cost of the proposed configuration, and determine how much better or worse it is compared to the previous best config
#     cost_prop, metrics_prop = objective(proposal)  
#     delta = cost_prop - current_cost  

#     # 4. 
#     # If worse: we might still accept it, depending on the temperature T
#     accepted = False
#     if delta <= 0 or random.random() < math.exp(-delta / max(1e-12, T)):
#         current_ants = proposal
#         current_cost = cost_prop
#         accepted = True

#     # If better: We always accept the new configuration
#     if current_cost < best_cost:
#         best_cost = current_cost
#         best_ants = current_ants.copy()
#         best_metrics = metrics_prop

#     #track the amount we accept movements, to validate the T
#     if accepted:
#         accepted_moves += 1
    
#     # record rolling acceptance rate
#     if (it + 1) % acceptance_window == 0:
#         rate = accepted_moves / acceptance_window
#         acceptance_history.append(rate)
#         accepted_moves = 0  # reset counter for next window

#     # # occasional random restart to ensure you can jump out of local minima
#     # if it % (iterations//4) == 0 and it>0 and random.random()<0.15:
#     #     k = random.randrange(n_ant)
#     #     current_ants[k] = random_point_in_disk(max_radius)
#     #     current_cost, _ = objective(current_ants)

#     # print the progress of the loop
#     if (it+1) % report_every == 0 or it==0 or it==iterations-1:
#         print(f"iter {it+1}/{iterations}  cost={current_cost:.4f}  best={best_cost:.4f}  metrics={best_metrics}")


# # ---------------- Results ----------------
# ants_final = best_ants
# cost_final, metrics_final = objective(ants_final)
# print("\nFinal cost: {:.4f}".format(cost_final))
# print("Metrics:", metrics_final)

# # ---------------- Acceptance rate plot ----------------
# plt.figure(figsize=(7,4))
# plt.scatter(np.arange(len(acceptance_history)) * acceptance_window, acceptance_history, s=5)
# plt.xlabel("Iteration")
# plt.ylabel("Acceptance rate")
# plt.title("Acceptance Rate vs Iteration")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ---------------- Save array definition file ----------------
out_cfg = f"Uniform_UV_results/{config_name}.config"
with open(out_cfg, "w") as f:
    f.write("# Array definition file for the SBCA mixed\n\n")
    f.write("# Name of the telescope\n")
    f.write("telescope = SCALAR\n\n")
    f.write("# Name of the configuration\n")
    f.write(f"config = {config_name}\n\n")
    f.write("# Latitude of the array centre\n")
    f.write("latitude_deg = 52.160114\n\n")
    f.write("# Antenna Diameter\n")
    f.write("diameter_m = 0.178\n\n")
    f.write("# Antenna coordinates (offset E, offset N)\n")
    for p in ants_final:
        f.write(f"{p[0]: .8f}, {p[1]: .8f}\n")

print(f"Array definition file saved to: {out_cfg}")

# ---------------- Plotting ----------------
plt.figure(figsize=(6,6))
plt.scatter(ants_final[:,0], ants_final[:,1], s=60)
for i,p in enumerate(ants_final):
    plt.text(p[0]*1.03, p[1]*1.03, f"A{i+1}", fontsize=9)
circle = plt.Circle((0,0), max_radius, fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.title("Antenna layout (m) — max coax radius = 10 m")
plt.xlabel("East offset (m)")
plt.ylabel("North offset (m)")
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()

metrics_final = best_run['metrics']

print("\nSummary:")
print(f" - χ² uniformity metric: {metrics_final['chi2']:.4f}")
print(f" - redundancy fraction:  {metrics_final['red']:.4f}")
print(f"\nArray definition file written to: {out_cfg}")

# ---------------------------------------------------------
# Baseline density histogram and expected uniform distribution
# ---------------------------------------------------------
bls = pairwise_baselines(ants_final)
r = np.linalg.norm(bls, axis=1)
theta = np.arctan2(bls[:, 1], bls[:, 0]) % (2 * np.pi)

# Define histogram bins
r_edges = np.linspace(0.0, max_baseline, nbins_r + 1)
theta_edges = np.linspace(0.0, 2 * np.pi, nbins_theta + 1)
r_mids = 0.5 * (r_edges[:-1] + r_edges[1:])
dr = r_edges[1] - r_edges[0]
dtheta = theta_edges[1] - theta_edges[0]

# Observed baseline density
H, _, _ = np.histogram2d(r, theta, bins=[r_edges, theta_edges])

# Expected uniform distribution ∝ area of annular sector
E = np.outer(r_mids * dr, np.ones(nbins_theta)) * dtheta
E *= H.sum() / E.sum()  # Normalize so total counts match

# Plot both observed and expected distributions
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
im0 = axs[0].imshow(H, origin='lower', aspect='auto',
                    extent=[0, 360, 0, max_baseline])
axs[0].set_title('Observed baseline density H(r,θ)')
axs[0].set_xlabel('θ (deg)')
axs[0].set_ylabel('Baseline length r (m)')
plt.colorbar(im0, ax=axs[0], label='Counts')

im1 = axs[1].imshow(E, origin='lower', aspect='auto',
                    extent=[0, 360, 0, max_baseline])
axs[1].set_title('Expected uniform distribution E(r,θ)')
axs[1].set_xlabel('θ (deg)')
axs[1].set_ylabel('Baseline length r (m)')
plt.colorbar(im1, ax=axs[1], label='Expected counts')
plt.tight_layout()
plt.savefig(f"Uniform_UV_results/FINAL_ANEALLING_baseline_dist_{config_name}")
plt.show()

# Optionally print χ² metric again
chi2_uniform = ((H - E)**2 / np.where(E>0, E, 1)).sum() / (nbins_r * nbins_theta)
print(f"Chi² uniformity metric (from histograms): {chi2_uniform:.4f}")

# ---------------------------------------------------------
# Visualization: UV coverage, 2D density, and 1D radial profile
# ---------------------------------------------------------
bls = pairwise_baselines(ants_final)
r = np.linalg.norm(bls, axis=1)
theta = np.arctan2(bls[:, 1], bls[:, 0]) % (2 * np.pi)

# Histogram setup
r_edges = np.linspace(0.0, max_baseline, nbins_r + 1)
theta_edges = np.linspace(0.0, 2 * np.pi, nbins_theta + 1)
r_mids = 0.5 * (r_edges[:-1] + r_edges[1:])
dr = r_edges[1] - r_edges[0]
dtheta = theta_edges[1] - theta_edges[0]

# Observed baseline density H(r,θ)
H, _, _ = np.histogram2d(r, theta, bins=[r_edges, theta_edges])

# Expected uniform baseline density E(r,θ) ∝ r
E = np.outer(r_mids * dr, np.ones(nbins_theta)) * dtheta
E *= H.sum() / E.sum()  # normalize so total counts match

# Azimuthally averaged profiles
H_r = H.mean(axis=1)
E_r = E.mean(axis=1)

# χ² metric for uniformity
chi2_uniform = ((H - E)**2 / np.where(E>0, E, 1)).sum() / (nbins_r * nbins_theta)
print(f"χ² uniformity metric = {chi2_uniform:.4f}")

# ---------------- Figure layout ----------------
fig = plt.figure(figsize=(14, 4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1])

# UV-plane scatter
ax0 = fig.add_subplot(gs[0,0])
ax0.scatter(bls[:,0], bls[:,1], s=6)
ax0.set_aspect('equal')
ax0.set_title('UV-plane coverage')
ax0.set_xlabel('u (m)')
ax0.set_ylabel('v (m)')
ax0.grid(True)
circ = plt.Circle((0,0), max_baseline, fill=False, linestyle='--', color='k', alpha=0.5)
ax0.add_artist(circ)

# 2D histogram comparison (Observed vs Expected)
ax1 = fig.add_subplot(gs[0,1])
im = ax1.imshow(H - E, origin='lower', aspect='auto',
                extent=[0, 360, 0, max_baseline], cmap='coolwarm')
ax1.set_title('Observed – Expected baseline density ΔH(r,θ)')
ax1.set_xlabel('θ (deg)')
ax1.set_ylabel('r (m)')
plt.colorbar(im, ax=ax1, label='Δ counts')

# 1D radial profile
ax2 = fig.add_subplot(gs[0,2])
ax2.plot(r_mids, H_r, 'o-', label='Observed mean H(r)')
ax2.plot(r_mids, E_r, 'k--', label='Expected uniform E(r) ∝ r')
ax2.set_xlabel('Baseline length r (m)')
ax2.set_ylabel('Mean counts per θ bin')
ax2.set_title('Azimuthally averaged density')
ax2.grid(True)
ax2.legend()

plt.suptitle(f"UV-plane uniformity diagnostics  (χ² = {chi2_uniform:.3f})", fontsize=13)
plt.tight_layout(rect=[0, 0.0, 1, 0.93])
plt.savefig(f"Uniform_UV_results/FINAL_ANEALLING_PLOT_{config_name}")
plt.show()