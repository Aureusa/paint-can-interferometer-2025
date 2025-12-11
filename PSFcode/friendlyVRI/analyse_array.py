import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_uv_points(path: str):
	data = np.loadtxt(path, delimiter=",", skiprows=1).T
	idx = data[0]
	u = data[1]
	v = data[2]

	mask = np.isfinite(u) & np.isfinite(v)
	return np.asarray(idx)[mask], np.asarray(u)[mask], np.asarray(v)[mask]

def plot_image(path: str, pixel_scale = 300, save_pdf = True):
	if 'Synthesized_Beam' in path:
		title = 'Synthesized Beam'
	elif 'Observed_Image' in path:
		title = 'Synthesized Dirty Image'
	else:
		title = 'Simulated Image'
	
	data = np.loadtxt(path)
	ext = pixel_scale * len(data[0]) // 2
	ext /= 3600 

	plt.title(title)
	plt.imshow(data, origin='lower', extent=[-ext, ext, -ext, ext])
	plt.ylabel('Sky position from centre (degrees)')
	plt.xlabel('Sky position from centre (degrees)')
	if save_pdf:
		out_path = os.path.splitext(path)[0] + f"{title}.pdf"
		plt.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.show()
	return data

def plot_crossection(path: str, index, pixel_scale = 300, save_pdf = True):
	if 'Synthesized_Beam' in path:
		title = 'Synthezised Beam'
	elif 'Observed_Image' in path:
		title = 'Observed Image'
	else:
		title = 'Simulated Image'
	
	data = np.loadtxt(path)
	ext = pixel_scale * len(data[0]) // 2
	ext /= 3600 

	crosssec = data[index]
	
	indcs = np.array(range(len(crosssec)))
	xticks = pixel_scale * (indcs - len(crosssec) // 2) / 3600

	plt.title(title)
	plt.plot(crosssec, label = f'Cross section at idx {index}')
	plt.xticks(indcs[::100], [f"{x:.0f}" for x in xticks[::100]])
	plt.ylabel('Strength')
	plt.xlabel('Sky position from centre (degrees)')
	if save_pdf:
		out_path = os.path.splitext(path)[0] + ".pdf"
		plt.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.show()
	return data


def plot_uv_points(path: str, show: bool = True, save_pdf: bool = False):
	"""Plot u-v coverage from a uv_points file.

	If save_png is True, saves alongside the input as .png and returns the path.
	"""
	idx, u_k, v_k = load_uv_points(path)

	# Sort by index for grouping colors (optional)
	order = np.argsort(idx)
	idx, u_k, v_k = idx[order], u_k[order], v_k[order]

	# Build a color map by array index
	unique_idx = np.unique(idx)
	cmap = plt.cm.tab10
	colors = {val: cmap(i % 10) for i, val in enumerate(unique_idx)}
	point_colors = [colors[i] for i in idx]

	plt.figure(figsize=(7, 7))
	plt.scatter(u_k, v_k, s=50, c=point_colors, marker=".", linewidths=0, alpha=0.8)
	plt.xlabel(r"u (k$\lambda$)")
	plt.ylabel(r"v (k$\lambda$)")
	plt.gca().set_aspect("equal", adjustable="datalim")
	plt.grid(True, ls=":", alpha=0.3)
	plt.title(f"uv-coverage: {os.path.basename(path)}")

	out_path = None
	if save_pdf:
		out_path = os.path.splitext(path)[0] + ".pdf"
		plt.savefig(out_path, dpi=150, bbox_inches="tight")

	if show:
		plt.show()
	else:
		plt.close()

	return out_path

file = 'SCALAR_maxbl60cm'
plot_image(f'Observed_Image_{file}.txt')
plot_image(f'Synthesized_Beam_{file}.txt')
plot_uv_points(f'uv_points_{file}.csv', save_pdf=True)

file = 'SCALAR_maxbl90cm'
plot_image(f'Observed_Image_{file}.txt')
plot_image(f'Synthesized_Beam_{file}.txt')
plot_uv_points(f'uv_points_{file}.csv', save_pdf=True)

file = 'SCALAR_maxbl2000cm'
plot_image(f'Observed_Image_{file}.txt')
plot_image(f'Synthesized_Beam_{file}.txt')
plot_uv_points(f'uv_points_{file}.csv', save_pdf=True)

