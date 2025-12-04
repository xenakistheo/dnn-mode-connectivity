import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import seaborn as sns

parser = argparse.ArgumentParser(description='Plane visualization')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: None). Can be a directory containing plane.npz or the full path to plane.npz')
parser.add_argument('--no-usetex', action='store_true',
                    help='Disable matplotlib usetex even if available (useful if no TeX installed).')
args = parser.parse_args()

# Accept either a directory or a direct .npz file path
if os.path.isdir(args.dir):
    npz_path = os.path.join(args.dir, 'plane.npz')
elif os.path.isfile(args.dir) and args.dir.lower().endswith('.npz'):
    npz_path = args.dir
else:
    candidate = os.path.normpath(args.dir)
    if os.path.isfile(candidate):
        npz_path = candidate
    else:
        raise FileNotFoundError(f"Could not find plane.npz. Checked: {args.dir} and {os.path.join(args.dir, 'plane.npz')}")

file = np.load(npz_path)

# Try to enable LaTeX rendering, but fall back safely if it fails or user requested no-usetex.
if args.no_usetex:
    matplotlib.rc('text', usetex=False)
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']})
else:
    try:
        # Enable usetex and set a preamble. matplotlib expects a string for the preamble.
        matplotlib.rc('text', usetex=True)
        # Provide preamble as a single string (not a list) to avoid ValueError.
        matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}\sansmath'
        matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']})
    except Exception as e:
        # If enabling usetex fails (no TeX installed or rc param types incompatible), fall back.
        print("Warning: enabling usetex failed; falling back to usetex=False. Reason:", e)
        matplotlib.rc('text', usetex=False)
        matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']})

matplotlib.rc('xtick.major', pad=12)
matplotlib.rc('ytick.major', pad=12)
matplotlib.rc('grid', linewidth=0.8)

sns.set_style('whitegrid')


class LogNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)


def plane(grid, values, vmax=None, log_alpha=-5, N=7, cmap='jet_r'):
    cmap = plt.get_cmap(cmap)
    if vmax is None:
        clipped = values.copy()
    else:
        clipped = np.minimum(values, vmax)
    log_gamma = (np.log(clipped.max() - clipped.min()) - log_alpha) / N
    levels = clipped.min() + np.exp(log_alpha + log_gamma * np.arange(N + 1))
    levels[0] = clipped.min()
    levels[-1] = clipped.max()
    levels = np.concatenate((levels, [1e10]))
    norm = LogNormalize(clipped.min() - 1e-8, clipped.max() + 1e-8, log_alpha=log_alpha)
    contour = plt.contour(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                          linewidths=2.5,
                          zorder=1,
                          levels=levels)
    contourf = plt.contourf(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                            levels=levels,
                            zorder=0,
                            alpha=0.55)
    colorbar = plt.colorbar(format='%.2g')
    labels = list(colorbar.ax.get_yticklabels())
    if len(labels) >= 2:
        labels[-1].set_text(r'$>\,$' + labels[-2].get_text())
        colorbar.ax.set_yticklabels(labels)
    return contour, contourf, colorbar


plt.figure(figsize=(12.4, 7))

contour, contourf, colorbar = plane(
    file['grid'],
    file['tr_loss'],
    vmax=5.0,
    log_alpha=-5.0,
    N=7
)

bend_coordinates = file['bend_coordinates']
curve_coordinates = file['curve_coordinates']

plt.scatter(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], marker='o', c='k', s=120, zorder=2)
plt.scatter(bend_coordinates[1, 0], bend_coordinates[1, 1], marker='D', c='k', s=120, zorder=2)
plt.plot(curve_coordinates[:, 0], curve_coordinates[:, 1], linewidth=4, c='k', label='$w(t)$', zorder=4)
plt.plot(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], c='k', linestyle='--', dashes=(3, 4), linewidth=3, zorder=2)

plt.margins(0.0)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
colorbar.ax.tick_params(labelsize=18)
out_dir = os.path.dirname(npz_path) if os.path.isfile(npz_path) else args.dir
plt.savefig(os.path.join(out_dir, 'train_loss_plane.pdf'), format='pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(12.4, 7))

contour, contourf, colorbar = plane(
    file['grid'],
    file['te_err'],
    vmax=40,
    log_alpha=-1.0,
    N=7
)

bend_coordinates = file['bend_coordinates']
curve_coordinates = file['curve_coordinates']

plt.scatter(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], marker='o', c='k', s=120, zorder=2)
plt.scatter(bend_coordinates[1, 0], bend_coordinates[1, 1], marker='D', c='k', s=120, zorder=2)
plt.plot(curve_coordinates[:, 0], curve_coordinates[:, 1], linewidth=4, c='k', label='$w(t)$', zorder=4)
plt.plot(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], c='k', linestyle='--', dashes=(3, 4), linewidth=3, zorder=2)

plt.margins(0.0)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
colorbar.ax.tick_params(labelsize=18)
plt.savefig(os.path.join(out_dir, 'test_error_plane.pdf'), format='pdf', bbox_inches='tight')
plt.show()