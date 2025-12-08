import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import seaborn as sns

paths = [
    "./plane_data/ResNet8_CIFAR10_Bezier/plane.npz",
    "./plane_data/ResNet26_CIFAR10_Bezier/plane.npz",
    "./plane_data/ResNet38_CIFAR10_Bezier/plane.npz",
    "./plane_data/ResNet65_CIFAR10_Bezier/plane.npz",
    "./plane_data/ResNet119_CIFAR10_Bezier/plane.npz"
]

resnet_labels = ["ResNet8", "ResNet26", "ResNet38", "ResNet65", "ResNet119"]

out_dir = os.path.dirname(paths[0])

matplotlib.rc('xtick.major', pad=12)
matplotlib.rc('ytick.major', pad=12)
matplotlib.rc('grid', linewidth=0.8)

sns.set_style('whitegrid')


# --------------------------------------------------------------------
# Normalization class (same as your original)
# --------------------------------------------------------------------

class LogNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)


# --------------------------------------------------------------------
# Plotting helper that draws into a specific axis
# --------------------------------------------------------------------

def plane(grid, values, vmax=None, log_alpha=-5, N=7, cmap='jet_r', ax=None):
    if ax is None:
        ax = plt.gca()

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

    contour = ax.contour(
        grid[:, :, 0], grid[:, :, 1], values,
        cmap=cmap, norm=norm,
        linewidths=2.5,
        zorder=1,
        levels=levels
    )

    contourf = ax.contourf(
        grid[:, :, 0], grid[:, :, 1], values,
        cmap=cmap, norm=norm,
        levels=levels,
        zorder=0,
        alpha=0.55
    )

    return contour, contourf


# --------------------------------------------------------------------
# Common function for the 5×1 plot + centered horizontal colorbar
# --------------------------------------------------------------------

def make_figure(data_key, title, vmax, log_alpha, filename):

    fig, axes = plt.subplots(5, 1, figsize=(12.4, 5 * 4.5))

    last_cf = None

    for i, (npz_path, label) in enumerate(zip(paths, resnet_labels)):
        file = np.load(npz_path)
        ax = axes[i]

        # Use each file's own grid
        contour, contourf = plane(
            file["grid"],
            file[data_key],
            vmax=vmax,
            log_alpha=log_alpha,
            N=7,
            ax=ax
        )
        last_cf = contourf

        bend_coordinates = file["bend_coordinates"]
        curve_coordinates = file["curve_coordinates"]

        ax.scatter(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1],
                   marker='o', c='k', s=120, zorder=2)
        ax.scatter(bend_coordinates[1, 0], bend_coordinates[1, 1],
                   marker='D', c='k', s=120, zorder=2)
        ax.plot(curve_coordinates[:, 0], curve_coordinates[:, 1],
                linewidth=4, c='k', zorder=4)
        ax.plot(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1],
                c='k', linestyle='--', dashes=(3, 4), linewidth=3, zorder=2)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set limits to exactly the data range (no whitespace)
        grid = file["grid"]
        ax.set_xlim(grid[:, :, 0].min(), grid[:, :, 0].max())
        ax.set_ylim(grid[:, :, 1].min(), grid[:, :, 1].max())
        
        ax.margins(0.0)
        ax.set_title(label, fontsize=16)

    # ---- Title above everything ----
    fig.suptitle(title, fontsize=20, y=0.97)

    # ---- Create horizontal colorbar ABOVE subplots, centered ----
    cax = fig.add_axes([0.25, 0.92, 0.5, 0.018])

    cbar = fig.colorbar(last_cf, cax=cax, orientation='horizontal', format='%.2g')
    cbar.ax.tick_params(labelsize=14)

    # Manually adjust last label to ">" style
    ticks = cbar.get_ticks()
    if len(ticks) >= 2:
        labels = [f"{t:.2g}" for t in ticks]
        labels[-1] = r">$\,{}$".format(labels[-2])
        cbar.ax.set_xticklabels(labels)

    # ---- Adjust layout to prevent overlap ----
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    # ---- Save ----
    fig.savefig(os.path.join(out_dir, filename), format="pdf", bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------
# Run
# --------------------------------------------------------------------

make_figure(
    data_key="tr_loss",
    title="Training Loss Plane",
    vmax=5.0,
    log_alpha=-5.0,
    filename="train_loss_planes_5x1.pdf"
)

make_figure(
    data_key="te_err",
    title="Test Error Plane",
    vmax=40,
    log_alpha=-1.0,
    filename="test_error_planes_5x1.pdf"
)