import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FixedLocator, FixedFormatter


# filename = "data/params/scs_weight_01.pkl"
filename = "data/params/scs_p_01.pkl"
# filename = "data/params/scs_out_01.pkl"

def main():
    with open(filename, "rb") as f:
        scs_out = pkl.load(f)
        n_vals_per_iter = 0
        n_iters = 0
        n_epochs = 0
        out_vals = {}
        for name, vals in scs_out.items():
            n_vals_per_iter = vals.size
            layer, i_epoch, i_iter = name.split("_")
            n_iters = np.maximum(n_iters, int(i_iter) + 1)
            n_epochs = np.maximum(n_epochs, int(i_epoch) + 1)
            out_vals[layer] = None

        for layer in out_vals.keys():
            out_vals[layer] = np.zeros((n_epochs, n_iters * n_vals_per_iter))

        for name, vals in scs_out.items():
            layer, i_epoch, i_iter = name.split("_")
            i_iter = int(i_iter)
            i_epoch = int(i_epoch)
            out_vals[layer][
                    i_epoch,
                    n_vals_per_iter * i_iter: n_vals_per_iter * (i_iter + 1)
                    ] = vals

    for layer, vals in out_vals.items():
        histogram(vals, layer)


def histogram(data, name):
    dpi = 300
    border = .125
    eps = 1e-6
    # n_bins = 50
    n_bins = 100
    n_target_rows = 16

    fig = plt.figure()
    ax = fig.gca()

    n_rows, n_cols = data.shape
    d_keep = np.maximum(1, np.floor(n_rows / n_target_rows))
    i_keep = np.arange(0, n_rows, d_keep, dtype=int)
    data = data[i_keep, :]
    n_rows, n_cols = data.shape

    x_min = np.min(data)
    x_max = np.max(data)
    x_range = x_max - x_min + eps
    x_border = x_range * border
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)

    y_range = n_rows
    y_border = y_range * border
    ax.set_ylabel("Epoch")
    scale = .1

    for i in np.arange(n_rows)[::-1]:
        y_0 = i

        ax.plot(
            [x_min, x_max],
            [y_0, y_0],
            color="gray",
            alpha=.4,
            linewidth=.1,
        )

        counts, bins = np.histogram(data[i, :], bins=bin_edges)
        x = np.array([x_min] + list(bin_centers) + [x_max])
        y = y_0 + np.log2(np.array([0] + list(counts) + [0]) + 1) * scale
        ax.add_patch(patches.Polygon(
            np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1),
            alpha=1,
            edgecolor="black",
            facecolor="lightsteelblue",
            linewidth=.2,
        ))

    y_formatter = FixedFormatter([str(i) for i in i_keep])
    y_locator = FixedLocator(list(np.arange(i_keep.size)))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.set_major_locator(y_locator)

    plt.savefig("vis/hist_" + name + ".png", dpi=dpi)
    plt.close()


if __name__ == "__main__":
    main()
