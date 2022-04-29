import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    scs1_p = np.load("data/params/scs1_p.npy")
    scs2_depth_p = np.load("data/params/scs2_depth_p.npy")
    scs2_point_p = np.load("data/params/scs2_point_p.npy")
    scs3_p = np.load("data/params/scs3_p.npy")
    scs1_q = np.log10(np.load("data/params/scs1_q.npy"))
    scs2_depth_q = np.log10(np.load("data/params/scs2_depth_q.npy"))
    scs2_point_q = np.log10(np.load("data/params/scs2_point_q.npy"))
    scs3_q = np.log10(np.load("data/params/scs3_q.npy"))
    # scs1_alpha = np.load("data/params/scs1_alpha.npy")
    # scs2_depth_alpha = np.load("data/params/scs2_depth_alpha.npy")
    # scs2_point_alpha = np.load("data/params/scs2_point_alpha.npy")
    # scs3_alpha = np.load("data/params/scs3_alpha.npy")
    scs1_weights = np.load("data/params/scs1_weights.npy")
    scs2_depth_weights = np.load("data/params/scs2_depth_weights.npy")
    scs2_point_weights = np.load("data/params/scs2_point_weights.npy")
    scs3_weights = np.load("data/params/scs3_weights.npy")

    '''
    lollipop(scs1_p, "scs1_p")
    lollipop(scs2_depth_p, "scs2_depth_p")
    lollipop(scs2_point_p, "scs2_point_p")
    lollipop(scs3_p, "scs3_p")
    lollipop(scs1_q, "scs1_q")
    lollipop(scs2_depth_q, "scs2_depth_q")
    lollipop(scs2_point_q, "scs2_point_q")
    lollipop(scs3_q, "scs3_q")
    lollipop(scs1_alpha, "scs1_alpha")
    lollipop(scs2_depth_alpha, "scs2_depth_alpha")
    lollipop(scs2_point_alpha, "scs2_point_alpha")
    lollipop(scs3_alpha, "scs3_alpha")
    '''
    histogram(scs1_p, "scs1_p")
    histogram(scs2_depth_p, "scs2_depth_p")
    histogram(scs2_point_p, "scs2_point_p")
    histogram(scs3_p, "scs3_p")
    histogram(scs1_q, "scs1_q")
    histogram(scs2_depth_q, "scs2_depth_q")
    histogram(scs2_point_q, "scs2_point_q")
    histogram(scs3_q, "scs3_q")
    # histogram(scs1_alpha, "scs1_alpha")
    # histogram(scs2_depth_alpha, "scs2_depth_alpha")
    # histogram(scs2_point_alpha, "scs2_point_alpha")
    # histogram(scs3_alpha, "scs3_alpha")
    histogram(scs1_weights, "scs1_weights")
    histogram(scs2_depth_weights, "scs2_depth_weights")
    histogram(scs2_point_weights, "scs2_point_weights")
    histogram(scs3_weights, "scs3_weights")


def lollipop(data, name):
    dpi = 300
    border = .125
    lolli_height = .8
    lolli_width = .5
    eps = 1e-6

    fig = plt.figure()
    # ax = fig.add_axes((0, 0, 1, 1))
    ax = fig.gca()

    n_rows, n_cols = data.shape
    x_min = np.min(data)
    x_max = np.max(data)
    x_range = x_max - x_min + eps
    x_border = x_range * border
    # ax.set_xlim(x_min - x_border, x_max + x_border)

    y_range = n_rows
    y_border = y_range * border
    # ax.set_ylim(-y_border, y_range + y_border)
    ax.set_ylabel("Epoch")

    for i in range(n_rows):
        y_0 = i
        ax.plot(
            [x_min, x_max],
            [y_0, y_0],
            color="black",
            linewidth=.5,
        )
        for j in range(n_cols):
            x = data[i, j]
            ax.plot(
                [x, x],
                [y_0, y_0 + lolli_height],
                color="black",
                linewidth=lolli_width,
            )
            markersize = 6 - np.log2(n_rows)
            ax.plot(
                x,
                y_0,
                color="black",
                marker='o',
                markersize=markersize,
            )

    plt.savefig("vis/lolli_" + name + ".png", dpi=dpi)
    plt.close()

def histogram(data, name):
    dpi = 300
    border = .125
    eps = 1e-6
    n_bins = 100
    n_target_rows = 10

    fig = plt.figure()
    # ax = fig.add_axes((0, 0, 1, 1))
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
    # ax.set_xlim(x_min - x_border, x_max + x_border)
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)

    y_range = n_rows
    y_border = y_range * border
    # ax.set_ylim(-y_border, y_range + y_border)
    ax.set_ylabel("Epoch")
    # scale = bin_centers.size / data.shape[1]
    scale = 1

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
        y = y_0 + np.log2( np.array([0] + list(counts) + [0]) + 1) * scale
        '''
        ax.plot(
            x, y,
            color="black",
            linewidth=.5,
        )
        '''
        ax.add_patch(patches.Polygon(
            np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1),
            alpha=1,
            edgecolor="black",
            facecolor="lightsteelblue",
            linewidth=.2,
        ))

    plt.savefig("vis/hist_" + name + ".png", dpi=dpi)
    plt.close()

if __name__ == "__main__":
    main()
