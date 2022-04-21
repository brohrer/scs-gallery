import numpy as np
import matplotlib.pyplot as plt


def main():
    scs1_p = np.exp(np.load("data/params/scs1_p.npy") / 5)
    scs2_depth_p = np.exp(np.load("data/params/scs2_depth_p.npy") / 5)
    scs2_point_p = np.exp(np.load("data/params/scs2_point_p.npy") / 5)
    scs3_p = np.exp(np.load("data/params/scs3_p.npy") / 5)
    scs1_q = np.log10(np.exp(-np.load("data/params/scs1_q.npy") / .3))
    scs2_depth_q = np.log10(np.exp(-np.load("data/params/scs2_depth_q.npy") / .3))
    scs2_point_q = np.log10(np.exp(-np.load("data/params/scs2_point_q.npy") / .3))
    scs3_q = np.log10(np.exp(-np.load("data/params/scs3_q.npy") / .3))
    scs1_alpha = np.load("data/params/scs1_alpha.npy")
    scs2_depth_alpha = np.load("data/params/scs2_depth_alpha.npy")
    scs2_point_alpha = np.load("data/params/scs2_point_alpha.npy")
    scs3_alpha = np.load("data/params/scs3_alpha.npy")
    scs1_weights = np.load("data/params/scs1_weights.npy")
    scs2_depth_weights = np.load("data/params/scs2_depth_weights.npy")
    scs2_point_weights = np.load("data/params/scs2_point_weights.npy")
    scs3_weights = np.load("data/params/scs3_weights.npy")

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
    lollipop(scs1_weights, "scs1_weights")
    lollipop(scs2_depth_weights, "scs2_depth_weights")
    lollipop(scs2_point_weights, "scs2_point_weights")
    lollipop(scs3_weights, "scs3_weights")

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

if __name__ == "__main__":
    main()
