"""Module for visualization functions."""

# Authors
# -------
# Author: Lukas Behammer
# Research Center Wels
# University of Applied Sciences Upper Austria, 2023
# CT Research Group
#
# Modifications
# -------------
# Original code, 2024, Lukas Behammer
#
# License
# -------
# BSD-3-Clause License

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def _visualize_cnr_2d(img, signal_center, background_center, radius):
    fig, axs = plt.subplots(2, 1, figsize=(6, 12), dpi=300)
    fig.suptitle("Regions for CNR Calculation",
                 y=0.92)
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Background")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            background_center[1] - radius,
            background_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0].add_patch(rect_1)

    axs[1].imshow(img, cmap="gray")
    axs[1].set_title("Signal")
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1].add_patch(rect_1)
    plt.show()


def _visualize_cnr_3d(img, signal_center, background_center, radius):
    fig, axs = plt.subplots(2, 3, figsize=(14, 10), dpi=300)
    fig.suptitle("Background (Upper) and Signal Region (Lower) for CNR Calculation",
                 y=0.92)
    axs[0][0].imshow(img[background_center[0], ::-1, :],
                     cmap="gray")
    axs[0][0].set_title(f"z-axis, slice {background_center[0]}")
    axs[0][0].set_xlabel("y")
    axs[0][0].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            background_center[1] - radius,
            background_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0][0].axhline(
        y=background_center[2], color="g", linestyle="--"
    )
    axs[0][0].axvline(
        x=background_center[1], color="g", linestyle="--"
    )
    axs[0][0].add_patch(rect_1)

    axs[0][1].imshow(img[::-1, background_center[1], :],
                     cmap="gray")
    axs[0][1].set_title(f"y-axis, slice {background_center[1]}")
    axs[0][1].set_xlabel("x")
    axs[0][1].set_ylabel("z")
    rect_2 = patches.Rectangle(
        (
            background_center[2] - radius,
            background_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0][1].axhline(
        y=background_center[0], color="g", linestyle="--"
    )
    axs[0][1].axvline(
        x=background_center[2], color="g", linestyle="--"
    )
    axs[0][1].add_patch(rect_2)

    axs[0][2].imshow(img[::-1, :, background_center[2]],
                     cmap="gray")
    axs[0][2].set_title(f"x-axis, slice {background_center[2]}")
    axs[0][2].set_xlabel("y")
    axs[0][2].set_ylabel("z")
    rect_3 = patches.Rectangle(
        (
            background_center[1] - radius,
            background_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0][2].axhline(
        y=background_center[0], color="g", linestyle="--"
    )
    axs[0][2].axvline(
        x=background_center[1], color="g", linestyle="--"
    )
    axs[0][2].add_patch(rect_3)

    axs[1][0].imshow(img[signal_center[0], ::-1, :], cmap="gray")
    axs[1][0].set_title(f"z-axis, slice {signal_center[0]}")
    axs[1][0].set_xlabel("y")
    axs[1][0].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1][0].axhline(y=signal_center[2], color="g",
                      linestyle="--")
    axs[1][0].axvline(x=signal_center[1], color="g",
                      linestyle="--")
    axs[1][0].add_patch(rect_1)

    axs[1][1].imshow(img[::-1, signal_center[1], :], cmap="gray")
    axs[1][1].set_title(f"y-axis, slice {signal_center[1]}")
    axs[1][1].set_xlabel("x")
    axs[1][1].set_ylabel("z")
    rect_2 = patches.Rectangle(
        (
            signal_center[2] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1][1].axhline(y=signal_center[0], color="g",
                      linestyle="--")
    axs[1][1].axvline(x=signal_center[2], color="g",
                      linestyle="--")
    axs[1][1].add_patch(rect_2)

    axs[1][2].imshow(img[::-1, :, signal_center[2]], cmap="gray")
    axs[1][2].set_title(f"x-axis, slice {signal_center[2]}")
    axs[1][2].set_xlabel("y")
    axs[1][2].set_ylabel("z")
    rect_3 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1][2].axhline(y=signal_center[0], color="g",
                      linestyle="--")
    axs[1][2].axvline(x=signal_center[1], color="g",
                      linestyle="--")
    axs[1][2].add_patch(rect_3)
    plt.show()


def _visualize_snr_2d(img, signal_center, radius):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    fig.suptitle("Signal Region for SNR Calculation", y=0.92)

    ax.imshow(img, cmap="gray")
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            signal_center[0] - radius,
            signal_center[1] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    ax.add_patch(rect_1)
    plt.show()


def _visualize_snr_3d(img, signal_center, radius):
    fig, axs = plt.subplots(1, 3, figsize=(14, 6), dpi=300)
    fig.suptitle("Signal Region for SNR Calculation", y=0.92)

    axs[0].imshow(img[signal_center[0], ::-1, :],
                  cmap="gray")
    axs[0].set_title(f"z-axis, slice {signal_center[0]}")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[0].axhline(y=signal_center[2], color="g",
                   linestyle="--")
    axs[0].axvline(x=signal_center[1], color="g",
                   linestyle="--")
    axs[0].add_patch(rect_1)

    axs[1].imshow(img[::-1, signal_center[1], :],
                  cmap="gray")
    axs[1].set_title(f"y-axis, slice {signal_center[1]}")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    rect_2 = patches.Rectangle(
        (
            signal_center[2] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1].axhline(y=signal_center[0], color="g",
                   linestyle="--")
    axs[1].axvline(x=signal_center[2], color="g",
                   linestyle="--")
    axs[1].add_patch(rect_2)

    axs[2].imshow(img[::-1, :, signal_center[2]],
                  cmap="gray")
    axs[2].set_title(f"x-axis, slice {signal_center[2]}")
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("z")
    rect_3 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[2].axhline(y=signal_center[0], color="g",
                   linestyle="--")
    axs[2].axvline(x=signal_center[1], color="g",
                   linestyle="--")
    axs[2].add_patch(rect_3)
    plt.show()
