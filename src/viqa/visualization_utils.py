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
import numpy as np


def _visualize_cnr_2d(img, signal_center, background_center, radius):
    fig, axs = plt.subplots(2, 1, figsize=(6, 12), dpi=300)
    fig.suptitle("Regions for CNR Calculation", y=0.92)
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


def _visualize_cnr_3d(img, signal_center, background_center, radius, **kwargs):
    figsize = kwargs.pop("figsize", (14, 10))
    dpi = kwargs.pop("dpi", 300)

    fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=dpi, **kwargs)
    fig.suptitle(
        "Background (Upper) and Signal Region (Lower) for CNR Calculation", y=0.92
    )
    # Background Region
    axs[0][0].imshow(np.rot90(img[background_center[0], :, ::-1]), cmap="gray")
    axs[0][0].set_title(f"x-axis, slice: {background_center[0]}", c="#d7191c")
    axs[0][0].set_xlabel("y")
    axs[0][0].set_ylabel("z")
    axs[0][0].invert_yaxis()
    rect_1 = patches.Rectangle(
        (
            background_center[1] - radius,
            background_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#ffffbf",
        facecolor="none",
    )
    axs[0][0].axvline(x=background_center[1], color="#fdae61", linestyle="--")
    axs[0][0].axhline(y=background_center[2], color="#2c7bb6", linestyle="--")
    axs[0][0].add_patch(rect_1)

    axs[0][1].imshow(np.rot90(img[:, background_center[1], ::-1]), cmap="gray")
    axs[0][1].set_title(f"y-axis, slice: {background_center[1]}", c="#fdae61")
    axs[0][1].set_xlabel("x")
    axs[0][1].set_ylabel("z")
    axs[0][1].invert_yaxis()
    rect_2 = patches.Rectangle(
        (
            background_center[0] - radius,
            background_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#ffffbf",
        facecolor="none",
    )
    axs[0][1].axvline(x=background_center[0], color="#d7191c", linestyle="--")
    axs[0][1].axhline(y=background_center[2], color="#2c7bb6", linestyle="--")
    axs[0][1].add_patch(rect_2)

    axs[0][2].imshow(np.rot90(img[::-1, :, background_center[2]], -1), cmap="gray")
    axs[0][2].set_title(f"z-axis, slice: {background_center[2]}", c="#2c7bb6")
    axs[0][2].set_xlabel("x")
    axs[0][2].set_ylabel("y")
    axs[0][2].invert_yaxis()
    rect_3 = patches.Rectangle(
        (
            background_center[0] - radius,
            background_center[1] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#ffffbf",
        facecolor="none",
    )
    axs[0][2].axvline(x=background_center[0], color="#d7191c", linestyle="--")
    axs[0][2].axhline(y=background_center[1], color="#fdae61", linestyle="--")
    axs[0][2].add_patch(rect_3)

    # Signal Region
    axs[1][0].imshow(np.rot90(img[signal_center[0], :, ::-1]), cmap="gray")
    axs[1][0].set_title(f"x-axis, slice: {signal_center[0]}", c="#d7191c")
    axs[1][0].set_xlabel("y")
    axs[1][0].set_ylabel("z")
    axs[1][0].invert_yaxis()
    rect_1 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#abd9e9",
        facecolor="none",
    )
    axs[1][0].axvline(x=signal_center[1], color="#fdae61", linestyle="--")
    axs[1][0].axhline(y=signal_center[2], color="#2c7bb6", linestyle="--")
    axs[1][0].add_patch(rect_1)

    axs[1][1].imshow(np.rot90(img[:, signal_center[1], ::-1]), cmap="gray")
    axs[1][1].set_title(f"y-axis, slice: {signal_center[1]}", c="#fdae61")
    axs[1][1].set_xlabel("x")
    axs[1][1].set_ylabel("z")
    axs[1][1].invert_yaxis()
    rect_2 = patches.Rectangle(
        (
            signal_center[0] - radius,
            signal_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#abd9e9",
        facecolor="none",
    )
    axs[1][1].axvline(x=signal_center[0], color="#d7191c", linestyle="--")
    axs[1][1].axhline(y=signal_center[2], color="#2c7bb6", linestyle="--")
    axs[1][1].add_patch(rect_2)

    axs[1][2].imshow(np.rot90(img[::-1, :, signal_center[2]], -1), cmap="gray")
    axs[1][2].set_title(f"z-axis, slice: {signal_center[2]}", c="#2c7bb6")
    axs[1][2].set_xlabel("x")
    axs[1][2].set_ylabel("y")
    axs[1][2].invert_yaxis()
    rect_3 = patches.Rectangle(
        (
            signal_center[0] - radius,
            signal_center[1] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#abd9e9",
        facecolor="none",
    )
    axs[1][2].axvline(x=signal_center[0], color="#d7191c", linestyle="--")
    axs[1][2].axhline(y=signal_center[1], color="#fdae61", linestyle="--")
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


def _visualize_snr_3d(img, signal_center, radius, **kwargs):
    figsize = kwargs.pop("figsize", (14, 6))
    dpi = kwargs.pop("dpi", 300)

    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    fig.suptitle("Signal Region for SNR Calculation", y=0.92)

    axs[0].imshow(np.rot90(img[signal_center[0], :, ::-1]), cmap="gray")
    axs[0].set_title(f"x-axis, slice: {signal_center[0]}", c="#d7191c")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("z")
    axs[0].invert_yaxis()
    rect_1 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#ffffbf",
        facecolor="none",
    )
    axs[0].axvline(x=signal_center[1], color="#fdae61", linestyle="--")
    axs[0].axhline(y=signal_center[2], color="#2c7bb6", linestyle="--")
    axs[0].add_patch(rect_1)

    axs[1].imshow(np.rot90(img[:, signal_center[1], ::-1]), cmap="gray")
    axs[1].set_title(f"y-axis, slice: {signal_center[1]}", c="#fdae61")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[1].invert_yaxis()
    rect_2 = patches.Rectangle(
        (
            signal_center[0] - radius,
            signal_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#ffffbf",
        facecolor="none",
    )
    axs[1].axvline(x=signal_center[0], color="#d7191c", linestyle="--")
    axs[1].axhline(y=signal_center[2], color="#2c7bb6", linestyle="--")
    axs[1].add_patch(rect_2)

    axs[2].imshow(np.rot90(img[::-1, :, signal_center[2]], -1), cmap="gray")
    axs[2].set_title(f"z-axis, slice: {signal_center[2]}", c="#2c7bb6")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].invert_yaxis()
    rect_3 = patches.Rectangle(
        (
            signal_center[0] - radius,
            signal_center[1] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#ffffbf",
        facecolor="none",
    )
    axs[2].axvline(x=signal_center[0], color="#d7191c", linestyle="--")
    axs[2].axhline(y=signal_center[1], color="#fdae61", linestyle="--")
    axs[2].add_patch(rect_3)
    plt.show()
