"""Module for visualization functions.

Examples
--------
    .. doctest-skip::

    >>> import numpy as np
    >>> from viqa.utils.visualization import visualize_2d, visualize_3d
    >>> img = np.random.rand(100, 100)
    >>> visualize_2d(img)
    >>> img = np.random.rand(100, 100, 100)
    >>> visualize_3d(img, (50, 50, 50))
"""

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

from viqa.utils._module import try_import

widgets, has_ipywidgets = try_import("ipywidgets")

FIGSIZE_CNR_2D = (10, 5.5)
FIGSIZE_CNR_3D = (10, 8)
FIGSIZE_SNR_2D = (7, 7)
FIGSIZE_SNR_3D = (10, 4)


def _visualize_cnr_2d(
    img, signal_center, background_center, radius, export_path=None, show=True, **kwargs
):
    figsize = kwargs.pop("figsize", FIGSIZE_CNR_2D)
    dpi = kwargs.pop("dpi", 300)

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.suptitle("Regions for CNR Calculation", y=0.92)
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Background")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].invert_yaxis()
    rect_1 = patches.Rectangle(
        (
            background_center[0] - radius,
            background_center[1] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#ca0020",
        facecolor="none",
    )
    axs[0].add_patch(rect_1)

    axs[1].imshow(img[..., ::-1], cmap="gray")
    axs[1].set_title("Signal")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].invert_yaxis()
    rect_1 = patches.Rectangle(
        (
            signal_center[0] - radius,
            signal_center[1] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#0571b0",
        facecolor="none",
    )
    axs[1].add_patch(rect_1)
    if show:
        plt.show()
    if export_path:
        plt.savefig(export_path, bbox_inches="tight", pad_inches=0.5)


def _visualize_cnr_3d(
    img, signal_center, background_center, radius, export_path=None, show=True, **kwargs
):
    figsize = kwargs.pop("figsize", FIGSIZE_CNR_3D)
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
    if show:
        plt.show()
    if export_path:
        plt.savefig(export_path, bbox_inches="tight", pad_inches=0.5)


def _visualize_snr_2d(
    img, signal_center, radius, export_path=None, show=True, **kwargs
):
    figsize = kwargs.pop("figsize", FIGSIZE_SNR_2D)
    dpi = kwargs.pop("dpi", 300)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.suptitle("Signal Region for SNR Calculation", y=0.92)

    ax.imshow(img[..., ::-1], cmap="gray")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    rect_1 = patches.Rectangle(
        (
            signal_center[0] - radius,
            signal_center[1] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="#0571b0",
        facecolor="none",
    )
    ax.add_patch(rect_1)
    if show:
        plt.show()
    if export_path:
        plt.savefig(export_path, bbox_inches="tight", pad_inches=0.5)


def _visualize_snr_3d(
    img, signal_center, radius, export_path=None, show=True, **kwargs
):
    figsize = kwargs.pop("figsize", FIGSIZE_SNR_3D)
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
    if show:
        plt.show()
    if export_path:
        plt.savefig(export_path, bbox_inches="tight", pad_inches=0.5)


def _create_slider_widget(**kwargs):
    if not has_ipywidgets:
        raise ImportError(
            "ipywidgets is not installed. Please install it to use " "this function."
        )

    min_val = kwargs.pop("min", 0)
    step = kwargs.pop("step", 1)
    continuous_update = kwargs.pop("continuous_update", False)

    slider = widgets.IntSlider(
        min=min_val,
        step=step,
        continuous_update=continuous_update,
        **kwargs,
    )
    return slider


def visualize_2d(img, export_path=None, **kwargs):
    """
    Visualize a 2D image.

    The function visualizes a 2D image. If `export_path` is provided, the
    visualization is saved to the specified path.

    Parameters
    ----------
    img : np.ndarray
        The 2D image to visualize.
    export_path : str or Path, optional
        The path to save the visualization.
    kwargs :
        Additional keyword arguments for the plot. Passed to
        ``matplotlib.pyplot.imshow``.

    Raises
    ------
    ValueError
        If the image is not 2D.

    Returns
    -------
    None
    """
    if img.ndim != 2:
        raise ValueError("The image must be 2D.")

    figsize = kwargs.pop("figsize", (6, 6))
    dpi = kwargs.pop("dpi", 300)

    plt.figure(figsize=figsize, dpi=dpi)
    if "cmap" not in kwargs:
        plt.imshow(img, cmap="gray", **kwargs)
    else:
        plt.imshow(img, **kwargs)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()
    plt.show()
    if export_path:
        plt.savefig(export_path, bbox_inches="tight", pad_inches=0.5)


def visualize_3d(img, slices, export_path=None, **kwargs):
    """
    Visualize 3D image slices in 3 different planes.

    The function visualizes the 3D image slices in the ``x``, ``y`` and ``z`` direction.
    If ``export_path`` is provided, the visualization is saved to the specified path.

    Parameters
    ----------
    img : np.ndarray
        The 3D image to visualize.
    slices : tuple
        The slices to visualize in the ``x``, ``y`` and ``z`` direction. The slices must
        be positive or negative integers.
    export_path : str or Path, optional
        The path to save the visualization.
    kwargs :
        Additional keyword arguments for the plot. Passed to
        :py:func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the number of slices is not 3 or if the slices are not integers.
        If the image is not 3D.
        If the slices are out of bounds.
    """
    if len(slices) != 3:
        raise ValueError("The number of slices must be 3.")
    if not all(isinstance(slice_, int) for slice_ in slices):
        raise ValueError("All slices must be integers.")
    if img.ndim != 3:
        raise ValueError("The image must be 3D.")
    if not all(
        -img.shape[i] <= slice_ <= img.shape[i] for i, slice_ in enumerate(slices)
    ):
        raise ValueError("The slices are out of bounds.")

    x = slices[0]
    y = slices[1]
    z = slices[2]

    figsize = kwargs.pop("figsize", (14, 6))
    dpi = kwargs.pop("dpi", 300)

    _, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi, **kwargs)

    axs[0].imshow(np.rot90(img[x, :, ::-1]), cmap="gray")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("z")
    axs[0].invert_yaxis()
    axs[0].axhline(y=z, color="#7570b3", linestyle="--")
    axs[0].axvline(x=y, color="#d95f02", linestyle="--")
    axs[0].set_title(f"x-axis, slice: {x}", c="#1b9e77")

    axs[1].imshow(np.rot90(img[:, y, ::-1]), cmap="gray")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[1].invert_yaxis()
    axs[1].axhline(y=z, color="#7570b3", linestyle="--")
    axs[1].axvline(x=x, color="#1b9e77", linestyle="--")
    axs[1].set_title(f"y-axis, slice: {y}", c="#d95f02")

    axs[2].imshow(np.rot90(img[::-1, :, z], -1), cmap="gray")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].invert_yaxis()
    axs[2].axhline(y=y, color="#d95f02", linestyle="--")
    axs[2].axvline(x=x, color="#1b9e77", linestyle="--")
    axs[2].set_title(f"z-axis, slice: {z}", c="#7570b3")

    plt.show()
    if export_path:
        plt.savefig(export_path, bbox_inches="tight", pad_inches=0.5)
