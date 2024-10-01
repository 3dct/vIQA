"""Module for calculating the signal-to-noise ratio (SNR) for an image.

Examples
--------
    .. doctest-requires:: numpy

        >>> import numpy as np
        >>> from viqa import SNR
        >>> img = np.random.rand(256, 256)
        >>> snr = SNR(data_range=1, normalize=False)
        >>> snr
        SNR(score_val=None)
        >>> score = snr.score(img,
        ...                   signal_center=(32, 32),
        ...                   radius=16)
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

from warnings import warn

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from ipywidgets import HBox, Label, VBox

from viqa._metrics import NoReferenceMetricsInterface
from viqa.utils import _rgb_to_yuv, _to_grayscale
from viqa.visualization_utils import (
    FIGSIZE_SNR_2D,
    FIGSIZE_SNR_3D,
    _create_slider_widget,
    _visualize_snr_2d,
    _visualize_snr_3d,
)

glob_signal_center = None
glob_radius = None

FIGSIZE_SNR_2D_ = tuple(f"{val}in" for val in FIGSIZE_SNR_2D)
FIGSIZE_SNR_3D_ = tuple(f"{val}in" for val in FIGSIZE_SNR_3D)


class SNR(NoReferenceMetricsInterface):
    """Class to calculate the signal-to-noise ratio (SNR) for an image.

    Attributes
    ----------
    score_val : float
        SNR score value of the last calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True. Passed to :py:func:`viqa.load_utils.load_data`.
    normalize : bool, default False
        If True, the input images are normalized to the ``data_range`` argument.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`viqa.load_utils.load_data`.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.

        .. note::
            Currently not supported.

    """

    def __init__(self, data_range=255, normalize=False, **kwargs) -> None:
        """Construct method."""
        super().__init__(data_range=data_range, normalize=normalize, **kwargs)
        self._name = "SNR"

    def score(self, img, **kwargs):
        """Calculate the signal-to-noise ratio (SNR) for an image.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to calculate score of.
        **kwargs : optional
            Additional parameters for SNR calculation. The keyword arguments are passed
            to :py:func:`viqa.nr_metrics.snr.signal_to_noise_ratio`.

        Returns
        -------
        score_val : float
            SNR score value.
        """
        img = super().score(img)

        # check if signal_center and radius are provided
        if not {"signal_center", "radius"}.issubset(kwargs):
            if not self._parameters["signal_center"] or not self._parameters["radius"]:
                raise ValueError("No center or radius provided.")

            kwargs["signal_center"] = self._parameters["signal_center"]
            kwargs["radius"] = self._parameters["radius"]

        # write kwargs to ._parameters attribute
        self._parameters.update(kwargs)

        score_val = signal_to_noise_ratio(img, **kwargs)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the SNR score value of the last calculation.

        Parameters
        ----------
        decimals : int, default=2
            Number of decimal places to print the score value.

        Warns
        -----
        RuntimeWarning
            If :py:attr:`score_val` is not available.
        """
        if self.score_val is not None:
            print("SNR: {}".format(np.round(self.score_val, decimals)))
        else:
            warn("No score value for SNR. Run score() first.", RuntimeWarning)

    def visualize_centers(
        self, img, signal_center=None, radius=None, export_path=None, **kwargs
    ):
        """Visualize the centers for SNR calculation.

        The visualization shows the signal region in a matplotlib plot. If export_path
        is provided, the plot is saved to the path.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to visualize.
        signal_center : Tuple(int), optional
            Center of the signal.
            Order is ``(x, y)`` for 2D images and ``(x, y, z)`` for 3D images.
        radius : int, optional
            Width of the regions.
        export_path : str or os.PathLike, optional
            Path to export the visualization to.
        **kwargs : optional
            Additional parameters for visualization. The keyword arguments are passed to
            :py:func:`matplotlib.pyplot.subplots`.
        """
        if not signal_center or not radius:
            if not self._parameters["signal_center"] or not self._parameters["radius"]:
                raise ValueError("No center or radius provided.")

            signal_center = self._parameters["signal_center"]
            radius = self._parameters["radius"]

        # Check if img and signal_center have the same dimension
        if img.shape[-1] == 3:
            if img.ndim != len(signal_center) + 1:
                raise ValueError("Center has to be in the same dimension as img.")
        else:
            if img.ndim != len(signal_center):
                raise ValueError("Center has to be in the same dimension as img.")

        # Visualize centers
        if img.ndim == 3 and (img.shape[-1] != 3):  # 3D image
            _visualize_snr_3d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                export_path=export_path,
                **kwargs,
            )
        elif img.ndim == 3 and (img.shape[-1] == 3):  # 2D RGB image
            img = _to_grayscale(img)
            _visualize_snr_2d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                export_path=export_path,
                **kwargs,
            )
        elif img.ndim == 2:  # 2D image
            _visualize_snr_2d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                export_path=export_path,
                **kwargs,
            )
        else:
            raise ValueError("No visualization possible for non 2d or non 3d images.")

    def set_centers(
        self,
        img,
        **kwargs,
    ):
        """Visualize and set the centers for SNR calculation interactively.

        The visualization shows the signal region in a matplotlib plot.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to visualize.
        **kwargs : optional
            Additional parameters for visualization. The keyword arguments are passed to
            :py:func:`matplotlib.pyplot.subplots`.
        """
        # Prepare visualization functions and widgets

        # Define functions for visualization
        def _update_visualization_2d(
            signal_center_x,
            signal_center_y,
            radius,
        ):
            signal_center = (signal_center_x, signal_center_y)

            global glob_signal_center, glob_radius
            glob_signal_center = signal_center
            glob_radius = radius

            _visualize_snr_2d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                **kwargs,
            )

        def _update_visualization_3d(
            signal_center_x,
            signal_center_y,
            signal_center_z,
            radius,
        ):
            signal_center = (
                signal_center_x,
                signal_center_y,
                signal_center_z,
            )

            global glob_signal_center, glob_radius
            glob_signal_center = signal_center
            glob_radius = radius

            _visualize_snr_3d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                **kwargs,
            )

        # Define function to save values
        def _save_values(_):
            global glob_signal_center, glob_radius
            self._parameters.update(
                {
                    "signal_center": glob_signal_center,
                    "radius": glob_radius,
                }
            )
            print("Parameters saved.")

        # Create slider for radius
        slider_radius = _create_slider_widget(
            max=1,
            min=1,
            value=1,
            description="Radius",
        )
        slider_radius.style = {"handle_color": "#f7f7f7"}

        # Create sliders for signal center coordinates
        slider_signal_center_x = _create_slider_widget(
            min=slider_radius.value + 1,
            max=img.shape[0] - (slider_radius.value + 1),
            value=img.shape[0] // 2,
        )
        slider_signal_center_y = _create_slider_widget(
            min=slider_radius.value + 1,
            max=img.shape[1] - (slider_radius.value + 1),
            value=img.shape[1] // 2,
        )

        if img.ndim == 3 and img.shape[-1] != 3:
            # Add widget for z coordinate
            slider_signal_center_z = _create_slider_widget(
                min=slider_radius.value + 1,
                max=img.shape[2] - (slider_radius.value + 1),
                value=img.shape[2] // 2,
            )
            slider_signal_center_z.style = {"handle_color": "#2c7bb6"}
        elif img.ndim == 3 and img.shape[-1] == 3:
            img = _to_grayscale(img)

        # Update min and max values of sliders dynamically
        def _update_values(change):
            slider_signal_center_x.min = change.new + 1
            slider_signal_center_x.max = img.shape[0] - (change.new + 1)
            slider_signal_center_y.min = change.new + 1
            slider_signal_center_y.max = img.shape[1] - (change.new + 1)
            try:
                slider_signal_center_z.min = change.new + 1
                slider_signal_center_z.max = img.shape[2] - (change.new + 1)
            except NameError:
                pass

        slider_radius.observe(_update_values, "value")

        # Create button to save values
        save_button = widgets.Button(description="Save Current Values")
        save_button.on_click(_save_values)

        # Visualize centers
        if img.ndim == 3 and (img.shape[-1] != 3):  # 3D image
            # Set style of widgets
            slider_signal_center_x.style = {"handle_color": "#d7191c"}
            slider_signal_center_y.style = {"handle_color": "#fdae61"}

            # Set max value of radius slider
            slider_radius.max = min(img.shape) // 2

            # Create output
            out = widgets.interactive_output(
                _update_visualization_3d,
                {
                    "signal_center_x": slider_signal_center_x,
                    "signal_center_y": slider_signal_center_y,
                    "signal_center_z": slider_signal_center_z,
                    "radius": slider_radius,
                },
            )
            figsize = kwargs.get("figsize", FIGSIZE_SNR_3D_)
            width = figsize[0]
            height = figsize[1]
            out.layout = {
                "width": width,
                "height": height,
            }

            # Create UI
            ui = VBox(
                [
                    HBox(
                        [
                            VBox(
                                [
                                    Label("X Coordinate (Signal)"),
                                    slider_signal_center_x,
                                ]
                            ),
                            VBox(
                                [
                                    Label("Y Coordinate (Signal)"),
                                    slider_signal_center_y,
                                ]
                            ),
                            VBox(
                                [
                                    Label("Z Coordinate (Signal)"),
                                    slider_signal_center_z,
                                ]
                            ),
                        ],
                        layout=widgets.Layout(justify_content="space-around"),
                    ),
                    HBox(
                        [slider_radius, save_button],
                        layout=widgets.Layout(justify_content="center"),
                    ),
                ],
                layout=widgets.Layout(padding="10px 60px"),
            )

            display(out, ui)

        elif img.ndim == 2 or img.ndim == 3 and (img.shape[-1] == 3):  # 2D image
            # Set style of widgets
            slider_signal_center_x.style.handle_color = "#0571b0"
            slider_signal_center_y.style.handle_color = "#92c5de"

            # Set max value of radius slider
            slider_radius.max = min(img.shape[0:-1]) // 2

            # Create output
            out = widgets.interactive_output(
                _update_visualization_2d,
                {
                    "signal_center_x": slider_signal_center_x,
                    "signal_center_y": slider_signal_center_y,
                    "radius": slider_radius,
                },
            )
            figsize = kwargs.get("figsize", FIGSIZE_SNR_2D_)
            width = figsize[0]
            height = figsize[1]
            out.layout = {
                "width": width,
                "height": height,
            }

            # Create UI
            ui = VBox(
                [
                    HBox(
                        [
                            VBox(
                                [
                                    Label("X Coordinate (Signal)"),
                                    slider_signal_center_x,
                                ]
                            ),
                            VBox(
                                [
                                    Label("Y Coordinate (Signal)"),
                                    slider_signal_center_y,
                                ]
                            ),
                        ],
                        layout=widgets.Layout(justify_content="space-around"),
                    ),
                    HBox(
                        [slider_radius, save_button],
                        layout=widgets.Layout(justify_content="center"),
                    ),
                ],
                layout=widgets.Layout(padding="10px 60px"),
            )

            display(out, ui)

        else:
            raise ValueError("No visualization possible for non 2d or non 3d images.")


def signal_to_noise_ratio(img, signal_center, radius, yuv=True):
    """Calculate the signal-to-noise ratio (SNR) for an image.

    Parameters
    ----------
    img : np.ndarray or Tensor or str or os.PathLike
        Image to calculate score of.
    signal_center : Tuple(int)
        Center of the signal. Order is ``(x, y)`` for 2D images and ``(x, y, z)`` for
        3D images.
    radius : int
        Width of the regions.
    yuv : bool, default True

        .. important::
            Only applicable for color images.

        If True, the input images are expected to be RGB images and are converted to YUV
        color space. If False, the input images are kept as RGB images.

    Returns
    -------
    snr_lum : float
        SNR score value for grayscale image.
    snr_val[...] : float, optional
        SNR score values per channel for color image. The order is Y, U, V for YUV
        images and R, G, B for RGB images.

        .. note::
            For RGB images the first return value is the SNR for the luminance channel.

    Raises
    ------
    ValueError
        If the center is not a tuple of integers. \n
        If center is too close to the border. \n
        If the radius is not an integer. \n
        If the image is not 2D or 3D.

    Notes
    -----
    This implementation uses a cubic region to calculate the SNR. The calculation for
    grayscale images is based on the following formula:

    .. math::
       SNR = \\frac{\\mu}{\\sigma}

    where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation.

    For color images, the calculation is a lot more complicated. The image is first
    converted to YUV color space by matrix multiplication with the
    weighting matrix [1]_:

    .. math::
        \\begin{bmatrix}
            Y \\\\
            U \\\\
            V \\\\
        \\end{bmatrix}
        =
        \\begin{bmatrix}
            0.2126 & 0.7152 & 0.0722 \\\\
            -0.09991 & -0.33609 & 0.436 \\\\
            0.615 & -0.55861 & -0.05639 \\\\
        \\end{bmatrix}
        \\begin{bmatrix}
            R \\\\
            G \\\\
            B \\\\
        \\end{bmatrix}

    Then the SNR is calculated for each channel separately [2]_:

    .. math::
        SNR_{channel} = \\frac{\\mu_{Y}}{\\sigma_{channel}}

    where :math:`\\mu_{Y}` is the mean of the Y channel and :math:`\\sigma_{channel}` is
    the standard deviation of the channel for YUV images and:

    .. math::
        SNR_{channel} = \\frac{\\mu_{channel}}{\\sigma_{channel}}

    for RGB images.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YUV
    .. [2] https://www.imatest.com/docs/color-tone-esfriso-noise/#chromanoise
    """
    # check if signal_center is a tuple of integers and radius is an integer
    for center in signal_center:
        if not isinstance(center, int):
            raise TypeError("Center has to be a tuple of integers.")
        if abs(center) - radius < 0:  # check if center is too close to the border
            raise ValueError(
                "Center has to be at least the radius away from the border."
            )

    if not isinstance(radius, int) or radius <= 0:
        raise TypeError("Radius has to be a positive integer.")

    # Check if img and signal_center have the same dimension
    if img.shape[-1] == 3:
        if img.ndim != len(signal_center) + 1:
            raise ValueError("Center has to be in the same dimension as img.")
    else:
        if img.ndim != len(signal_center):
            raise ValueError("Center has to be in the same dimension as img.")

    # Color images
    if img.ndim == 3 and (img.shape[-1] == 3):  # 2D RGB image
        if yuv:
            img = _rgb_to_yuv(img)

        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
            :,
        ]

        sdev = np.std(signal, axis=(0, 1))

        if yuv:
            snr_val = [
                np.mean(signal[..., 0]) / sdev[i] if sdev[i] != 0 else 0
                for i in range(3)
            ]
            return snr_val[0], snr_val[1], snr_val[2]
        else:
            snr_lum = signal_to_noise_ratio(
                _to_grayscale(img), signal_center, radius, yuv=False
            )
            snr_val = [
                np.mean(signal[..., i]) / sdev[i] if sdev[i] != 0 else 0
                for i in range(3)
            ]

        return snr_lum, snr_val[0], snr_val[1], snr_val[2]

    # Define regions
    if img.ndim == 2:  # 2D image
        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
        ]
    elif img.ndim == 3 and (img.shape[-1] != 3):  # 3D image
        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
            signal_center[2] - radius : signal_center[2] + radius,
        ]
    else:
        raise ValueError("Image has to be either 2D or 3D.")

    # Calculate SNR
    if np.std(signal) == 0:
        snr_val = 0
    else:
        snr_val = np.mean(signal) / np.std(signal)

    return snr_val
