"""Module for calculating the contrast-to-noise ratio (CNR) for an image.

Examples
--------
    .. doctest-requires:: numpy

        >>> import numpy as np
        >>> from viqa import CNR
        >>> img = np.random.rand(256, 256)
        >>> cnr = CNR(data_range=1, normalize=False)
        >>> cnr
        CNR(score_val=None)
        >>> score = cnr.score(img,
        ...                   background_center=(128, 128),
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
from ipywidgets import HBox, VBox

from viqa._metrics import NoReferenceMetricsInterface
from viqa.visualization_utils import _visualize_cnr_2d, _visualize_cnr_3d

glob_signal_center = None
glob_background_center = None
glob_radius = None


class CNR(NoReferenceMetricsInterface):
    """Class to calculate the contrast-to-noise ratio (CNR) for an image.

    Attributes
    ----------
    score_val : float
        CNR score value of the last calculation.

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
        self._name = "CNR"

    def score(self, img, **kwargs):
        """Calculate the contrast-to-noise ratio (CNR) for an image.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to calculate score of.
        **kwargs : optional
            Additional parameters for CNR calculation. The keyword arguments are passed
            to :py:func:`.viqa.nr_metrics.cnr.contrast_to_noise_ratio`.

        Returns
        -------
        score_val : float
            CNR score value.
        """
        img = super().score(img)

        # write kwargs to ._parameters attribute
        self._parameters.update(kwargs)

        score_val = contrast_to_noise_ratio(img, **kwargs)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the CNR score value of the last calculation.

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
            print("CNR: {}".format(np.round(self.score_val, decimals)))
        else:
            warn("No score value for CNR. Run score() first.", RuntimeWarning)

    def visualize_centers(
        self,
        img,
        signal_center=None,
        background_center=None,
        radius=None,
        export_path=None,
        **kwargs,
    ):
        """Visualize the centers for CNR calculation.

        The visualization shows the signal and background regions in a matplotlib plot.
        If export_path is provided, the plot is saved to the given path.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to visualize.
        signal_center : Tuple(int), optional
            Center of the signal.
            Order is ``(x, y)`` for 2D images and ``(x, y, z)`` for 3D images.
        background_center : Tuple(int), optional
            Center of the background. Order is ``(x, y)`` for 2D images and
            ``(x, y, z)`` for 3D images.
        radius : int, optional
            Width of the regions.
        export_path : str or os.PathLike, optional
            Path to export the visualization to.
        **kwargs : optional
            Additional parameters for visualization. The keyword arguments are passed to
            :py:func:`matplotlib.pyplot.subplots`.
        """
        if not signal_center or not background_center or not radius:
            if (
                not self._parameters["signal_center"]
                or not self._parameters["background_center"]
                or not self._parameters["radius"]
            ):
                raise ValueError("No center or radius provided.")

            signal_center = self._parameters["signal_center"]
            background_center = self._parameters["background_center"]
            radius = self._parameters["radius"]

        if img.ndim != len(signal_center) or img.ndim != len(background_center):
            raise ValueError("Centers have to be in the same dimension as img.")

        if img.ndim == 2:
            _visualize_cnr_2d(
                img=img,
                signal_center=signal_center,
                background_center=background_center,
                radius=radius,
                export_path=export_path,
                **kwargs,
            )
        elif img.ndim == 3:
            _visualize_cnr_3d(
                img=img,
                signal_center=signal_center,
                background_center=background_center,
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
        """Visualize and set the centers for CNR calculation interactively.

        The visualization shows the signal and background regions in a matplotlib plot.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to visualize.
        **kwargs : optional
            Additional parameters for visualization. The keyword arguments are passed to
            :py:func:`matplotlib.pyplot.subplots`.
        """

        # Prepare visualization functions and widgets
        def _update_visualization_2d(
            signal_center_x,
            signal_center_y,
            background_center_x,
            background_center_y,
            radius,
        ):
            signal_center = (signal_center_x, signal_center_y)
            background_center = (background_center_x, background_center_y)

            global glob_signal_center, glob_background_center, glob_radius
            glob_signal_center = signal_center
            glob_background_center = background_center
            glob_radius = radius

            _visualize_cnr_2d(
                img=img,
                signal_center=signal_center,
                background_center=background_center,
                radius=radius,
                **kwargs,
            )

        def _update_visualization_3d(
            signal_center_x,
            signal_center_y,
            signal_center_z,
            background_center_x,
            background_center_y,
            background_center_z,
            radius,
        ):
            signal_center = (
                signal_center_x,
                signal_center_y,
                signal_center_z,
            )
            background_center = (
                background_center_x,
                background_center_y,
                background_center_z,
            )

            global glob_signal_center, glob_background_center, glob_radius
            glob_signal_center = signal_center
            glob_background_center = background_center
            glob_radius = radius

            _visualize_cnr_3d(
                img=img,
                signal_center=signal_center,
                background_center=background_center,
                radius=radius,
                **kwargs,
            )

        def _save_values(_):
            global glob_signal_center, glob_background_center, glob_radius
            self._parameters.update(
                {
                    "signal_center": glob_signal_center,
                    "background_center": glob_background_center,
                    "radius": glob_radius,
                }
            )
            print("Parameters saved.")

        # Create sliders for background center coordinates
        slider_background_center_x = widgets.IntSlider(
            min=0,
            max=img.shape[0] - 1,
            step=1,
            value=img.shape[0] // 2,
            description="X Coordinate (Background)",
            continuous_update=False,
        )
        slider_background_center_y = widgets.IntSlider(
            min=0,
            max=img.shape[1] - 1,
            step=1,
            value=img.shape[1] // 2,
            description="Y Coordinate (Background)",
            continuous_update=False,
        )
        slider_background_center_z = widgets.IntSlider(
            min=0,
            max=img.shape[2] - 1,
            step=1,
            value=img.shape[2] // 2,
            description="Z Coordinate (Background)",
            continuous_update=False,
        )

        # Create sliders for signal center coordinates
        slider_signal_center_x = widgets.IntSlider(
            min=0,
            max=img.shape[0] - 1,
            step=1,
            value=img.shape[0] // 2,
            description="X Coordinate (Signal)",
            continuous_update=False,
        )
        slider_signal_center_y = widgets.IntSlider(
            min=0,
            max=img.shape[1] - 1,
            step=1,
            value=img.shape[1] // 2,
            description="Y Coordinate (Signal)",
            continuous_update=False,
        )
        slider_signal_center_z = widgets.IntSlider(
            min=0,
            max=img.shape[2] - 1,
            step=1,
            value=img.shape[2] // 2,
            description="Z Coordinate (Signal)",
            continuous_update=False,
        )

        # Create slider for radius
        slider_radius = widgets.IntSlider(
            min=1,
            max=min(img.shape[0], img.shape[1], img.shape[2]) // 2,
            step=1,
            value=1,
            description="Radius",
            continuous_update=False,
        )

        # Create button to save values
        save_button = widgets.Button(description="Save Current Values")
        save_button.on_click(_save_values)

        # Set style of widgets
        slider_signal_center_x.style.handle_color = "#d7191c"
        slider_signal_center_y.style.handle_color = "#fdae61"
        slider_signal_center_z.style.handle_color = "#2c7bb6"
        slider_background_center_x.style.handle_color = "#d7191c"
        slider_background_center_y.style.handle_color = "#fdae61"
        slider_background_center_z.style.handle_color = "#2c7bb6"
        slider_radius.style.handle_color = "#f7f7f7"

        if img.ndim == 2:
            # Create output
            out = widgets.interactive_output(
                _update_visualization_2d,
                {
                    "background_center_x": slider_background_center_x,
                    "background_center_y": slider_background_center_y,
                    "signal_center_x": slider_signal_center_x,
                    "signal_center_y": slider_signal_center_y,
                    "radius": slider_radius,
                },
            )

            # Create UI
            ui = VBox(
                [
                    HBox(
                        [
                            slider_background_center_x,
                            slider_background_center_y,
                        ],
                        layout=widgets.Layout(justify_content="space-between"),
                    ),
                    HBox(
                        [
                            slider_signal_center_x,
                            slider_signal_center_y,
                        ],
                        layout=widgets.Layout(justify_content="space-between"),
                    ),
                    HBox(
                        [slider_radius, save_button],
                        layout=widgets.Layout(justify_content="space-between"),
                    ),
                ],
                layout=widgets.Layout(padding="10px 60px"),
            )

            display(out, ui)

        elif img.ndim == 3:
            # Create output
            out = widgets.interactive_output(
                _update_visualization_3d,
                {
                    "background_center_x": slider_background_center_x,
                    "background_center_y": slider_background_center_y,
                    "background_center_z": slider_background_center_z,
                    "signal_center_x": slider_signal_center_x,
                    "signal_center_y": slider_signal_center_y,
                    "signal_center_z": slider_signal_center_z,
                    "radius": slider_radius,
                },
            )

            # Create UI
            ui = VBox(
                [
                    HBox(
                        [
                            slider_background_center_x,
                            slider_background_center_y,
                            slider_background_center_z,
                        ],
                        layout=widgets.Layout(justify_content="space-around"),
                    ),
                    HBox(
                        [
                            slider_signal_center_x,
                            slider_signal_center_y,
                            slider_signal_center_z,
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


def contrast_to_noise_ratio(img, background_center, signal_center, radius):
    """Calculate the contrast-to-noise ratio (CNR) for an image.

    Parameters
    ----------
    img : np.ndarray or Tensor or str or os.PathLike
        Image to calculate score of.
    background_center : Tuple(int)
        Center of the background. Order is ``(x, y)`` for 2D images and ``(x, y, z)``
        for 3D images.
    signal_center : Tuple(int)
        Center of the signal. Order is ``(x, y)`` for 2D images and ``(x, y, z)`` for
        3D images.
    radius : int
        Width of the regions.

    Returns
    -------
    score_val : float
        CNR score value.

    Raises
    ------
    ValueError
        If the input image is not 2D or 3D. \n
        If the input center is not a tuple of integers. \n
        If the input center is too close to the border. \n
        If the input radius is not an integer.

    Notes
    -----
    This implementation uses cubic regions to calculate the CNR. The calculation is
    based on the following formula:

    .. math::
        CNR = \\frac{\\mu_{signal} - \\mu_{background}}{\\sigma_{background}}

    where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation.

    .. important::
        The background region should be chosen in a homogeneous area, while the signal
        region should be chosen in an area with a high contrast.

    References
    ----------
    .. [1] Desai, N., Singh, A., & Valentino, D. J. (2010). Practical evaluation of
        image quality in computed radiographic (CR) imaging systems. Medical Imaging
        2010: Physics of Medical Imaging, 7622, 76224Q. https://doi.org/10.1117/12.844640
    """
    # check if signal_center and background_center are tuples of integers and radius is
    # an integer
    for center in signal_center:
        if not isinstance(center, int):
            raise TypeError("Signal center has to be a tuple of integers.")
        if abs(center) - radius < 0:  # check if center is too close to the border
            raise ValueError(
                "Signal center has to be at least the radius away from the border."
            )

    for center in background_center:
        if not isinstance(center, int):
            raise TypeError("Background center has to be a tuple of integers.")
        if abs(center) - radius < 0:
            raise ValueError(
                "Background center has to be at least the radius away from the border."
            )

    if not isinstance(radius, int) or radius <= 0:
        raise TypeError("Radius has to be an integer.")

    # Check if img and centers have the same dimension
    if img.ndim != len(signal_center) or img.ndim != len(background_center):
        raise ValueError("Centers have to be in the same dimension as img.")

    # Define regions
    if img.ndim == 2:  # 2D image
        background = img[
            background_center[0] - radius : background_center[0] + radius,
            background_center[1] - radius : background_center[1] + radius,
        ]
        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
        ]
    elif img.ndim == 3:  # 3D image
        background = img[
            background_center[0] - radius : background_center[0] + radius,
            background_center[1] - radius : background_center[1] + radius,
            background_center[2] - radius : background_center[2] + radius,
        ]
        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
            signal_center[2] - radius : signal_center[2] + radius,
        ]
    else:
        raise ValueError("Image has to be either 2D or 3D.")

    # Calculate CNR
    if np.std(background) == 0:
        cnr_val = 0
    else:
        cnr_val = (np.mean(signal) - np.mean(background)) / np.std(background)

    return cnr_val
