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
# Add interactive center selection, 2024, Michael Stidi
# Add automatic center detection, 2024, Michael Stidi
# Update automatic center detection, 2024, Lukas Behammer
#
# License
# -------
# BSD-3-Clause License

from warnings import warn

import numpy as np

from viqa._metrics import NoReferenceMetricsInterface
from viqa.utils import (
    FIGSIZE_CNR_2D,
    FIGSIZE_CNR_3D,
    _check_border_too_close,
    _create_slider_widget,
    _get_binary,
    _to_cubic,
    _to_grayscale,
    _to_spherical,
    _visualize_cnr_2d,
    _visualize_cnr_3d,
    find_largest_region,
    load_data,
)
from viqa.utils._module import check_interactive_vis_deps, is_ipython, try_import

widgets, has_ipywidgets = try_import("ipywidgets")
display, has_ipython = try_import("IPython.display", "display")

glob_signal_center = ()
glob_background_center = ()
glob_radius = None

FIGSIZE_CNR_2D_ = tuple(f"{val}in" for val in FIGSIZE_CNR_2D)
FIGSIZE_CNR_3D_ = tuple(f"{val}in" for val in FIGSIZE_CNR_3D)


class CNR(NoReferenceMetricsInterface):
    """Class to calculate the contrast-to-noise ratio (CNR) for an image.

    Attributes
    ----------
    score_val : float
        CNR score value of the last calculation.
    parameters : dict
        Dictionary containing the parameters for CNR calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True. Passed to :py:func:`viqa.utils.load_data`.
    normalize : bool, default False
        If True, the input images are normalized to the ``data_range`` argument.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`viqa.utils.load_data`.

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
            If ``signal_center``, ``background_center`` and ``radius`` are not given the
            class attribute :py:attr:`parameters` is used.

        Returns
        -------
        score_val : float
            CNR score value.
        """
        img = self.load_images(img)

        # check if signal_center, background_center and radius are provided
        if not {"signal_center", "background_center", "radius"}.issubset(kwargs):
            if not {"signal_center", "background_center", "radius"}.issubset(
                self.parameters.keys()
            ):
                raise ValueError("No center or radius provided.")

            kwargs["signal_center"] = self.parameters["signal_center"]
            kwargs["background_center"] = self.parameters["background_center"]
            kwargs["radius"] = self.parameters["radius"]

        # write kwargs to .parameters attribute
        self.parameters.update(kwargs)

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
            If not given the class attribute :py:attr:`parameters` is used.
        background_center : Tuple(int), optional
            Center of the background. Order is ``(x, y)`` for 2D images and
            ``(x, y, z)`` for 3D images.
            If not given the class attribute :py:attr:`parameters` is used.
        radius : int, optional
            Width of the regions.
            If not given the class attribute :py:attr:`parameters` is used.
        export_path : str or os.PathLike, optional
            Path to export the visualization to.
        **kwargs : optional
            Additional parameters for visualization. The keyword arguments are passed to
            :py:func:`matplotlib.pyplot.subplots`.

        Raises
        ------
        ImportError
            If the visualization fails in a non-interactive environment.
        ValueError
            If the given centers are not in the same dimension as the image.
            If the center is too close to the border.
            If the image is not 2D or 3D.
        TypeError
            If the center is not a tuple of integers.
            If the radius is not a positive integer.
        """
        # Load image
        img = self.load_images(img)

        if not signal_center or not background_center or not radius:
            if not {"signal_center", "background_center", "radius"}.issubset(
                self.parameters.keys()
            ):
                raise ValueError("No center or radius provided.")

            signal_center = self.parameters["signal_center"]
            background_center = self.parameters["background_center"]
            radius = self.parameters["radius"]

        # Check if img and signal_center have the same dimension
        if img.shape[-1] == 3:
            if (
                img.ndim != len(signal_center) + 1
                or img.ndim != len(background_center) + 1
            ):
                raise ValueError("Centers have to be in the same dimension as img.")
        else:
            if img.ndim != len(signal_center) or img.ndim != len(background_center):
                raise ValueError("Centers have to be in the same dimension as img.")

        # Check if radius is an integer and positive
        if not isinstance(radius, int) or radius <= 0:
            raise TypeError("Radius has to be an integer.")

        # check if signal_center and background_center are tuples of integers and not
        # too close to the border
        _check_border_too_close(signal_center, radius)
        _check_border_too_close(background_center, radius)

        # Visualize centers
        if img.ndim == 3 and img.shape[-1] != 3:  # 3D image
            _visualize_cnr_3d(
                img=img,
                signal_center=signal_center,
                background_center=background_center,
                radius=radius,
                export_path=export_path,
                **kwargs,
            )
        elif img.ndim == 3 and img.shape[-1] == 3:  # 2D RGB image
            img = _to_grayscale(img)
            _visualize_cnr_2d(
                img=img,
                signal_center=signal_center,
                background_center=background_center,
                radius=radius,
                export_path=export_path,
                **kwargs,
            )
        elif img.ndim == 2:  # 2D image
            _visualize_cnr_2d(
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
            :py:meth:`visualize_centers` and :py:func:`matplotlib.pyplot.subplots`.
            ``background_center``, ``signal_center``, and ``radius`` can be provided as
            starting points for the interactive center selection. If not provided, the
            center of the image is used as the starting point.

            .. caution::
                The class attribute :py:attr:`parameters` takes precedence over the
                given parameters.

        Warnings
        --------
        This method is only available in an IPython environment. If not in an IPython
        environment, the method will try to visualize the centers in a non-interactive
        environment.

        Raises
        ------
        ImportError
            If the visualization fails in a non-interactive environment.
        ValueError
            If the given centers are not in the same dimension as the image.
            If the center is too close to the border.
            If the image is not 2D or 3D.
        TypeError
            If the center is not a tuple of integers.
            If the radius is not a positive integer.
        """
        if not is_ipython():
            try:
                warn("Trying to visualize in a non-interactive environment.")
                self.visualize_centers(img, **kwargs)
                return
            except Exception:
                raise ImportError(
                    "Failed to visualize in a non-interactive "
                    "environment. Please use an IPython environment or try giving "
                    "the parameters for the centers and radius directly."
                ) from None

        check_interactive_vis_deps(has_ipywidgets, has_ipython)

        # Load image
        img = self.load_images(img)

        # Prepare visualization functions and widgets

        # Define output layout
        def _output_layout(out, figsize_variable):
            figsize = kwargs.get("figsize", figsize_variable)
            width = figsize[0]
            height = figsize[1]
            out.layout = {
                "width": width,
                "height": height,
            }

        # Define function to save values from global variables to .parameters attribute
        def _save_values(_):
            global glob_signal_center, glob_background_center, glob_radius
            self.parameters.update(
                {
                    "signal_center": glob_signal_center,
                    "background_center": glob_background_center,
                    "radius": glob_radius,
                }
            )

        # Define function to save values to global variables
        def _write_values_to_global(signal_value, background_value, radius_value):
            # Set global variables
            global glob_signal_center, glob_background_center, glob_radius
            glob_signal_center = signal_value
            glob_background_center = background_value
            glob_radius = radius_value

        # Define functions for visualization
        def _update_visualization_2d(
            signal_center_x,
            signal_center_y,
            background_center_x,
            background_center_y,
            radius,
        ):
            signal_center = (signal_center_x, signal_center_y)
            background_center = (background_center_x, background_center_y)

            _write_values_to_global(signal_center, background_center, radius)

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

            _write_values_to_global(signal_center, background_center, radius)

            _visualize_cnr_3d(
                img=img,
                signal_center=signal_center,
                background_center=background_center,
                radius=radius,
                **kwargs,
            )

        # Check if img is 2D RGB image
        if img.ndim == 3 and img.shape[-1] == 3:
            img = _to_grayscale(img)

        center_point = tuple(val // 2 for val in img.shape)
        # Check if background_center and signal_center are provided
        if {"background_center", "signal_center", "radius"}.issubset(
            self.parameters.keys()
        ):
            background_start = self.parameters["background_center"]
            signal_start = self.parameters["signal_center"]
            radius_start = self.parameters["radius"]
        else:
            background_start = kwargs.pop("background_center", center_point)
            signal_start = kwargs.pop("signal_center", center_point)
            radius_start = kwargs.pop("radius", 1)

        _write_values_to_global(signal_start, background_start, radius_start)

        global glob_signal_center, glob_background_center, glob_radius
        # Check if background_center and signal_center are the right shape for 2d and 3d
        if img.ndim != len(glob_signal_center) or img.ndim != len(
            glob_background_center
        ):
            raise ValueError("Centers have to be in the same dimension as img.")

        # Check if radius is an integer and positive
        if not isinstance(glob_radius, int) or glob_radius <= 0:
            raise TypeError("Radius has to be an integer.")

        # check if signal_center and background_center are tuples of integers and not
        # too close to the border
        _check_border_too_close(glob_signal_center, glob_radius)
        _check_border_too_close(glob_background_center, glob_radius)

        # Write values to attributes
        _save_values(None)

        # Create slider for radius
        slider_radius = _create_slider_widget(
            max=min(img.shape) // 2,
            min=1,
            value=radius_start,
            description="Radius",
        )
        slider_radius.style = {"handle_color": "#f7f7f7"}

        # Create sliders for background center coordinates
        slider_background_center_x = _create_slider_widget(
            min=slider_radius.value + 1,
            max=img.shape[0] - (slider_radius.value + 1),
            value=background_start[0],
        )
        slider_background_center_y = _create_slider_widget(
            min=slider_radius.value + 1,
            max=img.shape[1] - (slider_radius.value + 1),
            value=background_start[1],
        )

        # Create sliders for signal center coordinates
        slider_signal_center_x = _create_slider_widget(
            min=slider_radius.value + 1,
            max=img.shape[0] - (slider_radius.value + 1),
            value=signal_start[0],
        )
        slider_signal_center_y = _create_slider_widget(
            min=slider_radius.value + 1,
            max=img.shape[1] - (slider_radius.value + 1),
            value=signal_start[1],
        )

        if img.ndim == 3 and img.shape[-1] != 3:
            # Add widgets for z coordinate
            slider_background_center_z = _create_slider_widget(
                min=slider_radius.value + 1,
                max=img.shape[2] - (slider_radius.value + 1),
                value=background_start[2],
            )
            slider_signal_center_z = _create_slider_widget(
                min=slider_radius.value + 1,
                max=img.shape[2] - (slider_radius.value + 1),
                value=signal_start[2],
            )
            slider_background_center_z.style = {"handle_color": "#2c7bb6"}
            slider_signal_center_z.style = {"handle_color": "#2c7bb6"}

        # Update min and max values of sliders dynamically
        def _update_values(change):
            slider_background_center_x.min = change.new + 1
            slider_background_center_x.max = img.shape[0] - (change.new + 1)
            slider_background_center_y.min = change.new + 1
            slider_background_center_y.max = img.shape[1] - (change.new + 1)
            slider_signal_center_x.min = change.new + 1
            slider_signal_center_x.max = img.shape[0] - (change.new + 1)
            slider_signal_center_y.min = change.new + 1
            slider_signal_center_y.max = img.shape[1] - (change.new + 1)
            try:
                slider_background_center_z.min = change.new + 1
                slider_background_center_z.max = img.shape[2] - (change.new + 1)
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
            slider_background_center_x.style = {"handle_color": "#d7191c"}
            slider_background_center_y.style = {"handle_color": "#fdae61"}
            slider_signal_center_x.style = {"handle_color": "#d7191c"}
            slider_signal_center_y.style = {"handle_color": "#fdae61"}

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
            _output_layout(out, FIGSIZE_CNR_3D_)

            # Create UI
            ui = widgets.VBox(
                [
                    widgets.HBox(
                        [
                            widgets.VBox(
                                [
                                    widgets.Label("X Coordinate (Background)"),
                                    slider_background_center_x,
                                ]
                            ),
                            widgets.VBox(
                                [
                                    widgets.Label("Y Coordinate (Background)"),
                                    slider_background_center_y,
                                ]
                            ),
                            widgets.VBox(
                                [
                                    widgets.Label("Z Coordinate (Background)"),
                                    slider_background_center_z,
                                ]
                            ),
                        ],
                        layout=widgets.Layout(justify_content="space-around"),
                    ),
                    widgets.HBox(
                        [
                            widgets.VBox(
                                [
                                    widgets.Label("X Coordinate (Signal)"),
                                    slider_signal_center_x,
                                ]
                            ),
                            widgets.VBox(
                                [
                                    widgets.Label("Y Coordinate (Signal)"),
                                    slider_signal_center_y,
                                ]
                            ),
                            widgets.VBox(
                                [
                                    widgets.Label("Z Coordinate (Signal)"),
                                    slider_signal_center_z,
                                ]
                            ),
                        ],
                        layout=widgets.Layout(justify_content="space-around"),
                    ),
                    widgets.HBox(
                        [slider_radius, save_button],
                        layout=widgets.Layout(justify_content="center"),
                    ),
                ],
                layout=widgets.Layout(padding="10px 60px"),
            )

            display(out, ui)

        elif img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 3):  # 2D image
            # Set style of widgets
            slider_background_center_x.style = {"handle_color": "#ca0020"}
            slider_background_center_y.style = {"handle_color": "#f4a582"}
            slider_signal_center_x.style = {"handle_color": "#0571b0"}
            slider_signal_center_y.style = {"handle_color": "#92c5de"}

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
            _output_layout(out, FIGSIZE_CNR_2D_)

            # Create UI
            ui = widgets.VBox(
                [
                    widgets.HBox(
                        [
                            widgets.VBox(
                                [
                                    widgets.Label("X Coordinate (Background)"),
                                    slider_background_center_x,
                                ]
                            ),
                            widgets.VBox(
                                [
                                    widgets.Label("X Coordinate (Signal)"),
                                    slider_signal_center_x,
                                ]
                            ),
                        ],
                        layout=widgets.Layout(justify_content="space-around"),
                    ),
                    widgets.HBox(
                        [
                            widgets.VBox(
                                [
                                    widgets.Label("Y Coordinate (Background)"),
                                    slider_background_center_y,
                                ]
                            ),
                            widgets.VBox(
                                [
                                    widgets.Label("Y Coordinate (Signal)"),
                                    slider_signal_center_y,
                                ]
                            ),
                        ],
                        layout=widgets.Layout(justify_content="space-around"),
                    ),
                    widgets.HBox(
                        [slider_radius, save_button],
                        layout=widgets.Layout(justify_content="center"),
                    ),
                ],
                layout=widgets.Layout(padding="10px 60px"),
            )

            display(out, ui)

        else:
            raise ValueError("No visualization possible for non 2d or non 3d images.")


def contrast_to_noise_ratio(
    img,
    background_center,
    signal_center,
    radius,
    region_type="cubic",
    auto_center=False,
    iterations=5,
    **kwargs,
):
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
    region_type : {'cubic', 'spherical', 'full', 'original'}, optional
        Type of region to calculate the SNR. Default is 'cubic'.
        Gets passed to :py:func:`viqa.utils.find_largest_region` if `auto_center` is
        True. If `auto_center` is False, the following options are available:
        If 'full' or 'original' the original image is used as signal and background
        region.
        If 'cubic' a cubic region around the center is used. Alias for 'cubic' are
        'cube' and 'square'.
        If 'spherical' a spherical region around the center is used. Alias for
        'spherical' are 'sphere' and 'circle'.
    auto_center : bool, default False
        Automatically find the center of the image. `signal_center`,
        `background_center` and `radius` are ignored if True.
    iterations : int, optional
        Number of iterations for morphological operations if `auto_center` is True.
        Default is 5.
    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`viqa.utils.load_data`.

    Returns
    -------
    score_val : float
        CNR score value.

    Raises
    ------
    ValueError
        If the input image is not 2D or 3D.
        If the input center is not a tuple of integers.
        If the input center is too close to the border.
        If the input radius is not an integer.
        If the passed region type is not valid.

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
        2010: Physics of Medical Imaging, 7622, 76224Q.
        https://doi.org/10.1117/12.844640
    """
    img = load_data(img, **kwargs)

    if auto_center is True:
        warn(
            "Signal and background center are automatically detected. Parameters "
            "signal_center, background_center and radius are ignored."
        )

        binary_foreground = _get_binary(
            img, lower_threshold=90, upper_threshold=100, show=False
        )
        binary_background = _get_binary(
            img, lower_threshold=0, upper_threshold=90, show=False
        )

        signal_center, signal_radius, signal_region = find_largest_region(
            img=binary_foreground, region_type=region_type, iterations=iterations
        )
        background_center, background_radius, background_region = find_largest_region(
            img=binary_background, region_type=region_type, iterations=iterations
        )
        radius = min(signal_radius, background_radius)

        # Mask the original image with the regions
        if region_type not in {
            "original",
            "cubic",
            "cube",
            "square",
            "spherical",
            "sphere",
            "circle",
        }:
            signal = np.ma.array(img, mask=~signal_region, copy=True)
            background = np.ma.array(img, mask=~background_region, copy=True)
        elif region_type == "original":
            signal = img
            background = img

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
    if region_type in {"full", "original"}:
        if not auto_center:
            signal = img
            background = img
    elif region_type in {"cubic", "cube", "square"}:
        signal = _to_cubic(img, signal_center, radius)
        background = _to_cubic(img, background_center, radius)
    elif region_type in {"spherical", "sphere", "circle"}:
        signal = _to_spherical(img, signal_center, radius)
        background = _to_spherical(img, background_center, radius)
    else:
        if not auto_center:
            raise ValueError("Region type not valid.")

    # Calculate CNR
    if np.std(background) == 0:
        cnr_val = 0
    else:
        cnr_val = (np.mean(signal) - np.mean(background)) / np.std(background)

    return np.float64(cnr_val)
