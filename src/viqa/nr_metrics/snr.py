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
    FIGSIZE_SNR_2D,
    FIGSIZE_SNR_3D,
    _check_border_too_close,
    _create_slider_widget,
    _get_binary,
    _rgb_to_yuv,
    _to_cubic,
    _to_grayscale,
    _to_spherical,
    _visualize_snr_2d,
    _visualize_snr_3d,
    find_largest_region,
    load_data,
)
from viqa.utils._module import check_interactive_vis_deps, is_ipython, try_import

widgets, has_ipywidgets = try_import("ipywidgets")
display, has_ipython = try_import("IPython.display", "display")

glob_signal_center = ()
glob_radius = None
glob_region_type = None
glob_iterations = None
glob_lower_threshold = None
glob_upper_threshold = None
glob_signal = None


class SNR(NoReferenceMetricsInterface):
    """Class to calculate the signal-to-noise ratio (SNR) for an image.

    Attributes
    ----------
    score_val : float
        SNR score value of the last calculation.
    parameters : dict
        Dictionary containing the parameters for SNR calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True. Passed to :py:func:`viqa.utils.load_data`.
    normalize : bool, default False
        If True, the input images are normalized to the ``data_range`` argument.
    name : str, default="SNR"
        Name of the metric.

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

    def __init__(self, data_range=255, normalize=False, name="SNR", **kwargs) -> None:
        """Construct method."""
        super().__init__(
            data_range=data_range, normalize=normalize, name=name, **kwargs
        )

    def score(self, img, **kwargs):
        """Calculate the signal-to-noise ratio (SNR) for an image.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to calculate score of.
        **kwargs : optional
            Additional parameters for SNR calculation. The keyword arguments are passed
            to :py:func:`viqa.nr_metrics.snr.signal_to_noise_ratio`.
            If ``signal_center`` and ``radius`` are not given the class attribute
            :py:attr:`parameters` is used.

        Returns
        -------
        score_val : float
            SNR score value.
        """
        # TODO: Check this function: maybe other parameters from attribute;
        #  what happens when auto_center is True but region_type != "full"
        img = self.load_images(img)

        # check if signal_center and radius are provided
        if not {"signal_center", "radius"}.issubset(kwargs):
            if not {"signal_center", "radius"}.issubset(self.parameters.keys()):
                raise ValueError("No center or radius provided.")

            kwargs["signal_center"] = self.parameters["signal_center"]
            kwargs["radius"] = self.parameters["radius"]

        # TODO: Write standard params of signal to noise ratio to .parameters attribute
        #  if not already done

        # write kwargs to .parameters attribute
        self.parameters.update(kwargs)

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
        self,
        img,
        signal_center=None,
        radius=None,
        region_type=None,
        signal=None,
        export_path=None,
        **kwargs,
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
            If not given the class attribute :py:attr:`parameters` is used.
        radius : int, optional
            Width of the regions.
            If not given the class attribute :py:attr:`parameters` is used.
        region_type : {'cubic', 'spherical', 'full', 'original'}, optional
            Type of region to visualize.
            If not given the class attribute :py:attr:`parameters` is used.

            .. note::
                See :py:func:`viqa.utils.find_largest_region` for more information on
                the possible region types.

        signal : np.ndarray, optional
            Region to visualize. If not given, the region is visualized based on the
            given parameters.
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
            If the given signal_center is not in the same dimension as the image.
            If the center is too close to the border.
            If the image is not 2D or 3D.
            If no region type is provided.
            If the passed region type is not valid.
            If no center or radius is provided when region type is not 'full' or
            'original'.
        TypeError
            If the center is not a tuple of integers.
            If the radius is not a positive integer.
        """
        # Load image
        img = self.load_images(img)

        # Check if parameters are provided
        if not region_type:
            try:
                region_type = self.parameters["region_type"]
            except KeyError:
                raise ValueError("No region type provided.") from None
        if not signal:
            try:
                signal = self._region
            except AttributeError:
                signal = None

        if signal is None and region_type not in {"full", "original"}:
            if not signal_center:
                try:
                    signal_center = self.parameters["signal_center"]
                except KeyError:
                    raise ValueError("No center provided.") from None
                if signal_center is None:
                    raise ValueError("No center provided.")
            if not radius:
                try:
                    radius = self.parameters["radius"]
                except KeyError:
                    raise ValueError("No radius provided.") from None
                if radius is None:
                    raise ValueError("No radius provided.")

            # Check if img and signal_center have the same dimension
            if img.shape[-1] == 3:
                if img.ndim != len(signal_center) + 1:
                    raise ValueError("Center has to be in the same dimension as img.")
            else:
                if img.ndim != len(signal_center):
                    raise ValueError("Center has to be in the same dimension as img.")

            # check if radius is an integer and positive
            if not isinstance(radius, int) or radius <= 0:
                raise TypeError("Radius has to be a positive integer.")

            # check if signal_center is a tuple of integers
            # and not too close to the border
            _check_border_too_close(signal_center, radius)
        else:
            if not signal_center:
                # Center of image
                signal_center = tuple(val // 2 for val in img.shape)

        # Visualize centers
        if img.ndim == 3 and (img.shape[-1] != 3):  # 3D image
            _visualize_snr_3d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                region_type=region_type,
                signal=signal,
                export_path=export_path,
                **kwargs,
            )
        elif img.ndim == 3 and (img.shape[-1] == 3):  # 2D RGB image
            img = _to_grayscale(img)
            _visualize_snr_2d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                region_type=region_type,
                signal=signal,
                export_path=export_path,
                **kwargs,
            )
        elif img.ndim == 2:  # 2D image
            _visualize_snr_2d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                region_type=region_type,
                signal=signal,
                export_path=export_path,
                **kwargs,
            )
        else:
            raise ValueError("No visualization possible for non 2d or non 3d images.")

    def set_centers(
        self,
        img,
        vis_kwargs=None,
        **kwargs,
    ):
        """Visualize and set the centers for SNR calculation interactively.

        The visualization shows the signal region in a matplotlib plot.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to visualize.
        vis_kwargs: dict, optional
            Additional parameters for visualization. The keyword arguments are passed to
            :py:meth:`visualize_centers` and :py:func:`matplotlib.pyplot.subplots`.
        **kwargs : optional
            Additional parameters as starting points for the interactive center
            selection. ``signal_center``, ``radius`` and ``region_type`` can be provided
            as starting points for the interactive center selection. If not provided,
            the center of the image is used as the starting point.
            If the region should be calculated automatically, the following parameters
            can also be given as starting points:
            - ``iterations``: Number of iterations for morphological operations.
            - ``lower_threshold``: Lower threshold for the binary image.
            - ``upper_threshold``: Upper threshold for the binary image.

            If not provided, the class attribute :py:attr:`parameters` is used.
            If the class attribute is not set, default values are used.

        Notes
        -----
        To calculate the region automatically takes some time after clicking the
        ``Calculate Full Region`` button until the image is updated.

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
            If the given signal_center is not in the same dimension as the image.
            If the center is too close to the border.
            If the image is not 2D or 3D.
            If region type is not valid.
        TypeError
            If the center is not a tuple of integers.
            If the radius is not a positive integer.
        """
        # Load image
        img = self.load_images(img)

        # Remove parameters from kwargs
        _ = kwargs.pop("auto_center", None)
        _ = kwargs.pop("yuv", None)

        if vis_kwargs is None:
            vis_kwargs = {}

        if not is_ipython():
            try:
                warn("Trying to visualize in a non-interactive environment.")
                self.visualize_centers(img, **vis_kwargs)
                return
            except Exception:
                raise ImportError(
                    "Failed to visualize in a non-interactive "
                    "environment. Please use an IPython environment or try giving "
                    "the parameters for the center and radius directly."
                ) from None

        check_interactive_vis_deps(has_ipywidgets, has_ipython)

        # Prepare visualization functions and widgets

        # Define output layout
        def _output_layout(out, figsize):
            figsize = list(vis_kwargs.get("figsize", figsize))
            figsize[1] += 2
            figsize = tuple(f"{val}in" for val in figsize)
            width = figsize[0]
            height = figsize[1]
            out.layout = {
                "width": width,
                "height": height,
            }

        # Define function to save values from global variables to .parameters attribute
        def _save_values(_):
            global \
                glob_signal, \
                glob_signal_center, \
                glob_radius, \
                glob_region_type, \
                glob_iterations, \
                glob_lower_threshold, \
                glob_upper_threshold
            if glob_region_type == "full":
                parameters = {
                    "region_type": glob_region_type,
                    "iterations": glob_iterations,
                    "lower_threshold": glob_lower_threshold,
                    "upper_threshold": glob_upper_threshold,
                    "auto_center": True,
                    "signal_center": None,
                    "radius": None,
                }
                self._region = glob_signal
            elif glob_region_type in ["cubic", "spherical"]:
                parameters = {
                    "signal_center": glob_signal_center,
                    "radius": glob_radius,
                    "region_type": glob_region_type,
                    "auto_center": False,
                    "iterations": None,
                    "lower_threshold": None,
                    "upper_threshold": None,
                }
            else:
                parameters = {
                    "region_type": glob_region_type,
                    "auto_center": False,
                    "iterations": None,
                    "lower_threshold": None,
                    "upper_threshold": None,
                    "signal_center": None,
                    "radius": None,
                }

            self.parameters.update(parameters)

            # if glob_signal is not None and glob_region_type == "full":
            #     parameters["signal"] = glob_signal
            #
            # return parameters

        # Calculate region
        def _calculate_region(_):
            global \
                glob_region_type, \
                glob_signal, \
                glob_iterations, \
                glob_lower_threshold, \
                glob_upper_threshold

            binary_foreground = _get_binary(
                img,
                lower_threshold=lower_threshold_slider.value,
                upper_threshold=upper_threshold_slider.value,
                show=False,
            )

            _, _, signal_region = find_largest_region(
                img=binary_foreground,
                region_type=glob_region_type,
                iterations=iterations_slider.value,
            )

            glob_signal = np.ma.array(img, mask=~signal_region, copy=True)
            glob_iterations = iterations_slider.value
            glob_lower_threshold = lower_threshold_slider.value
            glob_upper_threshold = upper_threshold_slider.value
            # TODO: Change to automatic visualization update
            dropdown_region_type.value = "full"

        # Define function to save values to global variables
        def _write_values_to_global(signal_value, radius_value, region_type):
            # Set global variables
            global glob_signal_center, glob_radius, glob_region_type
            glob_signal_center = signal_value
            glob_radius = radius_value
            glob_region_type = region_type

        # Define functions for visualization
        def _update_visualization_2d(
            signal_center_x,
            signal_center_y,
            radius,
            region_type,
        ):
            signal_center = (signal_center_x, signal_center_y)
            figsize = vis_kwargs.get("figsize", FIGSIZE_SNR_2D)
            kwargs.update({"figsize": figsize})

            _write_values_to_global(signal_center, radius, region_type)

            global glob_signal
            _visualize_snr_2d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                region_type=region_type,
                signal=glob_signal,
                **vis_kwargs,
            )

        def _update_visualization_3d(
            signal_center_x,
            signal_center_y,
            signal_center_z,
            radius,
            region_type,
        ):
            signal_center = (
                signal_center_x,
                signal_center_y,
                signal_center_z,
            )
            figsize = vis_kwargs.get("figsize", FIGSIZE_SNR_3D)
            kwargs.update({"figsize": figsize})

            _write_values_to_global(signal_center, radius, region_type)

            global glob_signal
            _visualize_snr_3d(
                img=img,
                signal_center=signal_center,
                radius=radius,
                region_type=region_type,
                signal=glob_signal,
                **vis_kwargs,
            )

        # Check if img is 2D RGB image
        if img.ndim == 3 and img.shape[-1] == 3:
            img = _to_grayscale(img)

        center_point = tuple(val // 2 for val in img.shape)

        # Check if starting parameters are provided
        if "signal_center" not in kwargs.keys():
            try:
                signal_start = self.parameters["signal_center"]
            except KeyError:
                signal_start = center_point
        else:
            signal_start = kwargs.pop("signal_center")
        if "radius" not in kwargs.keys():
            try:
                radius_start = self.parameters["radius"]
            except KeyError:
                radius_start = 1
        else:
            radius_start = kwargs.pop("radius")
        if "region_type" not in kwargs.keys():
            try:
                region_type_start = self.parameters["region_type"]
            except KeyError:
                region_type_start = "cubic"
        else:
            region_type_start = kwargs.pop("region_type")
        if "iterations" not in kwargs.keys():
            try:
                iterations_start = self.parameters["iterations"]
            except KeyError:
                iterations_start = 2
        else:
            iterations_start = kwargs.pop("iterations")
        if "lower_threshold" not in kwargs.keys():
            try:
                lower_threshold_start = self.parameters["lower_threshold"]
            except KeyError:
                lower_threshold_start = 90
        else:
            lower_threshold_start = kwargs.pop("lower_threshold")
        if "upper_threshold" not in kwargs.keys():
            try:
                upper_threshold_start = self.parameters["upper_threshold"]
            except KeyError:
                upper_threshold_start = 99
        else:
            upper_threshold_start = kwargs.pop("upper_threshold")

        _write_values_to_global(signal_start, radius_start, region_type_start)

        global glob_signal_center, glob_radius, glob_region_type
        # Check if background_center and signal_center are the right shape for 2d and 3d
        if len(glob_signal_center) != img.ndim:
            raise ValueError("Signal center has to be in the same dimension as img.")

        # Check if radius is an integer and positive
        if not isinstance(glob_radius, int) or glob_radius <= 0:
            raise TypeError("Radius has to be a positive integer.")

        # Check if region_type is valid
        if glob_region_type not in {
            "full",
            "original",
            "cubic",
            "spherical",
        }:
            raise ValueError("Region type not valid.")

        # Check if signal_center is a tuple of integers and not too close to the border
        _check_border_too_close(glob_signal_center, glob_radius)

        # Write values to attributes
        _save_values(None)

        # Create dropdown for region type
        dropdown_region_type = widgets.Dropdown(
            options=["cubic", "spherical", "full", "original"],
            value=region_type_start,
            description="Region Type:",
        )
        dropdown_region_type.style = {"description_width": "initial"}

        # Create slider for radius
        slider_radius = _create_slider_widget(
            max=min(img.shape) // 2,
            min=1,
            value=radius_start,
            description="Radius",
        )
        slider_radius.style = {"handle_color": "#f7f7f7"}

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
            # Add widget for z coordinate
            slider_signal_center_z = _create_slider_widget(
                min=slider_radius.value + 1,
                max=img.shape[2] - (slider_radius.value + 1),
                value=signal_start[2],
            )
            slider_signal_center_z.style = {"handle_color": "#2c7bb6"}
        else:
            slider_signal_center_z = None

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

        lower_threshold_slider = _create_slider_widget(
            min=0,
            max=100,
            value=lower_threshold_start,
            description="Lower Threshold",
        )
        lower_threshold_slider.style = {
            "handle_color": "#f7f7f7",
            "description_width": "initial",
        }

        upper_threshold_slider = _create_slider_widget(
            min=0,
            max=100,
            value=upper_threshold_start,
            description="Upper Threshold",
        )
        upper_threshold_slider.style = {
            "handle_color": "#f7f7f7",
            "description_width": "initial",
        }

        iterations_slider = _create_slider_widget(
            min=1,
            max=10,
            value=iterations_start,
            description="Iterations",
        )
        iterations_slider.style = {
            "handle_color": "#f7f7f7",
            "description_width": "initial",
        }

        # Create button to calculate region
        calc_button = widgets.Button(description="Calculate Full Region")
        calc_button.on_click(_calculate_region)

        # Create button to save values
        save_button = widgets.Button(
            description="Save Current Values", button_style="success"
        )
        save_button.on_click(_save_values)

        # Visualize centers
        if img.ndim == 3 and (img.shape[-1] != 3):  # 3D image
            # Set style of widgets
            slider_signal_center_x.style = {"handle_color": "#d7191c"}
            slider_signal_center_y.style = {"handle_color": "#fdae61"}

            # Create output
            out = widgets.interactive_output(
                _update_visualization_3d,
                {
                    "signal_center_x": slider_signal_center_x,
                    "signal_center_y": slider_signal_center_y,
                    "signal_center_z": slider_signal_center_z,
                    "radius": slider_radius,
                    "region_type": dropdown_region_type,
                },
            )
            _output_layout(out, FIGSIZE_SNR_3D)

            # Create UI
            ui = widgets.VBox(
                [
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
                        layout=widgets.Layout(
                            justify_content="space-around", padding="20px 0px"
                        ),
                    ),
                    widgets.HBox(
                        [slider_radius, dropdown_region_type],
                        layout=widgets.Layout(
                            justify_content="space-around",
                            padding="20px 0px",
                        ),
                    ),
                    widgets.HBox(
                        [
                            lower_threshold_slider,
                            upper_threshold_slider,
                            iterations_slider,
                            calc_button,
                        ],
                        layout=widgets.Layout(
                            justify_content="space-around", padding="20px 0px"
                        ),
                    ),
                    widgets.HBox(
                        [save_button],
                        layout=widgets.Layout(
                            justify_content="center", padding="20px 0px"
                        ),
                    ),
                ],
                layout=widgets.Layout(padding="10px 60px"),
            )

            display(out, ui)

        elif img.ndim == 2 or img.ndim == 3 and (img.shape[-1] == 3):  # 2D image
            # Set style of widgets
            slider_signal_center_x.style.handle_color = "#0571b0"
            slider_signal_center_y.style.handle_color = "#92c5de"

            # Create output
            out = widgets.interactive_output(
                _update_visualization_2d,
                {
                    "signal_center_x": slider_signal_center_x,
                    "signal_center_y": slider_signal_center_y,
                    "radius": slider_radius,
                    "region_type": dropdown_region_type,
                },
            )
            _output_layout(out, FIGSIZE_SNR_2D)

            # Create UI
            ui = widgets.VBox(
                [
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
                        ],
                        layout=widgets.Layout(justify_content="space-around"),
                    ),
                    widgets.HBox(
                        [slider_radius, dropdown_region_type],
                        layout=widgets.Layout(
                            justify_content="space-around", padding="20px 0px"
                        ),
                    ),
                    widgets.HBox(
                        [
                            lower_threshold_slider,
                            upper_threshold_slider,
                            iterations_slider,
                            calc_button,
                        ],
                        layout=widgets.Layout(
                            justify_content="space-around", padding="20px 0px"
                        ),
                    ),
                    widgets.HBox(
                        [save_button],
                        layout=widgets.Layout(
                            justify_content="center", padding="20px 0px"
                        ),
                    ),
                ],
                layout=widgets.Layout(padding="10px 60px"),
            )

            display(out, ui)

        else:
            raise ValueError("No visualization possible for non 2d or non 3d images.")


def signal_to_noise_ratio(
    img,
    signal_center,
    radius,
    region_type="cubic",
    auto_center=False,
    iterations=5,
    lower_threshold=90,
    upper_threshold=99,
    yuv=True,
    **kwargs,
):
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
        Automatically find the center of the image. `signal_center` and `radius` are
        ignored if True.
    iterations : int, optional
        Number of iterations for morphological operations if `auto_center` is True.
        Default is 5.
    lower_threshold : int, optional
        Lower threshold for the binary image if `auto_center` is True. Default is 90.
    upper_threshold : int, optional
        Upper threshold for the binary image if `auto_center` is True. Default is 99.
    yuv : bool, default True

        .. important::
            Only applicable for color images.

        If True, the input images are expected to be RGB images and are converted to YUV
        color space. If False, the input images are kept as RGB images.
    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`viqa.utils.load_data`.

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
        If the center is not a tuple of integers.
        If center is too close to the border.
        If the radius is not an integer.
        If the image is not 2D or 3D.
        If the passed region type is not valid.

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
    img = load_data(img, **kwargs)

    # Auto detect center
    if auto_center is True:
        warn(
            "Signal center is automatically detected. Parameters signal_center and "
            "radius are ignored."
        )

        binary_foreground = _get_binary(
            img,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            show=False,
        )

        signal_center, radius, signal_region = find_largest_region(
            img=binary_foreground, region_type=region_type, iterations=iterations
        )

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
        elif region_type == "original":
            signal = img

    # Check if signal_center is a tuple of integers and radius is an integer
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

        # Define region
        if region_type in {"full", "original"}:
            if not auto_center:
                signal = img
        elif region_type in {"cubic", "cube", "square"}:
            signal = _to_cubic(img, signal_center, radius)
        elif region_type in {"spherical", "sphere", "circle"}:
            signal = _to_spherical(img, signal_center, radius)
        else:
            if not auto_center:
                raise ValueError("Region type not valid.")

        sdev = np.std(signal, axis=(0, 1))

        if yuv:
            snr_val = [
                np.float64(np.mean(signal[..., 0]) / sdev[i]) if sdev[i] != 0 else 0
                for i in range(3)
            ]
            return snr_val[0], snr_val[1], snr_val[2]
        else:
            snr_lum = np.float64(
                signal_to_noise_ratio(
                    _to_grayscale(img),
                    signal_center,
                    radius,
                    region_type,
                    auto_center=False,
                    yuv=False,
                )
            )
            snr_val = [
                np.float64(np.mean(signal[..., i]) / sdev[i]) if sdev[i] != 0 else 0
                for i in range(3)
            ]

        return snr_lum, snr_val[0], snr_val[1], snr_val[2]

    # Define region
    if region_type in {"full", "original"}:
        if not auto_center:
            signal = img
    elif region_type in {"cubic", "cube", "square"}:
        signal = _to_cubic(img, signal_center, radius)
    elif region_type in {"spherical", "sphere", "circle"}:
        signal = _to_spherical(img, signal_center, radius)
    else:
        if not auto_center:
            raise ValueError("Region type not valid.")

    # Calculate SNR
    if np.std(signal) == 0:
        snr_val = 0
    else:
        snr_val = np.mean(signal) / np.std(signal)

    parameters = {}
    match region_type:
        case "full" if auto_center:
            parameters.update(
                {
                    "iterations": iterations,
                    "lower_threshold": lower_threshold,
                    "upper_threshold": upper_threshold,
                }
            )
        case "full" if not auto_center:
            region_type = "original"
        case "original":
            pass
        case _:
            parameters.update(
                {
                    "signal_center": signal_center,
                    "radius": radius,
                }
            )
    parameters.update({"region_type": region_type, "signal": signal})

    return np.float64(snr_val), parameters
