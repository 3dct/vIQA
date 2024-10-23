User Guide
==========

General Usage Advice
--------------------

You should use the classes for calculating the metrics. The functions are only intended
for advanced users who want to use the metrics in a non-standard way. A batch mode is
provided for batch processing of images.

Workflow
--------
Images are first loaded from .raw files or .mhd files and their corresponding .raw file, normalized to the chosen data
range (if the parameter ``normalize=True`` is set) and then compared. The scores are then calculated and can be printed.
If using paths file names need to be given with the bit depth denoted as a suffix (e.g. ``_8bit.raw``, ``_16bit.raw``) and
the dimensions of the images need to be given in the file name (e.g. ``512x512x512``). The images are assumed to be
grayscale. Treatment of color images is planned for later versions.
The metrics are implemented to calculate the scores for an 8-bit data range (0-255) per default. For some metrics the
resulting score is different for different data ranges. When calculating several metrics for the same image, the same
data range should be used for all metrics. The data range can be changed by setting the parameter ``data_range`` for each
metric. This parameter primarily affects the loading behaviour of the class instances when not using the
:py:func:`~viqa.utils.load_data` function directly as described further below, but for some metrics setting the data range is
necessary to calculate the score (e.g. PSNR).

Examples
--------
Better:

.. code-block:: python

    import viqa
    from viqa import load_data
    from viqa.utils import normalize_data

    ## load images
    file_path_img_r = 'path/to/reference_image_8bit_512x512x512.raw'
    file_path_img_m = 'path/to/modified_image_8bit_512x512x512.raw'
    img_r = load_data(
      file_path_img_r,
      data_range=1,
      normalize=False,
    )  # data_range ignored due to normalize=False
    img_m = load_data(file_path_img_m)  # per default: batch=False, normalize=False
    # --> both images are loaded as 8-bit images

    # calculate and print RMSE score
    rmse = viqa.RMSE()
    score_rmse = rmse.score(img_r, img_m)  # RMSE does not need any parameters
    rmse.print_score(decimals=2)

    # normalize to 16-bit
    img_r = normalize_data(img_r, data_range_output=(0, 65535))
    img_m = load_data(img_m, data_range=65535, normalize=True)
    # --> both functions have the same effect

    # calculate and print PSNR score
    psnr = viqa.PSNR(data_range=65535)  # PSNR needs data_range to calculate the score
    score_psnr = psnr.score(img_r, img_m)
    psnr.print_score(decimals=2)

    # set optional parameters for MAD as dict
    calc_parameters = {
        'block_size': 16,
        'block_overlap': 0.75,
        'beta_1': 0.467,
        'beta_2': 0.130,
        'luminance_function': {'b': 0, 'k': 0.02874, 'gamma': 2.2},
        'orientations_num': 4,
        'scales_num': 5,
        'weights': [0.5, 0.75, 1, 5, 6]
    }

    # calculate and print MAD score
    mad = viqa.MAD(data_range=65535)  # MAD needs data_range to calculate the score
    score_mad = mad.score(img_r, img_m, dim=2, **calc_parameters)
    mad.print_score(decimals=2)

Possible, but worse (recommended only if you want to calculate a single metric):

.. code-block:: python

    import viqa

    file_path_img_r = 'path/to/reference_image_512x512x512_16bit.raw'
    file_path_img_m = 'path/to/modified_image_512x512x512_16bit.raw'

    load_parameters = {'data_range': 1, 'normalize': True}
    # data_range is set to 1 to normalize the images
    # to 0-1 and for calculation, if not set 255 would
    # be used as default for loading and calculating
    # the score

    psnr = viqa.PSNR(**load_parameters)  # load_parameters necessary due to direct loading by class
    # also PSNR needs data_range to calculate the score
    # if images would not be normalized, data_range should be
    # 65535 for 16-bit images for correct calculation
    score = psnr.score(file_path_img_r, file_path_img_m)
    # --> images are loaded as 16-bit images and normalized to 0-1 via the `load_data` function
    #     called by the score method
    psnr.print_score(decimals=2)

.. tip::

    It is recommended to load the images directly with the :py:func:`~viqa.utils.load_data` function first and then pass the image
    arrays to the metrics functions. You can also pass the image paths directly to the metrics functions. In this case,
    the images will be loaded with the given parameters. This workflow is only recommended if you want to calculate a
    single metric.

.. important::

    The current recommended usage files are the Jupyter Notebooks on the `Github page`_.
    Additional information can be found in the documentation of the individual metrics under :doc:`api_reference`.

.. _Github page: https://github.com/3dct/vIQA
