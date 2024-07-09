################################
Welcome to vIQA's documentation!
################################

:Author: Lukas Behammer
:Release: |release|
:Date: |today|

vIQA (volumetric Image Quality Assessment) provides an extensive assessment suite for image quality of 2D-images or 3D-volumes as a python package.
Image Quality Assessment (IQA) is a field of research that aims to quantify the quality of an image. This is usually
done by comparing the image to a reference image (full-reference metrics), but can also be done by evaluating the image
without a reference (no-reference metrics). The reference image is usually the original image, but can also be
another image that is considered to be of high quality. The comparison is done by calculating a metric that quantifies
the difference between the two images or for the image itself.
This package implements several metrics to compare two images or volumes using different IQA metrics. In addition, some
metrics are implemented that can be used to evaluate a single image or volume.

The following metrics are implemented:

.. table:: Implemented metrics
    :widths: auto

    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | Metric    | Name                                          | Type | Dimensional behaviour | Colour Behaviour          | Range (different/worst - identical/best) | Tested                  | Validated                 | Reference |
    +===========+===============================================+======+=======================+===========================+==========================================+=========================+===========================+===========+
    | PSNR      | Peak Signal to Noise Ratio                    | FR   | 3D native             | :math:`\checkmark`        | :math:`[0, \infty)`                      | :math:`\checkmark`      | :math:`\checkmark`        | ---       |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | RMSE      | Root Mean Square Error                        | FR   | 3D native             | :math:`\checkmark`        | :math:`(\infty, 0]`                      | :math:`\checkmark`      | :math:`\checkmark`        | ---       |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | UQI [*]_  | Universal Quality Index                       | FR   | 3D native             | (:math:`\checkmark`) [*]_ | :math:`[-1, 1]`                          | :math:`\times`          | (:math:`\checkmark`) [*]_ | [1]_      |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | SSIM      | Structured Similarity                         | FR   | 3D native             | (:math:`\checkmark`) [*]_ | :math:`[-1, 1]` [*]_                     | :math:`\checkmark`      | :math:`\checkmark`        | [2]_      |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | MS-SSIM   | Multi-Scale Structural Similarity             | FR   | 3D slicing            | ?                         | :math:`[0, 1]`                           | :math:`\times`          | :math:`\checkmark`        | [3]_      |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | FSIM      | Feature Similarity                            | FR   | 3D slicing            | :math:`\checkmark`        | :math:`[0, 1]`                           | :math:`\checkmark`      | :math:`\checkmark`        | [4]_      |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | VIFp      | Visual Information Fidelity in *pixel* domain | FR   | 3D slicing            | ?                         | :math:`[0, \infty)` [*]_                 | :math:`\times`          | :math:`\times` [*]_       | [5]_      |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | VSI       | Visual Saliency-based Index                   | FR   | 3D slicing            | :math:`\checkmark` [*]_   | :math:`[0, 1]`                           | :math:`\times`          | :math:`\times`            | [6]_      |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | MAD       | Most Apparent Distortion                      | FR   | 3D slicing            |                           | :math:`[0, \infty)`                      | :math:`\checkmark`      | :math:`\times`            | [7]_      |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | GSM       | Gradient Similarity                           | FR   | 3D native or slicing  |                           | :math:`[0, 1]`                           | :math:`\times`          | :math:`\times`            | [8]_      |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | CNR       | Contrast to Noise Ratio                       | NR   | 3D native             |                           | :math:`[0, \infty)`                      | :math:`\checkmark`      | :math:`\times`            | [9]_      |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | SNR       | Signal to Noise Ratio                         | NR   | 3D native             | :math:`\checkmark`        | :math:`[0, \infty)`                      | :math:`\checkmark`      | :math:`\times`            | ---       |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+
    | Q-Measure | Q-Measure                                     | NR   | 3D only [*]_          | :math:`\times`            | :math:`[0, \infty)`                      | :math:`\times`          | :math:`\times`            | [10]_     |
    +-----------+-----------------------------------------------+------+-----------------------+---------------------------+------------------------------------------+-------------------------+---------------------------+-----------+


.. [*] UQI is a special case of SSIM. Also see [2]_.
.. [*] The metric is calculated channel-wise for color images. The values are then averaged after weighting.
.. [*] As UQI is a special case of SSIM, the validation of SSIM is also valid for UQI.
.. [*] The metric is calculated channel-wise for color images. The values are then averaged after weighting.
.. [*] The range for SSIM is given as :math:`[-1, 1]`, but is usually :math:`[0, 1]` in practice.
.. [*] Normally :math:`[0, 1]`, but can be higher than 1 for modified images with higher
    contrast than reference images.
.. [*] The calculated values for VIFp are probably not correct in this implementation.
    Those values should be treated with caution as further testing is required.
.. [*] The original metric supports RGB images only. This implementation can work
    with grayscale images by copying the luminance channel 3 times.
.. [*] The Q-Measure is a special metric designed for CT images. Therefore it only works
    with 3D volumes.


*******************
General Information
*******************

Requirements
============

The package is written in Python 3.11 and requires the following packages:

* matplotlib
* nibabel
* numpy
* piq
* pytorch
* scikit-image
* scipy
* (jupyter) if you want to use the notebook on Github

Installation
============

The package can be installed via ``pip``:

.. code-block:: bash

    pip install viqa

or ``conda``:

.. code-block:: bash

    conda install -c conda-forge viqa

.. important::

    The package is currently in development and not yet available on conda-forge.


General Usage Advice
====================

You should use the Classes for calculating the metrics. The functions are only intended
for advanced users who want to use the metrics in a non-standard way. A bash mode is
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
:doc:`generated/viqa.load_utils.load_data` function directly as described further below, but for some metrics setting the data range is
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

    It is recommended to load the images directly with the :doc:`generated/viqa.load_utils.load_data` function first and then pass the image
    arrays to the metrics functions. You can also pass the image paths directly to the metrics functions. In this case,
    the images will be loaded with the given parameters. This workflow is only recommended if you want to calculate a
    single metric.

.. important::

    The current recommended usage file is the Jupyter Notebook on the Github page.


Contacts
========

If you have any questions, please contact the author at: `<lukas.behammer@fh-wels.at>`_.


References
==========

.. [1] Wang, Z., & Bovik, A. C. (2002). A Universal Image Quality Index. IEEE SIGNAL
    PROCESSING LETTERS, 9(3). https://doi.org/10.1109/97.995823
.. [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality
    assessment: From error visibility to structural similarity. IEEE Transactions on
    Image Processing, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
.. [3] Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multi-scale structural
    similarity for image quality assessment. The Thirty-Seventh Asilomar Conference on
    Signals, Systems & Computers, 1298–1402. https://doi.org/10.1109/ACSSC.2003.1292216
.. [4] Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011). FSIM: A feature similarity
    index for image quality assessment. IEEE Transactions on Image Processing, 20(8).
    https://doi.org/10.1109/TIP.2011.2109730
.. [5] Sheikh, H. R., & Bovik, A. C. (2006). Image information and visual quality. IEEE
    Transactions on Image Processing, 15(2), 430–444.
    https://doi.org/10.1109/TIP.2005.859378
.. [6] Zhang, L., Shen, Y., & Li, H. (2014). VSI: A visual saliency-induced index for
    perceptual image quality assessment. IEEE Transactions on Image Processing, 23(10),
    4270–4281. https://doi.org/10.1109/TIP.2014.2346028
.. [7] Larson, E. C., & Chandler, D. M. (2010). Most apparent distortion: full-reference
    image quality assessment and the role of strategy. Journal of Electronic Imaging, 19
    (1), 011006. https://doi.org/10.1117/1.3267105
.. [8] Liu, A., Lin, W., & Narwaria, M. (2012). Image quality assessment based on
    gradient similarity. IEEE Transactions on Image Processing, 21(4), 1500–1512.
    https://doi.org/10.1109/TIP.2011.2175935
.. [9] Desai, N., Singh, A., & Valentino, D. J. (2010). Practical evaluation of image
    quality in computed radiographic (CR) imaging systems. Medical Imaging 2010: Physics
    of Medical Imaging, 7622, 76224Q. https://doi.org/10.1117/12.844640
.. [10] Reiter, M., Weiß, D., Gusenbauer, C., Erler, M., Kuhn, C., Kasperl, S., &
    Kastner, J. (2014). Evaluation of a Histogram-based Image Quality Measure for X-ray
    computed Tomography. 5th Conference on Industrial Computed Tomography (iCT) 2014,
    25-28 February 2014, Wels, Austria. e-Journal of Nondestructive Testing Vol. 19(6).
    https://www.ndt.net/?id=15715

*************
API Reference
*************

This is the API reference for vIQA, a package for volumetric Image Quality Assessment.

.. toctree::
   :maxdepth: 2

   fr_metrics

.. toctree::
   :maxdepth: 2

   nr_metrics

.. toctree::
   :maxdepth: 2

   utils

.. toctree::
   :maxdepth: 2

   fusion

.. toctree::
   :maxdepth: 2

   batch_mode


******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
