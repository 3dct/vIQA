<h1 style="text-align: center;">
<img src="https://raw.githubusercontent.com/3dct/vIQA/main/branding/logo/Logo_vIQA_wo-text.svg" width="300" alt="Logo for vIQA: A cube with three slices colored in red, green and blue in one direction and three slices colored in black, gray and white in another direction.">

vIQA &mdash; volumetric Image Quality Assessment
</h1><br>

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![PyPI - Version](https://img.shields.io/pypi/v/vIQA)](https://pypi.org/project/vIQA/latest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vIQA)](https://pypi.org/project/vIQA/)
[![PyPI - License](https://img.shields.io/pypi/l/vIQA)](https://pypi.org/project/vIQA/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/vIQA)](https://pypi.org/project/vIQA/)
[![Release](https://github.com/3dct/vIQA/actions/workflows/release.yaml/badge.svg)](https://github.com/3dct/vIQA/actions/workflows/release.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/3dct/vIQA/main.svg)](https://results.pre-commit.ci/latest/github/3dct/vIQA/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/3dct/vIQA/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)

## Table of Contents

* [Overview](#overview)
* [Documentation](#documentation)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
  * [Workflow](#workflow)
  * [Examples](#examples)
* [TODO](#todo)
* [License](#license)
* [Contacts](#contacts)
* [References](#references)

vIQA provides an extensive assessment suite for image quality of 2D-images or 3D-volumes as a python package.
Image Quality Assessment (IQA) is a field of research that aims to quantify the quality of an image. This is usually
done by comparing the image to a reference image (full-reference metrics), but can also be done by evaluating the image
without a reference (no-reference metrics). The reference image is usually the original image, but can also be
another image that is considered to be of high quality. The comparison is done by calculating a metric that quantifies
the difference between the two images or for the image itself. These quality metrics are used in various fields, such as
medical imaging, computer vision, and image processing. For example the efficiency of image compression algorithms can be
evaluated by comparing the compressed image to the original image.
This package implements several metrics to compare two images or volumes using different IQA metrics. In addition, some
metrics are implemented that can be used to evaluate a single image or volume.

The metrics used are:
- Peak Signal to Noise Ratio (PSNR)
- Root Mean Square Error (RMSE)
- Universal Quality Index (UQI) [^1]
- Structured Similarity (SSIM) [^2]
- Multi-Scale Structural Similarity (MS-SSIM) [^3]
- Feature Similarity Index (FSIM) [^4]
- Visual Information Fidelity in *pixel* domain (VIFp) [^5]

> [!CAUTION]
> The calculated values for VIFp are probably not correct in this implementation. Those values should be treated with
> caution as further testing is required.

- Visual Saliency Index (VSI) [^6]

> [!WARNING]
> The original metric supports RGB images only. This implementation can work with
> grayscale images by copying the luminance channel 3 times.

- Most Apparent Distortion (MAD) [^7]
- Gradient Similarity Measure (GSM) [^8]

> [!CAUTION]
> This metric is not yet tested. The metric should be only used for experimental purposes.

- Contrast to Noise Ratio (CNR) [^9]
- Signal to Noise Ratio (SNR)
- Q-Measure [^10]

## Overview

| Metric    | Name                                          | Type | Dimensional behaviour | Colour Behaviour | Range (different/worst - identical/best) | Tested | Validated | Reference |
|-----------|-----------------------------------------------|------|-----------------------|------------------|------------------------------------------|--------|-----------|-----------|
| PSNR      | Peak Signal to Noise Ratio                    | FR   | 3D native             | ✔️               | $[0, \infty)$                            | ✔️     | ✔️        | &mdash;   |
| RMSE      | Root Mean Square Error                        | FR   | 3D native             | ✔️               | $(\infty, 0]$                            | ✔️     | ✔️        | &mdash;   |
| UQI [^a]  | Universal Quality Index                       | FR   | 3D native             | (✔️) [^b]        | $[-1, 1]$                                | ❌      | (✔️) [^c] | [^1]      |
| SSIM      | Structured Similarity                         | FR   | 3D native             | (✔️) [^b]        | $[-1, 1]$ [^d]                           | ✔️     | ✔️        | [^2]      |
| MS-SSIM   | Multi-Scale Structural Similarity             | FR   | 3D slicing            | ❓                | $[0, 1]$                                 | ❌      | ✔️        | [^3]      |
| FSIM      | Feature Similarity Index                      | FR   | 3D slicing            | ✔️               | $[0, 1]$                                 | ✔️     | ✔️        | [^4]      |
| VIFp      | Visual Information Fidelity in *pixel* domain | FR   | 3D slicing            | ❓                | $[0, \infty)$ [^e]                       | ❌      | ❌         | [^5]      |
| VSI       | Visual Saliency Index                         | FR   | 3D slicing            | ✔️ [^f]          | $[0, 1]$                                 | ❌      | ❌         | [^6]      |
| MAD       | Most Apparent Distortion                      | FR   | 3D slicing            |                  | $[0, \infty)$                            | ✔️     | ❌         | [^7]      |
| GSM       | Gradient Similarity                           | FR   | 3D native or slicing  |                  | $[0, 1]$                                 | ❌      | ❌         | [^8]      |
| CNR       | Contrast to Noise Ratio                       | NR   | 3D native             |                  | $[0, \infty)$                            | ✔️     | ❌         | [^9]      |
| SNR       | Signal to Noise Ratio                         | NR   | 3D native             | ✔️               | $[0, \infty)$                            | ✔️     | ❌         | &mdash;   |
| Q-Measure | Q-Measure                                     | NR   | 3D only [^g]          | ❌                | $[0, \infty)$                            | ❌      | ❌         | [^10]     |

[^a]: UQI is a special case of SSIM. Also see [^2].
[^b]: The metric is calculated channel-wise for color images. The values are then averaged after weighting.
[^c]: As UQI is a special case of SSIM, the validation of SSIM is also valid for UQI.
[^d]: The range for SSIM is given as $[-1, 1]$, but is usually $[0, 1]$ in practice.
[^e]: Normally $[0, 1]$, but can be higher than 1 for modified images with higher
contrast than reference images.
[^f]: The original metric supports RGB images only. This implementation can work
with grayscale images by copying the luminance channel 3 times.
[^g]: The Q-Measure is a special metric designed for CT images. Therefore it only works
with 3D volumes.

## Documentation
The API documentation can be found [here](https://3dct.github.io/vIQA/).

## Requirements
The following packages have to be installed:
- matplotlib
- nibabel
- numpy
- piq
- pytorch
- scikit-image
- scipy
- tqdm
- (jupyter) if you want to use the provided notebook

## Installation
Use either `pip`
```
pip install viqa
```

or `conda`
```
conda install -c conda-forge viqa
```

> [!IMPORTANT]
> The package is currently in development and not yet available on conda-forge.


## Usage

### Workflow
Images are first loaded from .raw files or .mhd files and their corresponding .raw file, normalized to the chosen data
range (if the parameter `normalize=True` is set) and then compared. The scores are then calculated and can be printed.
If using paths file names need to be given with the bit depth denoted as a suffix (e.g. `_8bit.raw`, `_16bit.mhd`) and
the dimensions of the images need to be given in the file name (e.g. `512x512x512`). The images are assumed to be
grayscale. Treatment of color images is planned for later versions.
The metrics are implemented to calculate the scores for an 8-bit data range (0-255) per default. For some metrics the
resulting score is different for different data ranges. When calculating several metrics for the same image, the same
data range should be used for all metrics. The data range can be changed by setting the parameter `data_range` for each
metric. This parameter primarily affects the loading behaviour of the class instances when not using the
`vIQA.utils.load_data` function directly as described further below, but for some metrics setting the data range is
necessary to calculate the score (e.g. PSNR).

### Examples
Better:

```python
import viqa
from viqa import load_data, normalize_data

## load images
file_path_img_r = 'path/to/reference_image_8bit_512x512x512.raw'
file_path_img_m = 'path/to/modified_image_8bit_512x512x512.raw'
img_r = load_data(
  file_path_img_r,
  data_range=1,
  normalize=False,
)  # data_range ignored due to normalize=False
img_m = load_data(file_path_img_m)  # per default: normalize=False
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
```
Possible, but worse (recommended only if you want to calculate a single metric):

```python
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
```

> [!TIP]
> It is recommended to load the images directly with the `vIQA.utils.load_data` function first and then pass the image
> arrays to the metrics functions. You can also pass the image paths directly to the metrics functions. In this case,
> the images will be loaded with the given parameters. This workflow is only recommended if you want to calculate a
> single metric.

> [!IMPORTANT]
> The current recommended usage files are: [`Image_Comparison.ipynb`](notebooks/Image_Comparison.ipynb) and [`Image_comparison_batch.ipynb`](notebooks/Image_Comparison_batch.ipynb).

For more examples, see the provided Jupyter notebooks and the documentation under [API Reference](https://3dct.github.io/vIQA/api_reference.html).

<!-- ## Benchmark TODO: add benchmark results and instructions -->

## TODO
- [ ] Add metrics
    - [ ] Add SFF/IFS
    - [ ] Add Ma
    - [ ] Add PI
    - [ ] Add NIQE
- [ ] Add tests
    - [x] Add tests for RMSE
    - [x] Add tests for PSNR
    - [x] Add tests for SSIM
    - [ ] Add tests for MSSSIM
    - [x] Add tests for FSIM
    - [ ] Add tests for VSI
    - [ ] Add tests for VIF
    - [x] Add tests for MAD
    - [ ] Add tests for GSM
    - [x] Add tests for CNR
    - [x] Add tests for SNR
    - [ ] Add tests for Q-Measure
- [ ] Add support for different data ranges
- [ ] Validate metrics
- [ ] Add color image support
- [x] Add support for printing values
  - [ ] Add support for .txt files
  - [x] Add support for .csv files
- [x] Add support for fusions
  - [x] Add support for linear combination
  - [ ] Add support for decision fusion

<!-- ## Citation TODO: add citation instructions -->

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to the
project and [development guide](https://3dct.github.io/vIQA/developer_guide.html) for
further information.

## License
**BSD 3-Clause**

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Contacts
Lukas Behammer, [lukas.behammer@fh-wels.at](mailto:lukas.behammer@fh-wels.at)

## References
[^1]: Wang, Z., & Bovik, A. C. (2002). A Universal Image Quality Index. IEEE SIGNAL
        PROCESSING LETTERS, 9(3). https://doi.org/10.1109/97.995823
[^2]: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality
assessment: From error visibility to structural similarity. IEEE Transactions on
Image Processing, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
[^3]: Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multi-scale structural
similarity for image quality assessment. The Thirty-Seventh Asilomar Conference on
Signals, Systems & Computers, 1298–1402. https://doi.org/10.1109/ACSSC.2003.1292216
[^4]: Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011). FSIM: A feature similarity
index for image quality assessment. IEEE Transactions on Image Processing, 20(8).
https://doi.org/10.1109/TIP.2011.2109730
[^5]: Sheikh, H. R., & Bovik, A. C. (2006). Image information and visual quality. IEEE
Transactions on Image Processing, 15(2), 430–444.
https://doi.org/10.1109/TIP.2005.859378
[^6]: Zhang, L., Shen, Y., & Li, H. (2014). VSI: A visual saliency-induced index for
perceptual image quality assessment. IEEE Transactions on Image Processing, 23(10),
4270–4281. https://doi.org/10.1109/TIP.2014.2346028
[^7]: Larson, E. C., & Chandler, D. M. (2010). Most apparent distortion: full-reference
image quality assessment and the role of strategy. Journal of Electronic Imaging, 19
(1), 011006. https://doi.org/10.1117/1.3267105
[^8]: Liu, A., Lin, W., & Narwaria, M. (2012). Image quality assessment based on
gradient similarity. IEEE Transactions on Image Processing, 21(4), 1500–1512.
https://doi.org/10.1109/TIP.2011.2175935
[^9]: Desai, N., Singh, A., & Valentino, D. J. (2010). Practical evaluation of image
quality in computed radiographic (CR) imaging systems. Medical Imaging 2010: Physics
of Medical Imaging, 7622, 76224Q. https://doi.org/10.1117/12.844640
[^10]: Reiter, M., Weiß, D., Gusenbauer, C., Erler, M., Kuhn, C., Kasperl, S., &
Kastner, J. (2014). Evaluation of a Histogram-based Image Quality Measure for X-ray
computed Tomography. 5th Conference on Industrial Computed Tomography (iCT) 2014, 25-28
February 2014, Wels, Austria. e-Journal of Nondestructive Testing Vol. 19(6).
https://www.ndt.net/?id=15715
