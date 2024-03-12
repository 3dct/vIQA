[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
# vIQA &mdash; volumetric Image Quality Assessment

[[_TOC_]]

vIQA provides an extensive assessment suite for image quality of 2D-images or 3D-volumes as a python package.
Image Quality Assessment (IQA) is a field of research that aims to quantify the quality of an image. This is usually 
done by comparing the image to a reference image (full-reference metrics), but can also be done by evaluating the image 
without a reference (no-reference metrics). The reference image is usually the original image, but can also be
another image that is considered to be of high quality. The comparison is done by calculating a metric that quantifies
the difference between the two images or for the image itself.
This package implements several metrics to compare two images or volumes using different IQA metrics. In addition, some
metrics are implemented that can be used to evaluate a single image or volume.

The metrics used are:
- Peak Signal to Noise Ratio (PSNR)
- Root Mean Square Error (RMSE)
- Structured Similarity (SSIM) [^1]
- Multi-Scale Structural Similarity (MS-SSIM) [^2]
  > [!NOTE]
  > can only be used for 2D images currently
- Feature Similarity Index (FSIM) [^3]
- Visual Information Fidelity in *pixel* domain (VIFp) [^4]
  > [!NOTE]
  > can only be used for 2D images currently
  
  > [!CAUTION]
  > The calculated values for VIFp are probably not correct in this implementation. Those values should be treated with 
  > caution as further testing is required.
- Visual Saliency Index (VSI) [^5]
  > [!NOTE]
  > can only be used for 2D images currently

  > [!WARNING]
  > The original metric supports RGB images only. This implementation can work with 
  > grayscale images by copying the luminance channel 3 times. 
- Most Apparent Distortion (MAD) [^6]
- Gradient Similarity Measure (GSM) [^7]
  > [!CAUTION]
  > This metric is not yet tested. The metric should be only used for experimental purposes.
- Contrast to Noise Ratio (CNR) [^8]
- Signal to Noise Ratio (SNR)

Overview
--------
| Metric  | Name                                          | Type | Dimensional behaviour | Colour Behaviour        | Range              | Tested             | Validated | Reference |
|---------|-----------------------------------------------|------|-----------------------|-------------------------|--------------------|--------------------|-----------|-----------|
| PSNR    | Peak Signal to Noise Ratio                    | FR   | 3D native             |                         | $[0, \infty)$      | :heavy_check_mark: | :x:       | &mdash;   |
| RMSE    | Root Mean Square Error                        | FR   | 3D native             |                         | $[0, 1]$           | :heavy_check_mark: | :x:       | &mdash;   |
| SSIM    | Structured Similarity                         | FR   | 3D native             |                         | $[0, 1]$           | :heavy_check_mark: | :x:       | [^1]      |
| MS-SSIM | Multi-Scale Structural Similarity             | FR   | 3D                    | :question:              | $[0, 1]$           | :x:                | :x:       | [^2]      |
| FSIM    | Feature Similarity Index                      | FR   | 3D                    | :heavy_check_mark:      | $[0, 1]$           | :x:                | :x:       | [^3]      |
| VIFp    | Visual Information Fidelity in *pixel* domain | FR   | 3D                    | :question:              | $[0, \infty)$ [^a] | :x:                | :x:       | [^4]      |
| VSI     | Visual Saliency Index                         | FR   | 3D                    | :heavy_check_mark: [^b] | $[0, 1]$           | :x:                | :x:       | [^5]      |
| MAD     | Most Apparent Distortion                      | FR   | 3D slicing            |                         | $[0, \infty)$      | :heavy_check_mark: | :x:       | [^6]      |
| GSM     | Gradient Similarity                           | FR   | 3D native or slicing  |                         | $[0, 1]$           | :x:                | :x:       | [^7]      |
| CNR     | Contrast to Noise Ratio                       | FR   | 3D native             |                         | $[0, \infty)$      | :heavy_check_mark: | :x:       | [^8]      |
| SNR     | Signal to Noise Ratio                         | FR   | 3D native             |                         | $[0, \infty)$      | :heavy_check_mark: | :x:       | &mdash;   |

[^a]: Normally $[0, 1]$, but can be higher than 1 for modified images with higher 
contrast than reference images.
[^b]: The original metric supports RGB images only. This implementation can work 
with grayscale images by copying the luminance channel 3 times.

<!-- ## Installation TODO: add installation instructions -->

## Requirements
The following packages have to be installed:
- numpy
- scipy
- pytorch
- piq
- scikit-image
- matplotlib
- jupyter

<!-- ## Documentation TODO: add link to documentation -->

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
from viqa import load_data
from viqa.utils import normalize_data

## load images
file_path_img_r = 'path/to/reference_image_8bit_512x512x512.raw'
file_path_img_m = 'path/to/modified_image_8bit_512x512x512.raw'
img_r = load_data(
  file_path_img_r, 
  data_range=1, 
  normalize=False,
  batch=False,
)  # data_range ignored due to normalize=False
img_m = load_data(file_path_img_m)  # per default: batch=False, normalize=False
# --> both images are loaded as 8-bit images

# calculate and print RMSE score
rmse = viqa.RMSE()
score_rmse = rmse.score(img_r, img_m)  # RMSE does not need any parameters
rmse.print_score(decimals=2)

# normalize to 16-bit
img_r = normalize_data(img_r, data_range=65535)
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
> The current recommended usage file is: [`Image_Comparison.ipynb`](Image_Comparison.ipynb).

<!-- ## Metric List TODO: add list of metrics -->

<!-- ## Benchmark TODO: add benchmark results and instructions -->

## TODO
- [ ] Add metrics
    - [x] Add RMSE
    - [x] Add PSNR
    - [x] Add SSIM
    - [x] Add MSSSIM
    - [x] Add FSIM
    - [x] Add VSI
    - [x] Add VIF
    - [x] Add MAD
    - [x] Add GSM
    - [ ] Add SFF/IFS
    - [x] Add CNR
    - [x] Add SNR
    - [ ] Add Ma
    - [ ] Add PI
    - [ ] Add NIQE
    - [ ] Add Q-Factor
- [ ] Add tests
    - [x] Add tests for RMSE
    - [x] Add tests for PSNR
    - [x] Add tests for SSIM
    - [ ] Add tests for MSSSIM
    - [ ] Add tests for FSIM
    - [ ] Add tests for VSI
    - [ ] Add tests for VIF
    - [x] Add tests for MAD
    - [ ] Add tests for GSM
    - [ ] Add tests for SFF/IFS
    - [x] Add tests for CNR
    - [x] Add tests for SNR
    - [ ] Add tests for Ma
    - [ ] Add tests for PI
    - [ ] Add tests for NIQE
    - [ ] Add tests for Q-Factor
- [ ] Add documentation
    - [x] Add documentation for rmse.py
    - [x] Add documentation for psnr.py
    - [x] Add documentation for ssim.py
    - [x] Add documentation for msssim.py
    - [x] Add documentation for fsim.py
    - [x] Add documentation for vsi.py
    - [x] Add documentation for vif.py
    - [x] Add documentation for mad.py
    - [x] Add documentation for gsm.py
    - [ ] Add documentation for sff.py
    - [x] Add documentation for cnr.py
    - [x] Add documentation for snr.py
    - [ ] Add documentation for ma.py
    - [ ] Add documentation for pi.py
    - [ ] Add documentation for niqe.py
    - [ ] Add documentation for qfactor.py
    - [x] Add documentation for utils.py
- [ ] Adapt to 3D
    - [x] SSIM
    - [x] MSSSIM
    - [x] FSIM
    - [x] VSI
    - [x] VIF
- [ ] Add support for different data ranges
- [ ] Validate metrics
- [ ] Add color image support
- [ ] Add batch support 

<!-- ## Citation TODO: add citation instructions -->

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
[^1]:  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality 
assessment: From error visibility to structural similarity. IEEE Transactions on 
Image Processing, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
[^2]: Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multi-scale structural 
similarity for image quality assessment. The Thirty-Seventh Asilomar Conference on 
Signals, Systems & Computers, 1298–1402. https://doi.org/10.1109/ACSSC.2003.1292216
[^3]: Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011). FSIM: A feature similarity 
index for image quality assessment. IEEE Transactions on Image Processing, 20(8). 
https://doi.org/10.1109/TIP.2011.2109730
[^4]: Sheikh, H. R., & Bovik, A. C. (2006). Image information and visual quality. IEEE 
Transactions on Image Processing, 15(2), 430–444. 
https://doi.org/10.1109/TIP.2005.859378
[^5]: Zhang, L., Shen, Y., & Li, H. (2014). VSI: A visual saliency-induced index for 
perceptual image quality assessment. IEEE Transactions on Image Processing, 23(10), 
4270–4281. https://doi.org/10.1109/TIP.2014.2346028
[^6]: Larson, E. C., & Chandler, D. M. (2010). Most apparent distortion: full-reference 
image quality assessment and the role of strategy. Journal of Electronic Imaging, 19
(1), 011006. https://doi.org/10.1117/1.3267105
[^7]: Liu, A., Lin, W., & Narwaria, M. (2012). Image quality assessment based on 
gradient similarity. IEEE Transactions on Image Processing, 21(4), 1500–1512. 
https://doi.org/10.1109/TIP.2011.2175935
[^8]: Desai, N., Singh, A., & Valentino, D. J. (2010). Practical evaluation of image 
quality in computed radiographic (CR) imaging systems. Medical Imaging 2010: Physics 
of Medical Imaging, 7622, 76224Q. https://doi.org/10.1117/12.844640