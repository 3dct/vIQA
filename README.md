# vIQA &mdash; volumetric Image Quality Assessment
...

## Usage
...
It is recommended to load the images directly with the `vIQA.utils.load_data` function first and then pass the image 
arrays to the metrics functions. You can also pass the image paths directly to the metrics functions. In this case, the 
images will be loaded with the given parameters. This workflow is only recommended if you want to calculate a single 
metric.

### TODO:
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
    - [ ] Add CNR
    - [ ] Add Ma
    - [ ] Add PI
    - [ ] Add NIQE
    - [ ] Add Q-Factor
- [ ] Add tests
- [ ] Add documentation
    - [ ] Add documentation for fsim.py
    - [ ] Add documentation for ssim.py
    - [ ] Add documentation for msssim.py
    - [ ] Add documentation for gsm.py
    - [ ] Add documentation for vif.py
    - [ ] Add documentation for vsi.py
    - [ ] Add documentation for metrics.py
    - [ ] Add documentation for utils.py
    - [ ] Add documentation for kernels.py
- [ ] Adapt to 3D
    - [ ] SSIM
    - [ ] MSSSIM
    - [ ] FSIM
    - [ ] VSI
    - [ ] VIF