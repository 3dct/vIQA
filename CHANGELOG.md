# CHANGELOG



## v1.12.0 (2024-09-04)

### Feature

* feat: add function to create image report

move print_image method to utils.py as function and rename to export_image, add parameter to change filename, fix bug when number of fr-metrics is not a multiple of four ([`cc8b4ff`](https://github.com/3dct/vIQA/commit/cc8b4ff654f8721a552c13b9c9e26d8a6697fecd))


## v1.11.1 (2024-09-04)

### Documentation

* docs: add pre-commit usage in developer guide ([`4c313ab`](https://github.com/3dct/vIQA/commit/4c313ab31a66b5dd9369ad2ec8d226b9c46279b9))

* docs: Update docs

update developer guide regarding docs generation, rename pages for utility modules and usage ([`4f35dd3`](https://github.com/3dct/vIQA/commit/4f35dd3d1a51f54035640ce7cb6c2b9115c85e48))

### Performance

* perf: improve performance of load_data

call ImageArray only at the end of load_data function and only if necessary ([`b3fdecd`](https://github.com/3dct/vIQA/commit/b3fdecdad190ab8a8831d4885bbe5e0f5e96c420))

* perf: improve performance of BatchMetrics

add checks to load images only if not already in memory, add function to calculate no-reference metrics only if not already calculated ([`a0f3a26`](https://github.com/3dct/vIQA/commit/a0f3a267cf132c35e4416b9f5ae5be097e7dc162))


## v1.11.0 (2024-09-02)

### Feature

* feat: add function to crop images

cal cropping function during loading if parameter roi is given ([`140d101`](https://github.com/3dct/vIQA/commit/140d101ace4ffa3dbd97ecc11db8ada3dba0ae6b))


## v1.10.0 (2024-08-27)

### Documentation

* docs: update docs

remove sphinx-notfound-page, temporarily disable hoverxref ([`614d52a`](https://github.com/3dct/vIQA/commit/614d52aa0a05e87c664ed99fb8f7e76b43b9c589))

* docs: add sphinx-autobuild to developer_guide.rst ([`59381a4`](https://github.com/3dct/vIQA/commit/59381a4910dfbf2b4386c3766241dd54a4d42349))

### Feature

* feat: add function to compute multiple metrics at once

rename batch_mode.py to multiple.py, add MultipleMetrics class, move export_metadata to it&#39;s own function in utils.py, add base class for classes of type multiple, update class BatchMetrics ([`6f4f8a7`](https://github.com/3dct/vIQA/commit/6f4f8a7765078a322e12cfca6720f3d8e0842082))


## v1.9.1 (2024-08-23)

### Documentation

* docs: update docs

multiple changes to docstrings and docs rendering and docs building ([`080045f`](https://github.com/3dct/vIQA/commit/080045f19703513633d6a18a7da26e44b2b8f7eb))

### Fix

* fix: use ImageArray instead of np.ndarray

update according to 40f0d029ad416747135f79e8716de040335da592, fixes mypy ([`4ecfc86`](https://github.com/3dct/vIQA/commit/4ecfc8627321d7eee39e3ab0656bb155cfc56307))


## v1.9.0 (2024-08-22)

### Build

* build: set specific meson and meson-python versions ([`e395ba2`](https://github.com/3dct/vIQA/commit/e395ba2591718aef853d6de8136c7422b7a0262e))

* build: remove support for Python 3.13 until in stable

Python 3.13 build takes too long or fails, add specific dependency version for meson-python ([`d6dd0e6`](https://github.com/3dct/vIQA/commit/d6dd0e67c9c54458ca3bae15158f611ebab91236))

### Feature

* feat(batch_mode.py): add parameter data_range to export_metadata ([`00cb4c4`](https://github.com/3dct/vIQA/commit/00cb4c427ae4cf17db83a006397dca7aaf3ef0cc))

* feat: add parameter to set order of scaling in batch_mode.py ([`0c3886c`](https://github.com/3dct/vIQA/commit/0c3886c8d00e25b9400ce87eb801a4d3be3b0ac9))

### Fix

* fix: add Exception and warning to handle non 3D volumes in qmeasure.py ([`33e288a`](https://github.com/3dct/vIQA/commit/33e288ae032a1ed127c23c65412d084ae6054f8b))


## v1.8.0 (2024-08-22)

### Build

* build: update numpy dependency for build to be lower than 2.0.0 ([`54ac725`](https://github.com/3dct/vIQA/commit/54ac725020e2394baa9188774fce5f45ce49cfb7))

### Documentation

* docs: update Tuple order for visualization methods in snr.py and cnr.py

change Tuple order according to 4d7c3e02 and 0bab2b85 ([`3aef76d`](https://github.com/3dct/vIQA/commit/3aef76d95204a3fb3fdc736aaafbdc62e85cb0c4))

### Feature

* feat: add 2D visualization for class ImageArray ([`d09dfad`](https://github.com/3dct/vIQA/commit/d09dfad695470fea67b01a9e38f05154e31c21d1))

* feat: add parameter to export image in visualization functions ([`6e418e2`](https://github.com/3dct/vIQA/commit/6e418e2769a8168b2b9ba67a467c8175368e4851))

### Fix

* fix: update 2D visualization for snr and cnr

fix visualization based on 4d7c3e02 ([`0bab2b8`](https://github.com/3dct/vIQA/commit/0bab2b8534a320b18780f9a131c6384bf79c6303))


## v1.7.0 (2024-08-22)

### Documentation

* docs: update docstring for load_utils.py:load_raw

add periods to sentences in the &#34;Raises&#34; section of docstring ([`c3f3591`](https://github.com/3dct/vIQA/commit/c3f35911c7bd86e0860db92659115e7a41de33c9))

* docs: add pre-commit.ci status badge ([`b02ca8b`](https://github.com/3dct/vIQA/commit/b02ca8b7820f9333d0f3456905cd14c641693131))

### Feature

* feat: add custom visualization

add method for class ImageArray and function visualize_3d for custom visualization of slices of 3d volumes, update documentation for the respective method and function ([`8423592`](https://github.com/3dct/vIQA/commit/8423592d1e691b72550b5c2d915e1491db9056b1))

### Fix

* fix: change orientation of loaded image array

add rotation and flip when loading from binary, update visualization functions for cnr and snr ([`4d7c3e0`](https://github.com/3dct/vIQA/commit/4d7c3e025a40e4a61d911ad9aeda75bd96ca2923))

### Performance

* perf: use attribute of class ImageArray for method describe

attribute mean_value in method describe is now used, fix documentation for class ImageArray ([`b0830cf`](https://github.com/3dct/vIQA/commit/b0830cf2ede2e20a5349775ab23649557ca72f85))


## v1.6.1 (2024-08-20)

### Fix

* fix: remove __init__.py files for C extensions

import of the package is now possible again ([`e747cf7`](https://github.com/3dct/vIQA/commit/e747cf7f184b43b67bf3024d0a7bdd3cd0129546))


## v1.6.0 (2024-08-20)

### Documentation

* docs: change title of Changelog page to Release History ([`ef913a6`](https://github.com/3dct/vIQA/commit/ef913a6d0671ac1d7de26184ce2f7aadd6f1e2d2))

* docs: add changelog page ([`730ecf1`](https://github.com/3dct/vIQA/commit/730ecf17b6b4932479aee3ceebeeb40a42352c53))

* docs: add contributor covenant shield ([`1ab08c1`](https://github.com/3dct/vIQA/commit/1ab08c166fae1de6005d278ef83f75b71f6084b8))

### Feature

* feat(load_utils.py): add support for .tiff files ([`a7c8b74`](https://github.com/3dct/vIQA/commit/a7c8b74d01c4ed4a2476a3eef71b385651b15389))


## v1.5.2 (2024-08-16)

### Build

* build: add stub files and __init__.py files for C extensions to build process ([`6251b98`](https://github.com/3dct/vIQA/commit/6251b981d92874a22b7a927c9aecb8196a0ae8b9))

* build: update build_wheels_and_publish.yaml

add fetch_depth for code checkout ([`1fd84d2`](https://github.com/3dct/vIQA/commit/1fd84d21edfeee91a297056164bebac220713753))

* build: update workflows

add pull_request trigger for Build, add concurrency for Documentation, add event trigger for Documentation ([`6e8bc98`](https://github.com/3dct/vIQA/commit/6e8bc98ababa3264667651bed4913c3102be424e))

* build: fix build_wheels_and_publish.yaml

add code checkout to github release job ([`e2c9a39`](https://github.com/3dct/vIQA/commit/e2c9a39b7ba22b522502db637e822097a85a5c62))

### Documentation

* docs: update README.md

update Project Status badge to Active, clear todo list, add contributing, improve documentation ([`503e4f7`](https://github.com/3dct/vIQA/commit/503e4f744876cf8810f35c6e4ce077b4e2bd9a62))

* docs: rename batch mode page

make &#34;Module&#34; lowercase as all other modules ([`e1105e1`](https://github.com/3dct/vIQA/commit/e1105e12c183d619aa43642787423e21f151d35e))

* docs: restructure docs into api reference and usage page ([`0dad6df`](https://github.com/3dct/vIQA/commit/0dad6dfbf8a1748745704247df95f55c5d4f9d64))

* docs: add developer guide ([`5282655`](https://github.com/3dct/vIQA/commit/5282655f1d7d399861dd16f61846b9deef6b3ef2))

* docs: add favicon ([`f2b6ee1`](https://github.com/3dct/vIQA/commit/f2b6ee1018e7e798be75587f954764e20200c0a8))

### Fix

* fix(mad.py): update check for im_slice

in (int or None) int can always be evaluated as true ([`a245fe9`](https://github.com/3dct/vIQA/commit/a245fe98e5290c49253a651f8e93d0fe30eb1bc4))


## v1.5.1 (2024-08-08)

### Build

* build: fix build_wheels_and_publish.yaml

add own job for upload to github, Fix test install from PyPI ([`aa6c1d5`](https://github.com/3dct/vIQA/commit/aa6c1d57ceb1376bf5e7af020bc3a15e16291d36))

* build: fix build_wheels_and_publish.yaml

fix permissions for action ([`23301a2`](https://github.com/3dct/vIQA/commit/23301a29d9db4b987b4dca0c6cfbfa60e0a3e27b))

* build: fix build_wheels_and_publish.yaml

fix dependency for publishing to GitHub ([`9a48467`](https://github.com/3dct/vIQA/commit/9a48467304b4024ce693e200dd5087920508ae43))

* build: add semantic release ([`9ab3aeb`](https://github.com/3dct/vIQA/commit/9ab3aeb123f0a63f19a10bfd0bfadb73827fb10d))

* build: add test install after release

tests the install from pypi after publishing ([`054ffd5`](https://github.com/3dct/vIQA/commit/054ffd5a7030968065f0a1693d61b6c97c0ea5d5))

### Fix

* fix(msssim.py): fix scale weights

parameter scale_weights in msssim now gets converted from list to tensor. Fixes bug where using scale_weights resulted in an error. ([`b1aad49`](https://github.com/3dct/vIQA/commit/b1aad4984ec0fb401fbb03fe527fbb8aca90d264))


## v1.5.0 (2024-08-07)

### Documentation

* docs: add docs for ImageArray

add sphinx generated .rst file for ImageArray class ([`a8d4b6b`](https://github.com/3dct/vIQA/commit/a8d4b6bb8b43faf79a437fd489932898803f4e33))

### Feature

* feat: add tqdm progress bar ([`615d0f3`](https://github.com/3dct/vIQA/commit/615d0f3a2f3a8342683775e3b340f35fadc3e106))


## v1.4.0 (2024-08-07)

### Feature

* feat(batch_mode.py): improve metadata textfile

add underscores to structure exported textfile in export_metadata ([`3d00279`](https://github.com/3dct/vIQA/commit/3d002794c36e09f966750b9349dfbab664392b64))


## v1.3.0 (2024-08-07)

### Documentation

* docs(ssim.py): Fix doc for structural_similarity()

parameter data_range default is None, not 255 ([`dad5432`](https://github.com/3dct/vIQA/commit/dad5432857e793df859cc3f884ad9cb78f57a65b))

### Feature

* feat(batch_mode.py): add exceptions for batch mode

raise an exception if file extension is not correctly specified in export_results or export_metadata ([`be72f3b`](https://github.com/3dct/vIQA/commit/be72f3b96bce02c62c16c43622259dcf67872cea))


## v1.2.3 (2024-08-07)


## v1.2.2 (2024-08-07)

### Fix

* fix: use np.round instead of round

fixes problems with the new ImageArray class ([`96579a0`](https://github.com/3dct/vIQA/commit/96579a0f7f0d0f9e402ed973f5d3f6ed31a6bf87))


## v1.2.1 (2024-08-07)

### Fix

* fix: add exception in data loading

throws exception when loading a binary image and image size and dimensions do not match; rename ImageArray.mean to ImageArray.mean_value ([`df53d06`](https://github.com/3dct/vIQA/commit/df53d06b6a39c7b1aadb9bd9db6f029aa28d76f0))


## v1.2.0 (2024-08-06)

### Feature

* feat(batch_mode.py): add function to export metadata

function writes custom parameters and package version to a .txt file ([`5b8e607`](https://github.com/3dct/vIQA/commit/5b8e6072cd052068f3378038817e54371321ade8))


## v1.1.0 (2024-08-05)

### Feature

* feat: add function to get installed version ([`12e20ac`](https://github.com/3dct/vIQA/commit/12e20acfd494ea47d75cff1249b37b95e01245e6))


## v1.0.0 (2024-08-05)

### Breaking

* feat(ImageArray)!: add class for images

subclass of np.ndarray, calculates image statistics ([`40f0d02`](https://github.com/3dct/vIQA/commit/40f0d029ad416747135f79e8716de040335da592))

