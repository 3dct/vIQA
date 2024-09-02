# CHANGELOG



## v1.11.0 (2024-09-02)

### Chore

* chore: update .pre-commit-config.yaml

add submodule checkout, add custom autoupdate commit message, change sphinx building stage to pre-push ([`8998a73`](https://github.com/3dct/vIQA/commit/8998a736ea1883835139af88df2652888e7813d8))

* chore: update LICENSE.txt

include License statement for qmeasure submodule, fix file paths, update style ([`4ec7e17`](https://github.com/3dct/vIQA/commit/4ec7e1739c46d27e4cdc4f085b66882d53e2ecf7))

### Ci

* ci: add recursive checkout ([`ef72c4a`](https://github.com/3dct/vIQA/commit/ef72c4a11196a925d8aaf04933ca533a1f1a7640))

* ci: make checkout branch variable for wheel and sdist building

set checkout branch for semantic_release to main ([`59d0375`](https://github.com/3dct/vIQA/commit/59d037557cfb899eb6191e61aad331a6afe4bbf6))

* ci: remove sleep step in job test_install ([`fde39ba`](https://github.com/3dct/vIQA/commit/fde39baa64fa830d35451c35dd905de7f2affc04))

### Feature

* feat: add function to crop images

cal cropping function during loading if parameter roi is given ([`140d101`](https://github.com/3dct/vIQA/commit/140d101ace4ffa3dbd97ecc11db8ada3dba0ae6b))

### Refactor

* refactor: add warning to visualize method in class ImageArray

add warning that parameter slices will be ignored if method is called on a 2D image ([`cd496be`](https://github.com/3dct/vIQA/commit/cd496be36ed3ad8ba638ba4de04c942ff53ec5f4))

* refactor: handle data loading in interface classes

move data loading and checking to score methods of metrics interface classes ([`1ad011b`](https://github.com/3dct/vIQA/commit/1ad011bea42b52c94d3e17f26b750e21394f3e1b))

* refactor: use qmeasure submodule ([`1c1ae61`](https://github.com/3dct/vIQA/commit/1c1ae61dc8b061fe21c29fba69e705274eb07897))


## v1.10.0 (2024-08-27)

### Chore

* chore: update docs config to autodocument inherited members ([`9bc640f`](https://github.com/3dct/vIQA/commit/9bc640f86e3b01997ca27920b46a5e900da83e93))

* chore: update documentation build

add dependencies for Sphinx in GitHub Actions job ([`4596169`](https://github.com/3dct/vIQA/commit/4596169db77fc0c3f5940e5607e0e8e9baab25a0))

* chore: update release process

build only after semantic_release job ran ([`5fe798c`](https://github.com/3dct/vIQA/commit/5fe798c625e00786938a8bc0dbec272c0bd6799b))

### Documentation

* docs: update docs

remove sphinx-notfound-page, temporarily disable hoverxref ([`614d52a`](https://github.com/3dct/vIQA/commit/614d52aa0a05e87c664ed99fb8f7e76b43b9c589))

* docs: add sphinx-autobuild to developer_guide.rst ([`59381a4`](https://github.com/3dct/vIQA/commit/59381a4910dfbf2b4386c3766241dd54a4d42349))

### Feature

* feat: add function to compute multiple metrics at once

rename batch_mode.py to multiple.py, add MultipleMetrics class, move export_metadata to it&#39;s own function in utils.py, add base class for classes of type multiple, update class BatchMetrics ([`6f4f8a7`](https://github.com/3dct/vIQA/commit/6f4f8a7765078a322e12cfca6720f3d8e0842082))

### Refactor

* refactor: add new parent class metric

rename export_csv to export_results, restructure abstract base classes for metrics ([`4bc3bef`](https://github.com/3dct/vIQA/commit/4bc3bef3c85b9e21c4bf0b2259fb9a957c193e4b))

### Unknown

* book: update Image_Comparison.ipynb accordingly to 6f4f8a77 ([`8ac53e8`](https://github.com/3dct/vIQA/commit/8ac53e8d7a9d3fa193fade7cc295f5b581e917fb))


## v1.9.1 (2024-08-23)

### Chore

* chore: change GitHub emojis to emoji characters in README.md

improve rendering for PyPI ([`0a04811`](https://github.com/3dct/vIQA/commit/0a048118e9b9587d5cf498e914436559efb99d3d))

* chore: update automatic build process in build_wheels_and_publish.yaml ([`bcdc3b1`](https://github.com/3dct/vIQA/commit/bcdc3b13a365f63db2a5e1eee27a358e197c186a))

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

### Chore

* chore: update .gitignore

remove docs/source/**/generated/ files ([`be86072`](https://github.com/3dct/vIQA/commit/be86072bec3ff1d1df010d17466cf2cc51c4802c))

### Feature

* feat(batch_mode.py): add parameter data_range to export_metadata ([`00cb4c4`](https://github.com/3dct/vIQA/commit/00cb4c427ae4cf17db83a006397dca7aaf3ef0cc))

* feat: add parameter to set order of scaling in batch_mode.py ([`0c3886c`](https://github.com/3dct/vIQA/commit/0c3886c8d00e25b9400ce87eb801a4d3be3b0ac9))

### Fix

* fix: add Exception and warning to handle non 3D volumes in qmeasure.py ([`33e288a`](https://github.com/3dct/vIQA/commit/33e288ae032a1ed127c23c65412d084ae6054f8b))


## v1.8.0 (2024-08-22)

### Build

* build: update numpy dependency for build to be lower than 2.0.0 ([`54ac725`](https://github.com/3dct/vIQA/commit/54ac725020e2394baa9188774fce5f45ce49cfb7))

### Chore

* chore: remove check install job from build_wheels_and_publish.yaml ([`235cd1c`](https://github.com/3dct/vIQA/commit/235cd1c572ab37ac48ec1b12eb559d7d3f788beb))

* chore: use pip instead of pip3 in check_import job ([`d739e83`](https://github.com/3dct/vIQA/commit/d739e83c02972c61ca87410c4ef4b7f405bfa8d7))

* chore: add Set up Python step check_import job ([`70448e3`](https://github.com/3dct/vIQA/commit/70448e36fec50348c5c6904b15e9a119e8efba42))

* chore: add -vv argument in check_import job for debugging ([`03e50b1`](https://github.com/3dct/vIQA/commit/03e50b128bc1d1904166213feb4bad7ac362de86))

* chore: fix check_import job to use non-editable install ([`e772e8f`](https://github.com/3dct/vIQA/commit/e772e8ff51cacf7a525efd85d002f00c1daf6860))

* chore: move isort config from ruff.isort to ruff.lint.isort ([`8d7f21b`](https://github.com/3dct/vIQA/commit/8d7f21be7a8164d9310837fea7f8e248c18fbdc5))

* chore: update release process in build_wheels_and_publish.yaml

add check for import to catch non-importable releases, add upload to TestPyPI and check installing from there before real upload ([`53d69e9`](https://github.com/3dct/vIQA/commit/53d69e9edc82c6e0f896fe0e79e4039e904b39f4))

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

### Chore

* chore: update version to 1.6.1

fix versioning for python-semantic-release ([`6d00255`](https://github.com/3dct/vIQA/commit/6d0025588954ea20f28f316d768e222a00756eca))

### Ci

* ci: add commit type book to python semantic release ([`ec50245`](https://github.com/3dct/vIQA/commit/ec50245c91e763a2ec705fb688b4b2017a08e227))

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

### Unknown

* book: update visualization in jupyter notebooks

visualization now reflects the changes in commit 4d7c3e02 ([`7d94131`](https://github.com/3dct/vIQA/commit/7d94131d0c7a9a0296dcb819d0fb3c72f652f6ab))


## v1.6.1 (2024-08-20)

### Ci

* ci: add Setup Python step before Test install from PyPI ([`4471af6`](https://github.com/3dct/vIQA/commit/4471af61882190b62e9db019e5054caf16915409))

* ci: skip build-docs and mypy on pre-commit.ci ([`32654c5`](https://github.com/3dct/vIQA/commit/32654c58e0c6331452fbc5a5b22239e9e08d0db8))

### Fix

* fix: remove __init__.py files for C extensions

import of the package is now possible again ([`e747cf7`](https://github.com/3dct/vIQA/commit/e747cf7f184b43b67bf3024d0a7bdd3cd0129546))


## v1.6.0 (2024-08-20)

### Ci

* ci: update .pre-commit-config.yaml

exclude CHANGELOG.md and CODE_OF_CONDUCT.md from codespell hook, activate sphinx hook for push only, remove stage commit from validate-pyproject hook, add exclude for conda recipes for check-yaml hook ([`d84c21f`](https://github.com/3dct/vIQA/commit/d84c21f12072d6341f7282249d95c11f81b121cc))

* ci: update build_wheels_and_publish.yaml

remove vcs_release=&#34;false&#34; argument from semantic_release job, update test install ([`2eae29e`](https://github.com/3dct/vIQA/commit/2eae29ec52fe36a546b08340c0d47f9262520f1a))

* ci: add specific code checkout to build_wheels_and_publish.yaml

add &#34;ref: main&#34; to checkout step to ensure checking out the latest commit, temporarily deactivate conditions for jobs to release last version ([`9e8b5ce`](https://github.com/3dct/vIQA/commit/9e8b5ce5de64b8194b5cc6856fa2c4f96a38db60))

### Documentation

* docs: change title of Changelog page to Release History ([`ef913a6`](https://github.com/3dct/vIQA/commit/ef913a6d0671ac1d7de26184ce2f7aadd6f1e2d2))

* docs: add changelog page ([`730ecf1`](https://github.com/3dct/vIQA/commit/730ecf17b6b4932479aee3ceebeeb40a42352c53))

* docs: add contributor covenant shield ([`1ab08c1`](https://github.com/3dct/vIQA/commit/1ab08c166fae1de6005d278ef83f75b71f6084b8))

### Feature

* feat(load_utils.py): add support for .tiff files ([`a7c8b74`](https://github.com/3dct/vIQA/commit/a7c8b74d01c4ed4a2476a3eef71b385651b15389))

### Unknown

* Create CODE_OF_CONDUCT.md ([`3860e91`](https://github.com/3dct/vIQA/commit/3860e91f7fbd0bb6117d909185686c989bfa5437))


## v1.5.2 (2024-08-16)

### Build

* build: add stub files and __init__.py files for C extensions to build process ([`6251b98`](https://github.com/3dct/vIQA/commit/6251b981d92874a22b7a927c9aecb8196a0ae8b9))

* build: update build_wheels_and_publish.yaml

add fetch_depth for code checkout ([`1fd84d2`](https://github.com/3dct/vIQA/commit/1fd84d21edfeee91a297056164bebac220713753))

* build: update workflows

add pull_request trigger for Build, add concurrency for Documentation, add event trigger for Documentation ([`6e8bc98`](https://github.com/3dct/vIQA/commit/6e8bc98ababa3264667651bed4913c3102be424e))

* build: fix build_wheels_and_publish.yaml

add code checkout to github release job ([`e2c9a39`](https://github.com/3dct/vIQA/commit/e2c9a39b7ba22b522502db637e822097a85a5c62))

### Chore

* chore: update PyPI classifiers ([`21a1994`](https://github.com/3dct/vIQA/commit/21a199410b03f9e5f3705fe9304175a2dd49469e))

* chore: add CONTRIBUTING.md ([`8491e64`](https://github.com/3dct/vIQA/commit/8491e64759385c1881b3bffe7e31409161792483))

* chore: update dev dependencies ([`ac22ca6`](https://github.com/3dct/vIQA/commit/ac22ca6d17ddaaf3a3a585a9f3d7895eb88c0c53))

* chore: remove sphinx from pre-commit ([`2372f99`](https://github.com/3dct/vIQA/commit/2372f99bc475fc16d6e320fed4474dc06c7e17cd))

* chore: update args for pre-commit-sphinx ([`c672fa5`](https://github.com/3dct/vIQA/commit/c672fa576512dfc2ae6833b6b0e3d7a3e4dbe2e5))

* chore: update dev dependencies ([`d769aab`](https://github.com/3dct/vIQA/commit/d769aabe8b4bf1b8eebf258e21abcfa63e614188))

* chore: add pre-commit ([`83c300c`](https://github.com/3dct/vIQA/commit/83c300ca39606ba4253bc665271a5dd39e13748b))

* chore: add lint rules to ignore

ignore E203 and C901 ([`1372c60`](https://github.com/3dct/vIQA/commit/1372c605b226fb56dd3d76af9e7f1a2c5a8b019b))

### Ci

* ci: update build_wheels_and_publish.yaml

perform jobs only if condition released == true is met, add explicit tag for upload_github job ([`c941bd9`](https://github.com/3dct/vIQA/commit/c941bd9f04a85d4c856b6399f1d31d4f9e4de34c))

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

### Refactor

* refactor: update test assertion

change is 255 to == 255 ([`df39f34`](https://github.com/3dct/vIQA/commit/df39f345890cb280594b849f4f009d1c52f2b166))

### Style

* style: update end of file new line ([`3c89f86`](https://github.com/3dct/vIQA/commit/3c89f863cd94c2ad84d21549a27641f9080177a6))

* style: format files

remove trailing whitespace, adjust end of files ([`1cc275c`](https://github.com/3dct/vIQA/commit/1cc275c4de2933e9b7365280f006013902e8c507))

* style: add mypy check

implement mypy check, update files for mypy, add stub files qmeasurecalc.pyi and statisticscalc.pyi ([`1997516`](https://github.com/3dct/vIQA/commit/19975166b487609775126bb2fff5dce06ab2f8ef))

* style: add yamllint

implement yamllint and refactor .yaml files ([`6c46ace`](https://github.com/3dct/vIQA/commit/6c46ace37acc9285445b3e7467dabccbb04757af))

* style: add rst check

implement rst check and refactor .rst files ([`92f2dbd`](https://github.com/3dct/vIQA/commit/92f2dbdf89d19b877bb5e4e2f37049eda1b2ec3c))

* style: update formatting

use ruff formater to format codebase ([`dc48022`](https://github.com/3dct/vIQA/commit/dc48022d93f3c478a460e5545d1a39ad93e2075c))

* style: update linting

fix and update ruff config, fix lint errors ([`452b6fc`](https://github.com/3dct/vIQA/commit/452b6fcf3da3e7aefa8ff5eae965dbcc1a9a533a))

### Unknown

* Merge remote-tracking branch &#39;github/main&#39; ([`609891d`](https://github.com/3dct/vIQA/commit/609891d6f1114dd15bd33bd85ee8f0e1029dfedf))


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

### Chore

* chore: set versions for main dependencies ([`a1b8454`](https://github.com/3dct/vIQA/commit/a1b8454d568cffd9a897969d8986048cac784813))

* chore: update ci for documentation

install tqdm in github runner for documentation building ([`46c60f6`](https://github.com/3dct/vIQA/commit/46c60f603f0f31e2eac4b6ac0b723dc576481271))

### Fix

* fix(msssim.py): fix scale weights

parameter scale_weights in msssim now gets converted from list to tensor. Fixes bug where using scale_weights resulted in an error. ([`b1aad49`](https://github.com/3dct/vIQA/commit/b1aad4984ec0fb401fbb03fe527fbb8aca90d264))

### Refactor

* refactor(utils.py): remove import of normalize_data() ([`6a2bc73`](https://github.com/3dct/vIQA/commit/6a2bc7350497128c177b488c6fe2fe8b3efe81e5))


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

### Refactor

* refactor: update deprecation warnings ([`19b416b`](https://github.com/3dct/vIQA/commit/19b416be438a905575ed1b9a19c60892c4ac5e4c))


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

### Chore

* chore(README.md): add badges ([`65bcc2e`](https://github.com/3dct/vIQA/commit/65bcc2ef6038b8284ad11956a96ae0c13602b87b))

### Ci

* ci: update workflow names, update condition for documentation publishing ([`17db8fe`](https://github.com/3dct/vIQA/commit/17db8fec74d36c057712a78ee3c30979d99c652d))

