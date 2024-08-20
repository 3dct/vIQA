# CHANGELOG



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

