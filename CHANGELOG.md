# CHANGELOG



## v1.5.1 (2024-08-08)

### Build

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

