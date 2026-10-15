from pathlib import Path

import pytest
import viqa

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODIFIED = DATA_DIR / "test_mod_1110x944x425_32fbit.raw"
REFERENCE = DATA_DIR / "test_ref_1110x944x425_32fbit.raw"

# Regions of the 1110x944x425 test volumes
SLICE_2D_600X400 = (slice(200, 800), 200, slice(0, 400))
SLICE_3D_400X400X200 = (slice(300, 700), slice(300, 700), slice(200, 400))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run (skipped without --runslow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test, use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _load(file_path, **kwargs):
    if not file_path.exists():
        pytest.skip(f"Test data not found: {file_path}")
    return viqa.utils.load_data(str(file_path), **kwargs)


# Full volumes, each loaded only once per test session
@pytest.fixture(scope="session")
def reference_image_3d_255():
    return _load(REFERENCE, data_range=255, normalize=True)


@pytest.fixture(scope="session")
def modified_image_3d_255():
    return _load(MODIFIED, data_range=255, normalize=True)


# Regions are copied, so they do not keep the full volumes in memory
@pytest.fixture(scope="session")
def data_2d_255_600x400(reference_image_3d_255, modified_image_3d_255):
    img_r_2d = reference_image_3d_255[SLICE_2D_600X400].copy()
    img_m_2d = modified_image_3d_255[SLICE_2D_600X400].copy()
    return img_r_2d, img_m_2d


@pytest.fixture(scope="session")
def reference_image_2d_255(reference_image_3d_255):
    return reference_image_3d_255[:, :, 200].copy()


@pytest.fixture(scope="session")
def modified_image_2d_255(modified_image_3d_255):
    return modified_image_3d_255[:, :, 200].copy()


@pytest.fixture(scope="session")
def data_3d_255_400x400x200(reference_image_3d_255, modified_image_3d_255):
    img_r_3d = reference_image_3d_255[SLICE_3D_400X400X200].copy()
    img_m_3d = modified_image_3d_255[SLICE_3D_400X400X200].copy()
    return img_r_3d, img_m_3d


@pytest.fixture(scope="session")
def data_3d_native_400x400x200():
    img_r_3d = _load(REFERENCE)[SLICE_3D_400X400X200].copy()
    img_m_3d = _load(MODIFIED)[SLICE_3D_400X400X200].copy()
    return img_r_3d, img_m_3d
