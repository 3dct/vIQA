import pytest
import viqa

REFERENCE = "../data/test_ref.raw"
MODIFIED = "../data/test_mod.raw"


@pytest.fixture(scope="session")
def data_2d_255_600x400():
    img_r_2d = viqa.utils.load_data(REFERENCE, data_range=255, normalize=True)[200:800, 200, 200:600]
    img_m_2d = viqa.utils.load_data(MODIFIED, data_range=255, normalize=True)[200:800, 200, 200:600]
    return img_r_2d, img_m_2d


@pytest.fixture(scope="session")
def reference_image_2d_255():
    img_m_3d = viqa.utils.load_data(REFERENCE, data_range=255, normalize=True)[:, :, 200]
    return img_m_3d


@pytest.fixture(scope="session")
def modified_image_2d_255():
    img_m_3d = viqa.utils.load_data(MODIFIED, data_range=255, normalize=True)[:, :, 200]
    return img_m_3d


@pytest.fixture(scope="session")
def data_3d_255_400x400x200():
    img_r_3d = viqa.utils.load_data(REFERENCE, data_range=255, normalize=True)[300:700, 300:700, 300:500]
    img_m_3d = viqa.utils.load_data(MODIFIED, data_range=255, normalize=True)[300:700, 300:700, 300:500]
    return img_r_3d, img_m_3d


@pytest.fixture(scope="session")
def data_3d_native_400x400x200():
    img_r_3d = viqa.utils.load_data(REFERENCE)[300:700, 300:700, 300:500]
    img_m_3d = viqa.utils.load_data(MODIFIED)[300:700, 300:700, 300:500]
    return img_r_3d, img_m_3d


@pytest.fixture(scope="session")
def reference_image_3d_255():
    img_m_3d = viqa.utils.load_data(REFERENCE, data_range=255, normalize=True)
    return img_m_3d


@pytest.fixture(scope="session")
def modified_image_3d_255():
    img_m_3d = viqa.utils.load_data(MODIFIED, data_range=255, normalize=True)
    return img_m_3d
