import pytest

from heartstack.config import Settings
from heartstack.sources import build_dataset
from heartstack.synthetic import write_synthetic


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    return write_synthetic(tmp_path_factory.mktemp("raw"), n_hospital=160, n_kaggle=120, n_cleveland=70, seed=3)


@pytest.fixture(scope="session")
def dataset(raw_dir):
    return build_dataset(raw_dir)


@pytest.fixture()
def fast_settings(tmp_path):
    return Settings(data_dir=tmp_path, work_dir=tmp_path / "out", seed=1, outer_folds=2, inner_folds=2,
                    n_bootstrap=30, n_jobs=1)
