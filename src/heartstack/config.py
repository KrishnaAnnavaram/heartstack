"""Settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    pass


def _get(env, name, default):
    v = env.get(name, "")
    return v.strip() if v and v.strip() else default


def _num(env, name, default, low, high, cast):
    raw = _get(env, name, str(default))
    try:
        v = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not low <= v <= high:
        raise ConfigError(f"{name} must be between {low} and {high}")
    return v


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data/raw")
    work_dir: Path = Path("runs")
    seed: int = 42
    test_size: float = 0.2
    outer_folds: int = 5
    inner_folds: int = 3
    n_jobs: int = 1
    target_sensitivity: float = 0.90
    n_bootstrap: int = 1000
    include_source_feature: bool = False
    missing_indicators: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        src_flag = _get(env, "HEARTSTACK_INCLUDE_SOURCE", "false").lower()
        ind_flag = _get(env, "HEARTSTACK_MISSING_INDICATORS", "true").lower()
        if src_flag not in {"true", "false"} or ind_flag not in {"true", "false"}:
            raise ConfigError("HEARTSTACK_INCLUDE_SOURCE and HEARTSTACK_MISSING_INDICATORS must be true or false")
        return cls(
            data_dir=Path(_get(env, "HEARTSTACK_DATA_DIR", str(cls.data_dir))),
            work_dir=Path(_get(env, "HEARTSTACK_WORK_DIR", str(cls.work_dir))),
            seed=_num(env, "HEARTSTACK_SEED", cls.seed, 0, 2**31 - 1, int),
            test_size=_num(env, "HEARTSTACK_TEST_SIZE", cls.test_size, 0.05, 0.5, float),
            outer_folds=_num(env, "HEARTSTACK_OUTER_FOLDS", cls.outer_folds, 2, 20, int),
            inner_folds=_num(env, "HEARTSTACK_INNER_FOLDS", cls.inner_folds, 2, 20, int),
            n_jobs=_num(env, "HEARTSTACK_N_JOBS", cls.n_jobs, -1, 256, int),
            target_sensitivity=_num(env, "HEARTSTACK_TARGET_SENSITIVITY", cls.target_sensitivity, 0.5, 0.999, float),
            n_bootstrap=_num(env, "HEARTSTACK_BOOTSTRAP", cls.n_bootstrap, 0, 100000, int),
            include_source_feature=src_flag == "true",
            missing_indicators=ind_flag == "true",
        )
