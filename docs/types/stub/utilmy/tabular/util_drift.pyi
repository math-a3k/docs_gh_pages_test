os.getcwd /workspace/myutil/utilmy/tabular
from numpy import (
    float64,
    ndarray,
)
from pandas.core.series import Series
from typing import (
    Dict,
    Union,
)


def log(*s) -> None: ...


def test_all(): ...


def test_normality(
    error: Series,
    distribution: str = ...,
    test_size_limit: int = ...
) -> Dict[str, Dict[str, Union[float, float64, ndarray]]]: ...
