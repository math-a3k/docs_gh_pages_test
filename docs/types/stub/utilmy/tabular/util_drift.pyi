os.getcwd /home/runner/work/myutil/myutil/utilmy/tabular
from numpy import (
    float64,
    ndarray,
)
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from typing import (
    Dict,
    Optional,
    Union,
)


def log(*s) -> None: ...


def test_all(): ...


def test_anova(df: DataFrame, col1: str, col2: str) -> None: ...


def test_heteroscedacity(
    y: Series,
    y_pred: ndarray,
    pred_value_only: int = ...
) -> Dict[str, Dict[str, float64]]: ...


def test_hypothesis(df_obs: DataFrame, df_ref: DataFrame, method: str = ..., **kw): ...


def test_mutualinfo(
    error: Series,
    Xtest: DataFrame,
    colname: Optional[str] = ...,
    bins: int = ...
) -> Dict[str, float64]: ...


def test_normality(
    error: Series,
    distribution: str = ...,
    test_size_limit: int = ...
) -> Dict[str, Dict[str, Union[float, float64, ndarray]]]: ...


def test_normality2(df: DataFrame, column: str, test_type: str) -> None: ...


def test_plot_qqplot(df: DataFrame, col_name: str) -> None: ...
