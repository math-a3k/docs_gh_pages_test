from imodels.algebraic.slim import SLIMRegressor
from imodels.rule_set.rule_fit import RuleFitRegressor
from imodels.tree.figs import FIGSRegressor
from numpy import ndarray
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)


def model_evaluate(
    model: Union[RuleFitRegressor, FIGSRegressor, SLIMRegressor],
    data_pars: dict
) -> None: ...


def model_extract_rules(
    model: Union[RuleFitRegressor, FIGSRegressor, SLIMRegressor]
) -> None: ...


def model_fit(
    name: str = ...,
    model_pars: Optional[dict] = ...,
    data_pars: Optional[dict] = ...,
    do_eval: bool = ...,
    **kw
) -> Union[RuleFitRegressor, FIGSRegressor, SLIMRegressor]: ...


def model_load(
    path: str = ...
) -> Union[RuleFitRegressor, FIGSRegressor, SLIMRegressor]: ...


def model_save(
    model: Union[RuleFitRegressor, FIGSRegressor, SLIMRegressor],
    path: Optional[str] = ...,
    info: None = ...
) -> None: ...


def test1() -> None: ...


def test2() -> None: ...


def test_all() -> None: ...


def test_data_classifier_diabetes() -> Tuple[ndarray, ndarray, ndarray, ndarray, List[str]]: ...


def test_data_regression_boston() -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]: ...
