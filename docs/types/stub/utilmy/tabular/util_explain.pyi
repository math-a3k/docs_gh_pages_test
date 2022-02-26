2 traces failed to decode; use -v for details
from imodels.algebraic.slim import SLIMRegressor
from imodels.rule_set.rule_fit import RuleFitRegressor
from imodels.tree.figs import FIGSRegressor
from typing import (
    Optional,
    Union,
)


def model_evaluate(
    model: Union[RuleFitRegressor, SLIMRegressor, FIGSRegressor],
    data_pars: dict
) -> None: ...


def model_extract_rules(
    model: Union[RuleFitRegressor, SLIMRegressor, FIGSRegressor]
) -> None: ...


def model_fit(
    name: str = ...,
    model_pars: Optional[dict] = ...,
    data_pars: Optional[dict] = ...,
    do_eval: bool = ...,
    **kw
) -> Union[RuleFitRegressor, SLIMRegressor, FIGSRegressor]: ...


def model_load(
    path: str = ...
) -> Union[RuleFitRegressor, SLIMRegressor, FIGSRegressor]: ...


def model_save(
    model: Union[RuleFitRegressor, SLIMRegressor, FIGSRegressor],
    path: Optional[str] = ...,
    info: None = ...
) -> None: ...


def test1() -> None: ...


def test2() -> None: ...


def test_all() -> None: ...
