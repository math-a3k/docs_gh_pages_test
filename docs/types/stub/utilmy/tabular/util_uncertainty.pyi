from mapie.classification import MapieClassifier
from mapie.regression import MapieRegressor
from numpy import ndarray
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.linear_model._base import LinearRegression
from sklearn.tree._classes import DecisionTreeClassifier
from typing import (
    List,
    Optional,
    Tuple,
    Type,
    Union,
)


def load_function_uri(
    uri_name: str = ...
) -> Union[Type[MapieRegressor], Type[MapieClassifier]]: ...


def model_evaluate(
    model: Union[MapieClassifier, MapieRegressor],
    data_pars: dict,
    predict_pars: dict
) -> None: ...


def model_fit(
    name: str = ...,
    model: Optional[Union[RandomForestClassifier, DecisionTreeClassifier, LinearRegression]] = ...,
    mapie_pars: Optional[dict] = ...,
    predict_pars: Optional[dict] = ...,
    data_pars: Optional[dict] = ...,
    do_prefit: bool = ...,
    do_eval: bool = ...,
    test_size: float = ...
) -> Union[MapieClassifier, MapieRegressor]: ...


def model_load(path: str = ...) -> Union[MapieClassifier, MapieRegressor]: ...


def model_save(
    model: Union[MapieClassifier, MapieRegressor],
    path: Optional[str] = ...,
    info: None = ...
) -> None: ...


def model_viz_classification_preds(preds: ndarray, y_test: ndarray) -> None: ...


def test1() -> None: ...


def test2() -> None: ...


def test_all() -> None: ...


def test_data_classifier_digits() -> Tuple[ndarray, ndarray, ndarray, ndarray, List[str]]: ...


def test_data_regression_boston() -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]: ...
