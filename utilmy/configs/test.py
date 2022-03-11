import datetime

import pytest
from box import Box
from yamale import YamaleError

from zz936.configs.util_config import config_validate


# in order to run tests install required packages:
# python-box
# pyyaml
# yamale
# pytest


bad_data = """
string: 1
regex: 'edcba'
number: 14
integer: 1.4
boolean: 0
list: "list?"
enum: False
map: 'hello'
null: "null"
date: 2015-01-01
nest:
    integer: 1
    nest:
        string: "nested"
"""

good_data = """
string: "hello"
regex: 'abcde'
number: 13.12
integer: 2
boolean: True
list: ['hi']
enum: 1
map:
    hello: 1
    another: "hi"
null: null
date: 2015-01-01
nest:
    integer: 1
    nest:
        string: "nested"
"""


@pytest.fixture(autouse=True)
def create_fixtures_data(tmp_path):
    """function create_fixtures_data
    Args:
        tmp_path:   
    Returns:
        
    """
    good_data_yaml = tmp_path / "config_good_data.yaml"
    good_data_yaml.write_text(good_data)
    bad_data_yaml = tmp_path / "config_bad_data.yaml"
    bad_data_yaml.write_text(bad_data)


def test_validate_yaml_types(tmp_path):
    """function test_validate_yaml_types
    Args:
        tmp_path:   
    Returns:
        
    """
    schema = "config_val.yaml"
    data = tmp_path / "config_good_data.yaml"
    result = config_validate(data, schema)

    assert isinstance(result, Box)
    assert result == {
        None: None,
        "boolean": True,
        "date": datetime.date(2015, 1, 1),
        "enum": 1,
        "integer": 2,
        "list": ["hi"],
        "map": {"hello": 1, "another": "hi"},
        "nest": {"integer": 1, "nest": {"string": "nested"}},
        "number": 13.12,
        "regex": "abcde",
        "string": "hello",
    }


def test_validate_yaml_types_failed(tmp_path):
    """function test_validate_yaml_types_failed
    Args:
        tmp_path:   
    Returns:
        
    """
    schema = "config_val.yaml"
    data = tmp_path / "config_bad_data.yaml"

    expected = [
        "string: '1' is not a str.",
        "regex: 'edcba' is not a regex match.",
        "number: 14 is greater than 13.12",
        "integer: '1.4' is not a int.",
        "boolean: '0' is not a bool.",
        "list: 'list?' is not a list.",
        "enum: 'False' not in ('one', True, 1)",
        "map: 'hello' is not a map.",
        "None: 'null' is not a null.",
    ]

    with pytest.raises(YamaleError) as exc:
        config_validate(data, schema)
    actual = exc.value.results[0].errors
    assert sorted(actual) == sorted(expected)


def test_validate_yaml_failed_silent(tmp_path):
    """function test_validate_yaml_failed_silent
    Args:
        tmp_path:   
    Returns:
        
    """
    schema = "config_val.yaml"
    data = tmp_path / "config_bad_data.yaml"
    result = config_validate(data, schema, silent=True)
    assert result is None
