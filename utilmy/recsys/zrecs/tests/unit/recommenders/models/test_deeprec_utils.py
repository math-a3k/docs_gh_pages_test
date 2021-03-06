# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
from recommenders.datasets.amazon_reviews import (
    download_and_extract,
    data_preprocessing,
)

try:
    from recommenders.models.deeprec.deeprec_utils import (
        prepare_hparams,
        download_deeprec_resources,
        load_yaml,
    )
    from recommenders.models.deeprec.io.iterator import FFMTextIterator
    from recommenders.models.deeprec.io.dkn_item2item_iterator import (
        DKNItem2itemTextIterator,
    )
    from recommenders.models.deeprec.io.dkn_iterator import DKNTextIterator
    from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
    from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel
    import tensorflow as tf
except ImportError:
    pass  # skip this import if we are in cpu environment


@pytest.mark.parametrize(
    "must_exist_attributes", ["FEATURE_COUNT", "data_format", "dim"]
)
@pytest.mark.gpu
def test_prepare_hparams(deeprec_resource_path, must_exist_attributes):
    """function test_prepare_hparams
    Args:
        deeprec_resource_path:   
        must_exist_attributes:   
    Returns:
        
    """
    data_path = os.path.join(deeprec_resource_path, "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )
    hparams = prepare_hparams(yaml_file)
    assert hasattr(hparams, must_exist_attributes)


@pytest.mark.gpu
def test_load_yaml_file(deeprec_resource_path):
    """function test_load_yaml_file
    Args:
        deeprec_resource_path:   
    Returns:
        
    """
    data_path = os.path.join(deeprec_resource_path, "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )

    config = load_yaml(yaml_file)
    assert config is not None
