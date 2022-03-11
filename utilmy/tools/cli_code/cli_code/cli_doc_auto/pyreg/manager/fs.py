#!/usr/bin/env python

import re
import os
import logging


from ..context import config


LOGGER = logging.getLogger()


def list_source_files(srcdir):
    """function list_source_files
    Args:
        srcdir:   
    Returns:
        
    """
    srcs = []

    for dirpath, _, files in os.walk(srcdir):
        for fname in files:
            path = os.path.join(dirpath, fname)
            if any(map(lambda ext: fname.endswith(ext), config.EXT)):
                srcs.append(path)
            else:
                LOGGER.debug('skipping: {}'.format(path))

    return srcs


def filter_by_pattern(paths, patterns):
    """function filter_by_pattern
    Args:
        paths:   
        patterns:   
    Returns:
        
    """
    if not patterns:
        return paths

    result = []
    for path in paths:
        for pattern in patterns:
            if re.match(pattern, path):
                result.append(path)
                break

    return result
