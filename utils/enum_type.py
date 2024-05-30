# -*- coding: utf-8 -*-

"""
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.
    - ``GENERAL``: General Recommendation, DEFAULT
    - ``SEQUENTIAL``: Sequential Recommendation
    """
    GENERAL = 0
    SEQUENTIAL = 1

