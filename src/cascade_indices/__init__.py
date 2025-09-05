"""
PROJECT:
-------
CCF-paradigm

TITLE:
------
__init__.py

MAIN OBJECTIVE:
---------------
This package initializes the cascade sub-indices modules for enhanced media cascade analysis,
providing a unified interface for different cascade measurement approaches.

Dependencies:
-------------
- base_index module
- journalist_index module

MAIN FEATURES:
--------------
1) Standardized interface for cascade sub-indices
2) Journalist adoption and influence patterns
3) Media outlet diffusion patterns
4) Network topology and connectivity metrics
5) Viral spread and homogenization indicators

Author:
-------
Antoine Lemor
"""

from .base_index import BaseCascadeIndex
from .journalist_index import JournalistIndex

__all__ = [
    'BaseCascadeIndex',
    'JournalistIndex'
]