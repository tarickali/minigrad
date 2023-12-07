"""
title : types.py
create : @tarickali 23/12/07
update : @tarickali 23/12/07
"""

import numpy as np

Array = np.ndarray | list
Numeric = float | int | bool
Dtype = np.dtype
Shape = tuple[int, ...]
