"""
Shared utilities: logging, seeding, timing, directory creation, defensive helpers.
"""
import os
import random
import logging
import sys
import time
from functools import wraps
from typing import Any, Callable, Optional

import numpy as np

LOG_FORMAT: str = '%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s'
DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

# Custom VALIDATION log level (between INFO=20 and WARNING=30)
VALIDATION_LEVEL: int = 25
logging.addLevelName(VALIDATION_LEVEL, 'VALIDATION')


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except (ImportError, AttributeError):
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, AttributeError):
        pass


def validation(self, message, *args, **kwargs):
    """Log at VALIDATION level."""
    if self.isEnabledFor(VALIDATION_LEVEL):
        self._log(VALIDATION_LEVEL, message, args, **kwargs)

logging.Logger.validation = validation


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def timeit(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func.__module__)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info("%s completed in %.2f seconds", func.__name__, elapsed)
        return result
    return wrapper


def safe_append(df_list: list, new_df: Any, name: str = "") -> None:
    if new_df is not None and hasattr(new_df, 'empty') and not new_df.empty:
        df_list.append(new_df)
    else:
        logger = get_logger(__name__)
        logger.warning("Empty or None DataFrame encountered for '%s'", name)


def assert_no_nans(df, name: str = "DataFrame") -> None:
    if hasattr(df, 'isnull'):
        nans = df.isnull().sum().sum()
        if nans > 0:
            raise ValueError(f"{name} contains {nans} NaN values after expected fill stage")


def validate_shape(X, y, name: str = "data") -> None:
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{name}: X rows ({X.shape[0]}) != y rows ({y.shape[0]})")


def safe_select_dtypes(df, include=None, exclude=None):
    try:
        return df.select_dtypes(include=include, exclude=exclude)
    except Exception:
        return df
