from typing import Any
import numpy as np


def validate_type(value: Any, expected_type: Any, var_name: str, can_be_none: bool = False) -> None:
    """
    Validate that a variable is of the expected type.

    :param value: The variable to check.
    :type value: Any
    :param expected_type: The expected type of the variable.
    :type expected_type: type
    :param var_name: The name of the variable (for error messages).
    :type var_name: str
    :param can_be_none: Whether the variable is allowed to be None.
    :type can_be_none: bool
    :raises TypeError: If the variable is not of the expected type.
    """
    if can_be_none and value is None:
        return
    if not isinstance(value, expected_type):
        raise TypeError(f"Expected {var_name} to be of type {expected_type.__name__}, but got {type(value).__name__}.")

def validate_dtype(value: Any, expected_dtype: Any, var_name: str, can_be_none: bool = False) -> None:
    """
    Validate that a numpy array has the expected data type.

    :param value: The numpy array to check.
    :type value: Any
    :param expected_dtype: The expected data type of the numpy array.
    :type expected_dtype: type
    :param var_name: The name of the variable (for error messages).
    :type var_name: str
    :param can_be_none: Whether the variable is allowed to be None.
    :type can_be_none: bool
    :raises TypeError: If the numpy array does not have the expected data type.
    """
    validate_type(value, expected_type=np.ndarray, var_name=var_name, can_be_none=can_be_none)
    if can_be_none and value is None:
        return
    if not hasattr(value, 'dtype'):
        raise TypeError(f"Expected {var_name} to have a dtype attribute, but got {type(value).__name__}.")
    if value.dtype != expected_dtype:
        raise TypeError(f"Expected {var_name} to have dtype {expected_dtype}, but got {value.dtype}.")
    