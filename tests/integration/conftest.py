import pytest
import warnings

@pytest.fixture(autouse=True)
def ignore_macos_accelerate_warnings():
    """
    Automatically suppress known spurious warnings from Apple Accelerate framework 
    (NumPy on macOS M-series chips) for all tests.
    """
    for msg in [".*divide by zero encountered in matmul.*", 
                ".*overflow encountered in matmul.*", 
                ".*invalid value encountered in matmul.*"]:
        warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)