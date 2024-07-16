import sys as _sys
from pathlib import Path

PACKAGE_DATA_DIR = Path(__file__).parent.joinpath("data")
PACKAGE_BIN_DIR = Path(__file__).parent.joinpath("bin")

__version__ = "0.1.0"

from .androidsdk import Sdk as _Sdk  # noqa: E402

# add simpleperf module path to sys.path
_simpleperf_dir = _Sdk().get_latest_ndk().simpleperf
if not _simpleperf_dir.is_dir():
    raise FileNotFoundError(f"simpleperf {_simpleperf_dir} not found")
_sys.path.append(str(_simpleperf_dir))
