"""Android sdk directories finder module."""

import os
import sys
from pathlib import Path
from typing import Optional, Union

from . import PACKAGE_BIN_DIR, executable

StrPath = Union[str, os.PathLike]

# some possible dirs for android sdk
if sys.platform.startswith("win32"):
    POSSIBLE_SDK_DIRS = [
        Path(os.getenv("LOCALAPPDATA"), "Android/Sdk"),
        Path("C:/Android/Sdk"),
        Path("D:/Android/Sdk"),
        Path("E:/Android/Sdk"),
    ]
else:
    POSSIBLE_SDK_DIRS = [
        Path(os.getenv("HOME"), "Android/Sdk"),
    ]

POSSIBLE_SDK_DIRS.append(PACKAGE_BIN_DIR.joinpath("AndroidSdk"))

if os.getenv("ANDROID_HOME"):
    POSSIBLE_SDK_DIRS.insert(0, Path(os.getenv("ANDROID_HOME")))


class BaseToolDirectory:
    def __init__(self, directory: StrPath, version: Optional[str] = None) -> None:
        self.dir = Path(directory).resolve()
        self.version = version


class BuildTool(BaseToolDirectory):
    """Sdk build tool"""

    def __init__(self, directory: StrPath, version: Optional[str] = None) -> None:
        super().__init__(directory, version)

        self.apksigner = self.dir.joinpath("lib", "apksigner.jar")
        self.zipalign = self.dir.joinpath(executable.ZipAlign.filename)


class PlatformTools(BaseToolDirectory):
    """Sdk platform tools"""

    def __init__(self, directory: StrPath, version: Optional[str] = None) -> None:
        super().__init__(directory, version)

        self.adb = self.dir.joinpath(executable.Adb.filename)


class Ndk(BaseToolDirectory):
    """Sdk ndk"""

    def __init__(self, directory: StrPath, version: Optional[str] = None) -> None:
        super().__init__(directory, version)

        self.simpleperf = self.dir.joinpath("simpleperf")


class Sdk:
    """Android Sdk"""

    def __init__(self, directory: Optional[StrPath] = None) -> None:
        """Android Sdk

        Args:
            directory: Sdk directory. if not specified, find sdk in some possible places, like ANDROID_HOME, or default install directory

        Raises:
            FileNotFoundError: Sdk can't be found in possible places.
        """

        if directory is None:
            for d in POSSIBLE_SDK_DIRS:
                if d.is_dir():
                    directory = d
                    break
            else:
                raise FileNotFoundError(f"No valid Sdk found, try specify Sdk directory, {POSSIBLE_SDK_DIRS}")

        self.dir = Path(directory).resolve()
        self.platform_tools = PlatformTools(self.dir.joinpath("platform-tools"))

        self._build_tools = [BuildTool(d, d.name) for d in self.dir.joinpath("build-tools").iterdir()]
        self._ndk = [Ndk(d, d.name) for d in self.dir.joinpath("ndk").iterdir()]

        self._build_tools.sort(key=lambda v: v.version)
        self._ndk.sort(key=lambda v: v.version)

    def get_latest_buildtool(self) -> BuildTool:
        if len(self._build_tools) <= 0:
            raise FileNotFoundError(f"build-tools not found in sdk {self.dir}")
        return self._build_tools[-1]

    def get_latest_ndk(self) -> Ndk:
        if len(self._ndk) <= 0:
            raise FileNotFoundError(f"ndk not found in sdk {self.dir}")
        return self._ndk[-1]
