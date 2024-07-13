import os
from pathlib import Path
from typing import Optional, Union

from . import PACKAGE_BIN_DIR

StrPath = Union[str, os.PathLike]

POSSIBLE_SDK_DIRS = [
    Path(os.getenv("LOCALAPPDATA"), "Android/Sdk"),
    Path("C:/Android/Sdk"),
    Path("D:/Android/Sdk"),
    Path("E:/Android/Sdk"),
    PACKAGE_BIN_DIR.joinpath("AndroidSdk")
]

if os.getenv("ANDROID_HOME"):
    POSSIBLE_SDK_DIRS.insert(0, Path(os.getenv("ANDROID_HOME")))


class BaseToolDirectory:
    def __init__(self, directory: StrPath, version: Optional[str] = None) -> None:
        self.dir = Path(directory).resolve()
        self.version = version


class BuildTool(BaseToolDirectory):
    def __init__(self, directory: StrPath, version: Optional[str] = None) -> None:
        super().__init__(directory, version)

        self.apksigner = self.dir.joinpath("lib", "apksigner.jar")
        self.zipalign = self.dir.joinpath("zipalign.exe")


class PlatformTools(BaseToolDirectory):
    def __init__(self, directory: StrPath, version: Optional[str] = None) -> None:
        super().__init__(directory, version)

        self.adb = self.dir.joinpath("adb.exe")


class Ndk(BaseToolDirectory):
    def __init__(self, directory: StrPath, version: Optional[str] = None) -> None:
        super().__init__(directory, version)

        self.simpleperf = self.dir.joinpath("simpleperf")


class Sdk:
    def __init__(self, directory: Optional[StrPath] = None) -> None:
        if directory is None:
            for d in POSSIBLE_SDK_DIRS:
                if d.is_dir():
                    directory = d
                    break
            else:
                raise FileNotFoundError(f"No valid Sdk found, try specify Sdk directory, {POSSIBLE_SDK_DIRS}")

        self.dir = Path(directory).resolve()
        self._build_tools = [BuildTool(d, d.name) for d in self.dir.joinpath("build-tools").iterdir()]
        self._ndk = [Ndk(d, d.name) for d in self.dir.joinpath("ndk").iterdir()]
        self._platform_tools = PlatformTools(self.dir.joinpath("platform-tools"))

        self._build_tools.sort(key=lambda v: v.version)
        self._ndk.sort(key=lambda v: v.version)

    def get_latest_buildtool(self) -> Optional[BuildTool]:
        if len(self._build_tools) > 0:
            return self._build_tools[-1]
        return None

    def get_latest_ndk(self) -> Optional[Ndk]:
        if len(self._ndk) > 0:
            return self._ndk[-1]
        return None

    def get_platformtools(self) -> PlatformTools:
        return self._platform_tools
