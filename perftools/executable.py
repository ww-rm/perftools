"""Executable files helper module."""

import os
import subprocess as sp
from pathlib import Path
from typing import List, Optional, Union
import sys

from . import PACKAGE_BIN_DIR
from .logger import logger

StrPath = Union[str, os.PathLike]


def search_in_path(filename: str, search_dirs: Optional[List[StrPath]] = None) -> Path:
    """Search a file in specified directories or system PATH.

    Args:
        filename: Filename to be searched.
        search_dirs: Directories to be searched, if not specified, search in system PATH

    Raises:
        FileNotFoundError: File not found.
    """

    if search_dirs is None:
        search_dirs = os.get_exec_path()
    search_dirs.append(PACKAGE_BIN_DIR)

    for p in map(Path, search_dirs):
        filepath = p.joinpath(filename)
        if filepath.is_file():
            return filepath.resolve()

    raise FileNotFoundError(f"{filename} not found")


class BaseExecutable:
    """Executable file commands wrapper.

    Attributes:
        filename (str): Filename of executable file, maybe different in different platforms.
    """

    filename: str

    def __init__(self, path: Optional[StrPath] = None) -> None:
        """Executable file.

        Args:
            path: File path of executable file. If not specified, search it in system PATH
        """

        if path is None:
            path = search_in_path(self.filename)
        self.filepath = Path(path).resolve()

    def _exec_blocking(self, *args, input: Optional[bytes] = None, timeout: Optional[float] = None) -> sp.CompletedProcess[bytes]:
        """Execute file in blocking mode, return until end or timeout."""

        args = [str(self.filepath), *map(str, args)]
        logger.info("Exec: %s", repr(" ".join(args)))
        p = sp.run(args, input=input, capture_output=True, timeout=timeout)

        if p.stdout:
            logger.info("Exec stdout: %s", p.stdout.decode())

        if p.stderr:
            logger.error("Exec stderr: %s", p.stderr.decode())

        return p

    def _exec_nonblocking(self, *args) -> sp.Popen[bytes]:
        """Execute file in non-blocking mode, return right away."""

        args = [str(self.filepath), *map(str, args)]
        logger.info("Exec: %s", repr(" ".join(args)))
        return sp.Popen(args, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

    def exec(self, *args, blocking: bool = True, input: Optional[bytes] = None, timeout: Optional[float] = None):
        """Execute file with arguments.

        If blocking is True, use input and timeout, and return a CompletedProcess,
            else omit input and timeout and return a Popen object
        """

        if blocking:
            return self._exec_blocking(*args, input=input, timeout=timeout)
        else:
            return self._exec_nonblocking(*args)


class Adb(BaseExecutable):

    if sys.platform.startswith("win32"):
        filename = "adb.exe"
    else:
        filename = "adb"

    def exec(self, serial: str, *args, blocking: bool = True, input: Optional[bytes] = None, timeout: Optional[float] = None):
        return super().exec("-s", serial, *args, blocking=blocking, input=input, timeout=timeout)

    def push(self, serial: str, local_path: StrPath, device_path: str) -> sp.CompletedProcess[bytes]:
        return self.exec(serial, "push", local_path, device_path)

    def pull(self, serial: str, device_path: str, local_path: StrPath) -> sp.CompletedProcess[bytes]:
        return self.exec(serial, "pull", device_path, local_path)

    def shell(self, serial: str, *args, blocking: bool = True, input: Optional[bytes] = None,  timeout: Optional[float] = None):
        return self.exec(serial, "shell", *args, blocking=blocking, input=input, timeout=timeout)


class ZipAlign(BaseExecutable):

    if sys.platform.startswith("win32"):
        filename = "zipalign.exe"
    else:
        filename = "zipalign"

    def align(self, apk_path: StrPath, output_path: StrPath) -> sp.CompletedProcess[bytes]:
        # MUST use absolute path
        apk_path = Path(apk_path).resolve()
        output_path = Path(output_path).resolve()
        return self.exec("-f", "-p", "4", apk_path, output_path)


class Java(BaseExecutable):

    if sys.platform.startswith("win32"):
        filename = "java.exe"
    else:
        filename = "java"


class BaseJar:
    """Executable jar file commands wrapper."""

    def __init__(self, path: StrPath, java_path: Optional[StrPath] = None) -> None:
        """BaseJar.

        Args:
            path: Jar path.
            java_path: Java executable file path, if not specified, search it in system PATH.
        """

        self.filepath = Path(path).resolve()
        self._java = Java(java_path)

    def exec(self, *args, blocking: bool = True, input: Optional[bytes] = None, timeout: Optional[float] = None):
        return self._java.exec("-jar", self.filepath, *args, blocking=blocking, input=input, timeout=timeout)


class ApkTool(BaseJar):

    def pack(self, apk_dir: StrPath, output_path: StrPath) -> sp.CompletedProcess[bytes]:
        return self.exec("-f", "b", apk_dir, "-o", output_path)

    def unpack(self, apk_path: StrPath, output_dir: StrPath) -> sp.CompletedProcess[bytes]:
        return self.exec("-f", "d", apk_path, "-o", output_dir)


class ApkSigner(BaseJar):
    def sign(
        self,
        apk_path: StrPath,
        output_path: StrPath,
        ks_file: StrPath,
        ks_pwd: bytes
    ) -> sp.CompletedProcess[bytes]:
        # MUST exist a "\n"
        return self.exec("sign", "--ks", ks_file, "--out", output_path, apk_path, input=ks_pwd + b"\n")
