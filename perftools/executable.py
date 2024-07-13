import os
import subprocess as sp
from functools import wraps
from pathlib import Path
from typing import List, Optional, Union

from . import PACKAGE_BIN_DIR
from .logger import logger

StrPath = Union[str, os.PathLike]


def search_in_path(filename: str, search_dirs: Optional[List[StrPath]] = None) -> Path:
    if search_dirs is None:
        search_dirs = os.get_exec_path()
    search_dirs.append(PACKAGE_BIN_DIR)

    for p in map(Path, search_dirs):
        filepath = p.joinpath(filename)
        if filepath.is_file():
            return filepath.resolve()

    raise FileNotFoundError(f"{filename} not found")


class BaseExecutable:

    def __init__(self, path: Optional[StrPath] = None) -> None:
        if path is None:
            path = search_in_path(self.filename)
        self.filepath = Path(path).resolve()

    @property
    def filename(self) -> str:
        raise NotImplementedError

    def _exec_blocking(self, *args, input: Optional[bytes] = None, timeout: Optional[float] = None):
        args = [str(self.filepath), *map(str, args)]
        logger.info("Exec: %s", repr(" ".join(args)))
        p = sp.run(args, input=input, capture_output=True, timeout=timeout)

        if p.stdout:
            logger.info("Exec stdout: %s", p.stdout.decode())

        if p.stderr:
            logger.error("Exec stderr: %s", p.stderr.decode())

        return p

    def _exec_nonblocking(self, *args):
        args = [str(self.filepath), *map(str, args)]
        logger.info("Exec: %s", repr(" ".join(args)))
        return sp.Popen(args, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

    def exec(self, *args, blocking: bool = True, input: Optional[bytes] = None, timeout: Optional[float] = None):
        if blocking:
            return self._exec_blocking(*args, input=input, timeout=timeout)
        else:
            return self._exec_nonblocking(*args)


class Adb(BaseExecutable):

    @property
    def filename(self):
        return "adb.exe"

    def exec(self, serial: str, *args, blocking: bool = True, input: Optional[bytes] = None, timeout: Optional[float] = None):
        return super().exec("-s", serial, *args, blocking=blocking, input=input, timeout=timeout)

    def push(self, serial: str, local_path: StrPath, device_path: str) -> sp.CompletedProcess[bytes]:
        return self.exec(serial, "push", local_path, device_path)

    def pull(self, serial: str, device_path: str, local_path: StrPath) -> sp.CompletedProcess[bytes]:
        return self.exec(serial, "pull", device_path, local_path)

    def shell(self, serial: str, *args, blocking: bool = True, input: Optional[bytes] = None, timeout: Optional[float] = None):
        return self.exec(serial, "shell", *args, blocking=blocking, input=input, timeout=timeout)


class ZipAlign(BaseExecutable):
    @property
    def filename(self):
        return "zipalign.exe"

    def align(self, apk_path: StrPath, output_path: StrPath) -> sp.CompletedProcess[bytes]:
        # MUST use absolute path
        apk_path = Path(apk_path).resolve()
        output_path = Path(output_path).resolve()
        return self.exec("-f", "-p", "4", apk_path, output_path)


class Java(BaseExecutable):
    @property
    def filename(self):
        return "java.exe"


class BaseJar:

    def __init__(self, path: StrPath, java_path: Optional[StrPath] = None) -> None:
        self._path = Path(path).resolve()
        self._java = Java(java_path)

    @property
    def filepath(self) -> Path:
        return self._path

    def exec(self, *args, blocking: bool = True, input: Optional[bytes] = None, timeout: Optional[float] = None):
        return self._java.exec("-jar", self._path, *args, blocking=blocking, input=input, timeout=timeout)


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
        return self.exec("sign", "--ks", ks_file, "--out", output_path, apk_path, input=ks_pwd)
