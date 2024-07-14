
import os
import subprocess as sp
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from . import executable
from .logger import logger

StrPath = Union[str, os.PathLike]


class Simpleperf:
    DEVICE_BIN_PATH = "/data/local/tmp/simpleperf"
    DEVICE_PERFDATA_PATH = "/data/local/tmp/perf.data"

    def __init__(self, directory: StrPath, adb: Optional[executable.Adb] = None) -> None:
        self.dir = Path(directory).resolve()
        if adb is None:
            adb = executable.Adb()
        self._adb = adb
        self._proc: Optional[sp.Popen[bytes]] = None
        self.stdout: Optional[bytes] = None
        self.stderr: Optional[bytes] = None

    def is_running(self, serial: str) -> bool:
        p = self._adb.shell(serial, "pidof", "simpleperf")
        if p.returncode == 0 and p.stdout:
            return True
        return False

    def stop_all(self, serial: str):
        # SIGINT = 2
        has_killed = False
        while self.is_running(serial):
            if not has_killed:
                if self._adb.shell(serial, "pkill", "-l", "2", "simpleperf").returncode == 0:
                    has_killed = True
            time.sleep(1)

        if self._proc is not None and self._proc.returncode is None:
            self._proc.wait()

            self.stdout = self._proc.stdout.read()
            self.stderr = self._proc.stderr.read()

            if self.stdout:
                logger.info("simpleperf stdout: %s", self.stdout.decode())
            if self.stderr:
                logger.error("simpleperf stderr: %s", self.stderr.decode())

    def get_binary_path_for_device(self, serial: str) -> Path:
        output = self._adb.shell(serial, "uname", "-m").stdout.decode()
        arch = ""
        if "aarch64" in output:
            arch = "arm64"
        elif "arm" in output:
            arch = "arm"
        elif "x86_64" in output:
            arch = "x86_64"
        elif "86" in output:
            arch = "x86"
        elif "riscv64" in output:
            arch = "riscv64"
        else:
            raise TypeError("Unsupported architecture: %s", output)

        binary_path = self.dir.joinpath("bin", "android", arch, "simpleperf")
        if not binary_path.is_file():
            raise FileNotFoundError(binary_path)

        return binary_path

    def push_to_device(self, serial: str):
        local_bin_path = self.get_binary_path_for_device(serial)
        if self._adb.push(serial, local_bin_path, self.DEVICE_BIN_PATH).returncode != 0:
            raise RuntimeError(f"push simpleperf to device {serial} failed")
        if self._adb.shell(serial, "chmod", "a+x", self.DEVICE_BIN_PATH).returncode != 0:
            raise RuntimeError(f"chmod faild on device {serial}")

    def begin_record(self, serial: str, package_name: str, freq: int = 4000, duration: int = 0) -> bool:
        if self.is_running(serial):
            return False

        self.push_to_device(serial)

        args = [
            self.DEVICE_BIN_PATH, "record",
            "--app", package_name,
            "-o", self.DEVICE_PERFDATA_PATH,
            "-e", "cpu-clock",
            "-f", freq,
            "-g",
            "--trace-offcpu",
            "--post-unwind=yes"
        ]

        if duration > 0:
            args += ["--duration", duration]

        self._proc = self._adb.shell(serial, *args, blocking=False)
        return True

    def pull_perfdata(self, serial: str, local_perfdata_path: StrPath) -> bool:
        return self._adb.pull(serial, self.DEVICE_PERFDATA_PATH, local_perfdata_path).returncode == 0
