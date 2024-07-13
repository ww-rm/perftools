
import os
import signal
import subprocess as sp
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from . import executable
from .logger import logger

StrPath = Union[str, os.PathLike]


class RecordProc:

    def __init__(self, proc: sp.Popen[bytes], package_name: str, device_perfdata_path: str) -> None:
        self.proc = proc
        self.package_name = package_name
        self.device_perfdata_path = device_perfdata_path
        self.stdout: Optional[bytes] = None
        self.stderr: Optional[bytes] = None

    @property
    def is_running(self) -> bool:
        return self.proc.returncode is None

    @property
    def is_success(self) -> bool:
        return self.proc.returncode == 0

    def stop(self, timeout: Optional[float] = None) -> bool:
        if self.proc.returncode is not None:
            return True

        # signal should be SIGINT/SIGTERM/SIGHUP, can't be SIGKILL
        self.proc.terminate()
        self.proc.wait(timeout)

        if self.proc.returncode is not None:
            self.stdout = self.proc.stdout.read()
            self.stderr = self.proc.stderr.read()

            if self.stdout:
                logger.info("record stdout: %s", self.stdout.decode())
            if self.stderr:
                logger.error("record stderr: %s", self.stderr.decode())

        return self.proc.returncode is not None


class Simpleperf:
    DEVICE_BIN_PATH = "/data/local/tmp/simpleperf"

    def __init__(self, directory: StrPath, adb: Optional[executable.Adb] = None) -> None:
        self.dir = Path(directory).resolve()
        if adb is None:
            adb = executable.Adb()
        self._adb = adb

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

    def record(self, serial: str, package_name: str, freq: int = 4000, duration: int = 0) -> RecordProc:

        perfdata_filename = f"perf-{datetime.now().strftime('%Y%m%d-%H%M%S')}.data"
        device_perfdata_path = f"/data/local/tmp/{perfdata_filename}"

        self.push_to_device(serial)

        args = [
            self.DEVICE_BIN_PATH, "record",
            "--app", package_name,
            "-o", device_perfdata_path,
            "-e", "cpu-clock",
            "-f", freq,
            "-g",
            "--trace-offcpu",
            "--post-unwind=yes"
        ]

        if duration > 0:
            args += ["--duration", duration]

        proc = self._adb.shell(serial, *args, blocking=False)

        return RecordProc(proc, package_name, device_perfdata_path)
