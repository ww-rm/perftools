"""Simpleperf android tool wrapper module."""

import os
import subprocess as sp
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union

from . import executable
from .androidsdk import Sdk
from .logger import logger

StrPath = Union[str, os.PathLike]


class Simpleperf:
    """Simpleperf used to execute commands in android device.

    Attributes:
        DEVICE_BIN_PATH (str): File path of simpleperf to be pushed on device.
        DEVICE_PERFDATA_PATH (str): Output path of simpleperf record command runs on device.

        stdout (Optional[bytes]): stdout of command process.
        stderr (Optional[bytes]): stderr of command process.
    """

    DEVICE_BIN_PATH = "/data/local/tmp/simpleperf"
    DEVICE_PERFDATA_PATH = "/data/local/tmp/perf.data"

    def __init__(self, directory: Optional[StrPath] = None, adb: Optional[executable.Adb] = None) -> None:
        """Simpleperf

        Args:
            directory: Simpleperf directory, maybe like Android/Sdk/ndk/<version>/simpleperf, if not specified, search in possible places.
            adb: Adb helper, if not specified, search in possible places and creat it.
        """

        sdk = Sdk()  # get a default sdk

        if directory is None:
            directory = sdk.get_latest_ndk().simpleperf
        self.dir = Path(directory).resolve()

        if adb is None:
            try:
                adb = executable.Adb()
            except FileNotFoundError:
                adb = executable.Adb(sdk.platform_tools.adb)

        self._adb = adb
        self._proc: Optional[sp.Popen[bytes]] = None
        self.stdout: Optional[bytes] = None
        self.stderr: Optional[bytes] = None

    def print_tools(self):
        logger.info("Using %s", self._adb.filepath)
        logger.info("Using %s", self.dir)

    def is_running(self, serial: str) -> bool:
        p = self._adb.shell(serial, "pidof", "simpleperf")
        if p.returncode == 0 and p.stdout:
            return True
        return False

    def stop_all(self, serial: str):
        """Stop all simpleperf processes on device."""

        # SIGINT = 2
        has_killed = False
        while self.is_running(serial):
            if not has_killed:
                self._adb.shell(serial, "pkill", "-l", "2", "simpleperf")
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

    def _get_binary_path_for_device(self, serial: str) -> Path:
        """Get proper simpleperf binary filepath to be pushed to device."""

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
        """Push proper simpleperf binary file to device."""

        local_bin_path = self._get_binary_path_for_device(serial)
        if self._adb.push(serial, local_bin_path, self.DEVICE_BIN_PATH).returncode != 0:
            raise RuntimeError(f"push simpleperf to device {serial} failed")
        if self._adb.shell(serial, "chmod", "a+x", self.DEVICE_BIN_PATH).returncode != 0:
            raise RuntimeError(f"chmod faild on device {serial}")

    def begin_record(self, serial: str, package_name: str, freq: int = 4000, duration: int = 0) -> bool:
        """Execute simpleperf record command and begin record.

        Args:
            serial: serial of device.
            package_name: Package name of application to be recorded.
            freq: Frequency of record.
            duration: Duration to be recorded, if less than zero, will record infinitly until stop_all is called.
        """

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
            "--post-unwind=yes",
            "--no-cut-samples",
        ]

        if duration > 0:
            args += ["--duration", duration]

        self._proc = self._adb.shell(serial, *args, blocking=False)
        return True

    def pull_perfdata(self, serial: str, local_perfdata_path: StrPath) -> bool:
        """Pull record file from device."""

        return self._adb.pull(serial, self.DEVICE_PERFDATA_PATH, local_perfdata_path).returncode == 0


def do_record(serial: str, package_name: str, freq: int = 4000, duration: int = 0, output_path: StrPath = "perf.data"):
    """Do record, can use Ctr-C to stop it early."""

    simpleperf = Simpleperf()
    simpleperf.begin_record(serial, package_name, freq, duration)
    try:
        while simpleperf.is_running(serial):
            time.sleep(1)
    except KeyboardInterrupt:
        simpleperf.stop_all(serial)
    simpleperf.pull_perfdata(serial, output_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--serial", help="serial of device", required=True)
    parser.add_argument("-p", "--app", help="package name of app to be profiled", required=True)
    parser.add_argument("-o", "--output", default="perf.data", help="output path of perf data collected in device")
    parser.add_argument("-f", "--freq", type=int, default=4000, help="frequency of simpleperf record")
    parser.add_argument("-d", "--duration", type=int, default=0, help="duration of simpleperf record")

    args = parser.parse_args()

    do_record(args.serial, args.app, args.freq, args.duration, args.output)


if __name__ == "__main__":
    main()
