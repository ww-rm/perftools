"""Apk repakcer, can add debuggable flag to apk"""

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union
from xml.dom import minidom

from . import PACKAGE_BIN_DIR, PACKAGE_DATA_DIR
from .androidsdk import Sdk
from .executable import ApkSigner, ApkTool, ZipAlign
from .logger import logger

StrPath = Union[str, os.PathLike]


class AndroidManifest:
    """AndroidManifest"""

    FILENAME = "AndroidManifest.xml"

    def __init__(self, path: StrPath) -> None:
        self._tree = minidom.parse(str(path))

    def enable_debuggable(self, flag: bool = True):
        application_ele = self._tree.getElementsByTagName("application")[0]
        application_ele.setAttribute("android:debuggable", ("true" if flag else "false"))

    def enable_profileable(self, flag: bool = True):
        application_ele = self._tree.getElementsByTagName("application")[0]
        profileable_ele = application_ele.getElementsByTagName("profileable")
        if len(profileable_ele) <= 0:
            profileable_ele = self._tree.createElement("profileable")
            profileable_ele.setAttribute("android:shell", ("true" if flag else "false"))
            application_ele.appendChild(profileable_ele)
        else:
            profileable_ele = profileable_ele[0]
            profileable_ele.setAttribute("android:shell", ("true" if flag else "false"))

    def save(self, path: StrPath):
        path = Path(path)
        path.write_bytes(self._tree.toxml(encoding="utf-8", standalone=False))
        return path


class ApkRepacker:
    """Apk Repacker"""

    DEFAULT_KS_FILENAME = "repack_default.keystore"
    DEFAULT_KS_PASSWORD = b"123456"

    def __init__(self, sdk_dir: Optional[StrPath] = None) -> None:
        buildtool = Sdk(sdk_dir).get_latest_buildtool()

        self._apksigner = ApkSigner(buildtool.apksigner)
        self._apktool = ApkTool(PACKAGE_BIN_DIR.joinpath("apktool.jar"))
        self._zipalign = ZipAlign(buildtool.zipalign)

        self._ks_file = PACKAGE_DATA_DIR.joinpath(self.DEFAULT_KS_FILENAME)
        if not self._ks_file.is_file():
            raise FileNotFoundError(f"default keystore file {self.DEFAULT_KS_FILENAME} not found in {PACKAGE_DATA_DIR}")

    def print_tools(self):
        """Print tool paths"""

        logger.info("Using: %s", self._apktool.filepath)
        logger.info("Using: %s", self._apksigner.filepath)
        logger.info("Using: %s", self._zipalign.filepath)

    def pack(self, apk_dir: StrPath, output_path: StrPath) -> bool:
        return self._apktool.pack(apk_dir, output_path).returncode == 0

    def unpack(self, apk_path: StrPath, output_dir: StrPath) -> bool:
        return self._apktool.unpack(apk_path, output_dir).returncode == 0

    def align(self, apk_path: StrPath, output_path: StrPath) -> bool:
        return self._zipalign.align(apk_path, output_path).returncode == 0

    def sign(self, apk_path: StrPath, output_path: StrPath) -> bool:
        return self._apksigner.sign(apk_path, output_path, self._ks_file, self.DEFAULT_KS_PASSWORD).returncode == 0

    def add_debuggable_flag(self, apk_dir: StrPath):
        """Make apk debuggable."""

        manifest_path = Path(apk_dir).joinpath(AndroidManifest.FILENAME)
        manifest = AndroidManifest(manifest_path)
        manifest.enable_debuggable(True)
        manifest.save(manifest_path)

    def add_profileable_flag(self, apk_dir: StrPath):
        """Make apk profileable."""

        manifest_path = Path(apk_dir).joinpath(AndroidManifest.FILENAME)
        manifest = AndroidManifest(manifest_path)
        manifest.enable_profileable(True)
        manifest.save(manifest_path)


def do_repack(
    apk_path: StrPath,
    output_path: StrPath,
    enable_debuggable: bool = True,
    enable_profileable: bool = False,
    sdk_dir: Optional[StrPath] = None,
    tmp_dir: StrPath = "tmp"
):
    """Do repack

    Args:
        apk_path: Path apk to be repacked.
        output_path: Output path of repacked apk.
        enable_debuggable: Whether make apk debuggable.
        enable_profileable: Whether make apk profileable.
        sdk_dir: Directory of android sdk to be used, if not specified, search sdk in some possible places.
        tmp_dir: Directory to save temporary files.

    Raises:
        RuntimeError: If any step failed.
    """

    tmp = Path(tmp_dir).resolve()
    tmp.mkdir(parents=True, exist_ok=True)

    repacker = ApkRepacker(sdk_dir)
    repacker.print_tools()

    # apk paths and dirs
    apk_path = Path(apk_path)
    apk_name = apk_path.stem
    apk_dir = tmp.joinpath(apk_name)
    apk_repack_path = tmp.joinpath(f"{apk_name}_repack.apk")
    apk_align_path = tmp.joinpath(f"{apk_name}_align.apk")
    apk_output_path = Path(output_path)

    # unpack apk to dir
    if not repacker.unpack(apk_path, apk_dir):
        raise RuntimeError("unpack failed")

    # do some modifications
    if enable_debuggable:
        repacker.add_debuggable_flag(apk_dir)
    if enable_profileable:
        repacker.add_profileable_flag(apk_dir)

    # pack and resign
    if not repacker.pack(apk_dir, apk_repack_path):
        raise RuntimeError("pack failed")
    if not repacker.align(apk_repack_path, apk_align_path):
        raise RuntimeError("align failed")
    if not repacker.sign(apk_align_path, apk_output_path):
        raise RuntimeError("sign failed")


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--apk", type=str, help="path of input apk", required=True)
    parser.add_argument("-o", "--output", type=str, default="repacked.apk", help="output path of repacked apk file")

    parser.add_argument("--enable-debuggable", action="store_true", help="add debuggable in AndroidManifest.xml")
    parser.add_argument("--enable-profileable", action="store_true", help="add profileable in AndroidManifest.xml")

    args = parser.parse_args()

    do_repack(args.apk, args.output, args.enable_debuggable, args.enable_profileable)


if __name__ == "__main__":
    main()
