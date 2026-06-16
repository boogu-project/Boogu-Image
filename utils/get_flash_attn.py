"""
Install flash-attn using mjun0812 prebuild wheels, fall back to Dao-AILab,
then source compilation.

Usage:
    python get_flash_attn.py
    python get_flash_attn.py --flash-attn-version 2.8.3
    python get_flash_attn.py --dry-run
    python get_flash_attn.py --build
"""

import argparse
import json
import os
import platform as platform_module
import re
import subprocess
import sys
import sysconfig
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FLASH_ATTN_VERSION = "2.8.3"
DEFAULT_MJUN_ASSETS_JSON = "mjun_flash_attn_assets.json"

# Ported from https://mjunya.com/flash-attention-prebuild-wheels/.
# It supports flash_attn/flash_attn_3, cpXXX/cpXXXt/abi3 tags, optional git
# suffixes, and platform tags including manylinux variants.
WHEEL_PATTERN = re.compile(
    r"flash_attn(?:_3)?-"
    r"(\d+\.\d+\.\d+(?:\.[a-z0-9]+)?)"
    r"\+cu(\d+)torch(\d+\.\d+)"
    r"(?:git([0-9a-f]+))?"
    r"-cp(\d+)-(?:cp\d+(t?)|abi3)-(.+?)\.whl"
)


@dataclass(frozen=True)
class RuntimeEnv:
    torch_major: str
    torch_minor: str
    cuda_full: str
    cuda_major: str
    cxx11_abi: str
    python_tag: str

    @property
    def torch_version(self):
        return f"{self.torch_major}.{self.torch_minor}"

    @property
    def cuda_version(self):
        return f"{self.cuda_full[:-1]}.{self.cuda_full[-1]}"


def detect_runtime_env():
    import torch

    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return RuntimeEnv(
        torch_major=m.group(1),
        torch_minor=m.group(2),
        cuda_full=torch.version.cuda.replace(".", ""),
        cuda_major=torch.version.cuda.split(".")[0],
        cxx11_abi="TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE",
        python_tag=f"cp{sys.version_info.major}{sys.version_info.minor}",
    )


def normalize_platform_name(raw):
    """Normalize platform tags the same way as the mjunya wheel browser."""
    if raw.startswith("manylinux") and "." in raw:
        raw = raw.split(".")[0]

    if raw.startswith("manylinux"):
        parts = raw.split("_")
        if len(parts) >= 4:
            arch = "_".join(parts[3:])
            if arch == "aarch64":
                arch = "arm64"
            return f"Linux {arch}"

    name = raw[:1].upper() + raw[1:]
    name = name.replace("_", " ", 1)
    name = name.replace("Win", "Windows")
    name = name.replace("amd64", "x86_64")
    name = name.replace("aarch64", "arm64")
    return name


def current_platform_name():
    """Return the normalized platform name used for matching mjun0812 wheels."""
    machine = platform_module.machine().lower()
    if machine in {"amd64", "x86_64"}:
        arch = "x86_64"
    elif machine in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        arch = machine

    if sys.platform.startswith("linux"):
        return f"Linux {arch}"
    if sys.platform == "darwin":
        return f"Macosx {arch}"
    if sys.platform.startswith("win"):
        return f"Windows {arch}"

    return normalize_platform_name(sysconfig.get_platform().replace("-", "_"))


def parse_python_tag(python_raw, free_threaded):
    if free_threaded is None:
        return f"{python_raw[0]}.{python_raw[1:]}+ (abi3)"
    return f"{python_raw[0]}.{python_raw[1:]}{free_threaded}"


def python_tag_matches(python_raw, free_threaded, py):
    """Match exact CPython wheels and compatible abi3 wheels."""
    current = int(py[2:])
    wheel_py = int(python_raw)
    if free_threaded is None:
        return wheel_py <= current
    return wheel_py == current


def parse_version(version):
    return [int(part) for part in re.findall(r"\d+", version)]


def parse_wheel_filename(filename):
    match = WHEEL_PATTERN.match(filename)
    if not match:
        return None

    (
        flash_version,
        cuda_raw,
        torch_version,
        git_hash,
        python_raw,
        free_threaded,
        platform_raw,
    ) = match.groups()
    package_name = (
        "flash_attn_3" if filename.startswith("flash_attn_3-") else "flash_attn"
    )
    cuda_version = f"{cuda_raw[:-1]}.{cuda_raw[-1]}"
    return {
        "package_name": package_name,
        "flash_version": flash_version,
        "cuda_raw": cuda_raw,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "git_hash": git_hash,
        "python_raw": python_raw,
        "python_version": parse_python_tag(python_raw, free_threaded),
        "free_threaded": free_threaded,
        "platform": normalize_platform_name(platform_raw),
        "platform_raw": platform_raw,
        "filename": filename,
    }


def resolve_default_asset_path(path):
    path = Path(path)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return Path(__file__).with_name(path.name)


def load_mjun_asset_data(path):
    """Load pre-fetched mjun0812 asset data from the repository."""
    path = Path(path)
    try:
        with path.open() as f:
            payload = json.load(f)
    except FileNotFoundError:
        print(f"  mjun asset JSON not found: {path}")
        return []
    except Exception as e:
        print(f"  Failed to load mjun asset JSON: {e}")
        return []

    if isinstance(payload, dict):
        if isinstance(payload.get("assets"), list):
            return payload["assets"]
        if isinstance(payload.get("releases"), list):
            return payload["releases"]
    if isinstance(payload, list):
        return payload

    print(f"  Unsupported mjun asset JSON format: {path}")
    return []


def iter_mjun_assets(items):
    """Yield asset dicts from either a flat asset list or GitHub release list."""
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("assets"), list):
            for asset in item["assets"]:
                if not isinstance(asset, dict):
                    continue
                yield {
                    **asset,
                    "tag": item.get("tag_name") or item.get("tag"),
                    "release_url": item.get("html_url") or item.get("release_url"),
                }
        else:
            yield item


def extract_mjun_wheels(items):
    wheels = []
    for asset in iter_mjun_assets(items):
        name = asset.get("name", "")
        if not name.endswith(".whl"):
            continue
        parsed = parse_wheel_filename(name)
        if not parsed:
            continue
        parsed.update(
            {
                "url": asset.get("browser_download_url") or asset.get("url"),
                "tag": asset.get("tag") or tag_from_github_url(asset.get("url", "")),
                "release_url": asset.get("release_url"),
            }
        )
        wheels.append(parsed)
    return wheels


def tag_from_github_url(url):
    match = re.search(r"/releases/download/([^/]+)/", url)
    return match.group(1) if match else None


def deduplicate_wheels(wheels):
    """Keep the first wheel per website key; JSON is expected newest-first."""
    seen = {}
    for wheel in wheels:
        key = (
            wheel["package_name"],
            wheel["flash_version"],
            wheel["python_version"],
            wheel["torch_version"],
            wheel["cuda_version"],
            wheel["platform"],
        )
        if key not in seen:
            seen[key] = wheel
    return list(seen.values())


def search_mjun(asset_json, flash_attn_version, env):
    """Search mjun0812 releases using the same parse/filter model as the site."""
    asset_data = load_mjun_asset_data(asset_json)
    if not asset_data:
        return None

    platform = current_platform_name()

    print(
        "Searching for parsed wheel fields: "
        f"flash_attn {flash_attn_version}, torch {env.torch_version}, "
        f"CUDA {env.cuda_version}, {env.python_tag}, {platform}"
    )

    candidates = []
    for wheel in deduplicate_wheels(extract_mjun_wheels(asset_data)):
        if wheel["package_name"] != "flash_attn":
            continue
        if wheel["flash_version"] != flash_attn_version:
            continue
        if wheel["torch_version"] != env.torch_version:
            continue
        if wheel["cuda_raw"] != env.cuda_full:
            continue
        if not python_tag_matches(
            wheel["python_raw"], wheel["free_threaded"], env.python_tag
        ):
            continue
        if wheel["platform"] != platform:
            continue
        candidates.append(wheel)

    if not candidates:
        return None

    candidates.sort(
        key=lambda wheel: (
            wheel["free_threaded"] is not None,
            parse_version(wheel["flash_version"]),
            wheel["filename"],
        ),
        reverse=True,
    )
    wheel = candidates[0]
    print(f"Matched mjun0812 wheel: {wheel['filename']} ({wheel['tag']})")
    return wheel["url"], wheel["filename"]


def download_and_install(url, name):
    dest = f"/tmp/{name}"
    print(f"Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        subprocess.check_call([sys.executable, "-m", "pip", "install", dest])
        os.remove(dest)
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def pip_install_command(target):
    return f"{sys.executable} -m pip install {target}"


def source_build_command():
    return f"FLASH_ATTENTION_FORCE_BUILD=TRUE {sys.executable} -m pip install flash-attn --no-build-isolation"


def print_install_command(command):
    print("\nInstall command:")
    print(command)


def daoai_candidates(flash_attn_version, env):
    daoai_base = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{flash_attn_version}"
    for abi in [env.cxx11_abi, "FALSE" if env.cxx11_abi == "TRUE" else "TRUE"]:
        name = (
            f"flash_attn-{flash_attn_version}"
            f"+cu{env.cuda_major}torch{env.torch_version}cxx11abi{abi}"
            f"-{env.python_tag}-{env.python_tag}-linux_x86_64.whl"
        )
        yield name, f"{daoai_base}/{name}"


def try_daoai(flash_attn_version, env):
    """Try Dao-AILab official releases (both ABI variants)."""
    for name, url in daoai_candidates(flash_attn_version, env):
        print(f"Trying Dao-AILab: {name}")
        try:
            urllib.request.urlretrieve(url, f"/tmp/{name}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", f"/tmp/{name}"]
            )
            os.remove(f"/tmp/{name}")
            return True
        except Exception as e:
            print(f"  Failed: {e}")
    return False


def install_from_source():
    print("Falling back to source compilation (30-60 min)...")
    env = {**os.environ, "FLASH_ATTENTION_FORCE_BUILD": "TRUE"}
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"],
        env=env,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Install flash-attn from prebuilt wheels when possible."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Skip prebuilt wheels and build flash-attn from source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected install command without running it.",
    )
    parser.add_argument(
        "--flash-attn-version",
        default=DEFAULT_FLASH_ATTN_VERSION,
        help=f"flash-attn version to install from mjun0812 or Dao-AILab (default: {DEFAULT_FLASH_ATTN_VERSION}).",
    )
    parser.add_argument(
        "--mjun-assets-json",
        default=resolve_default_asset_path(DEFAULT_MJUN_ASSETS_JSON),
        help="Path to the pre-fetched mjun0812 release asset JSON.",
    )
    args = parser.parse_args()

    if args.build:
        if args.dry_run:
            print_install_command(source_build_command())
            return
        install_from_source()
        return

    env = detect_runtime_env()
    print(
        f"Environment: torch={env.torch_version}, cu{env.cuda_full}, "
        f"{env.python_tag}, cxx11abi={env.cxx11_abi}"
    )

    # 1. Try mjun0812 prebuild wheels
    print("\n[1/3] Searching mjun0812 prebuild wheels...")
    result = search_mjun(
        args.mjun_assets_json,
        args.flash_attn_version,
        env,
    )
    if result:
        url, name = result
        if args.dry_run:
            print_install_command(pip_install_command(url))
            return
        if download_and_install(url, name):
            return
    else:
        print("  No matching wheel found in mjun0812.")

    # 2. Try Dao-AILab official releases
    print("\n[2/3] Trying Dao-AILab official releases...")
    if args.dry_run:
        name, url = next(daoai_candidates(args.flash_attn_version, env))
        print(f"Selected Dao-AILab candidate: {name}")
        print_install_command(pip_install_command(url))
        return
    if try_daoai(args.flash_attn_version, env):
        return

    # 3. Source compilation
    print("\n[3/3] No prebuilt wheel found.")
    if args.dry_run:
        print_install_command(source_build_command())
        return
    install_from_source()


if __name__ == "__main__":
    main()
