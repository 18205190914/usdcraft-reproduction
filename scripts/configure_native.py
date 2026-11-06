"""Prepare a portable native PhysX runtime on a Windows drive shared with WSL."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from articraft.usdcraft_physx import native_path

REPO = Path(__file__).resolve().parents[1]
NATIVE_SCRIPTS = (
    "apply_compatibility.py",
    "orthogonal_compound.py",
    "validate_asset.py",
    "launch_native.py",
)


def configure(
    windows_env: Path, output: Path, portable_root: Path, *, accept_eula: bool = False
) -> Path:
    windows_env = windows_env.resolve(strict=True)
    python = (windows_env / "Scripts/python.exe").resolve(strict=True)
    launcher = (windows_env / "Scripts/isaacsim.exe").resolve(strict=True)
    output = output.resolve()
    portable_root = portable_root.resolve()
    # Reject Linux-only paths before writing any configuration.
    folders = [
        native_path(windows_env / "Lib/site-packages/isaacsim" / p) for p in ("exts", "extscache")
    ]
    native_output, native_portable = native_path(output), native_path(portable_root)
    output.mkdir(parents=True, exist_ok=True)
    portable_root.mkdir(parents=True, exist_ok=True)
    runtime = REPO / "scripts/isaac"
    for name in NATIVE_SCRIPTS:
        shutil.copyfile(runtime / name, output / name)
    shutil.copyfile(REPO / "configs/usdcraft_physx.json", output / "profile.json")
    experience = runtime.joinpath("physx.kit").read_text()
    experience += "\n[settings.app.exts]\nfolders = " + json.dumps(folders) + "\n"
    output.joinpath("physx.kit").write_text(experience, encoding="utf-8")
    environment = {"PRIVACY_CONSENT": "N"}
    if accept_eula:
        environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    config = {
        "windows_python": str(python),
        "isaacsim": native_path(launcher),
        "experience": native_output + "/physx.kit",
        "portable_root": native_portable,
        "script_dir": str(output),
        "profile": str(output / "profile.json"),
        "timeout_seconds": 600,
        "environment": environment,
    }
    target = output / "native-config.json"
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument(
        "--accept-eula", action="store_true", help="Confirm you accept the NVIDIA Omniverse license"
    )
    args = parser.parse_args()
    print(
        configure(args.windows_env, args.output, args.portable_root, accept_eula=args.accept_eula)
    )


if __name__ == "__main__":
    main()
