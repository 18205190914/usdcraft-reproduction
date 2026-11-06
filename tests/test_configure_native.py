from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "configure_native", Path(__file__).parents[1] / "scripts/configure_native.py"
)
configure_native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configure_native)


@pytest.mark.parametrize("accept", [False, True])
def test_native_configuration_uses_selected_environment_and_keeps_profile(
    tmp_path, monkeypatch, accept
):
    env = tmp_path / "selected-env"
    env.joinpath("Scripts").mkdir(parents=True)
    for name in ("python.exe", "isaacsim.exe"):
        env.joinpath("Scripts", name).touch()
    monkeypatch.setattr(configure_native, "native_path", lambda p: "W:/" + p.name)
    output = tmp_path / "native"
    path = configure_native.configure(env, output, tmp_path / "portable", accept_eula=accept)
    config = json.loads(path.read_text())
    assert config["windows_python"] == str(env / "Scripts/python.exe")
    assert config["isaacsim"] == "W:/isaacsim.exe"
    assert config["environment"].get("OMNI_KIT_ACCEPT_EULA") == ("YES" if accept else None)
    kit = tomllib.loads((output / "physx.kit").read_text())
    assert kit["settings"]["app"]["exts"]["folders"] == ["W:/exts", "W:/extscache"]
    assert (
        output.joinpath("profile.json").read_bytes()
        == configure_native.REPO.joinpath("configs/usdcraft_physx.json").read_bytes()
    )
    for name in configure_native.NATIVE_SCRIPTS:
        assert (
            output.joinpath(name).read_bytes()
            == configure_native.REPO.joinpath("scripts/isaac", name).read_bytes()
        )


@pytest.mark.skipif(sys.platform == "win32", reason="Linux-only path rejection")
def test_linux_only_native_output_is_rejected_before_writing(tmp_path):
    env = tmp_path / "env"
    env.joinpath("Scripts").mkdir(parents=True)
    for name in ("python.exe", "isaacsim.exe"):
        env.joinpath("Scripts", name).touch()
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="shared"):
        configure_native.configure(env, output, tmp_path / "portable")
    assert not output.exists()
