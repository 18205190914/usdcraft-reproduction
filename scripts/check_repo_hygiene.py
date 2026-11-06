"""Reject generated artifacts, environments and common secrets before publishing."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import yaml

SECRET = re.compile(
    r"sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{28,}|AIza[A-Za-z0-9_-]{35}|"
    r"github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".json",
    ".yml",
    ".yaml",
    ".txt",
    ".html",
    ".kit",
    ".example",
}
FORBIDDEN_SUFFIXES = {
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".ply",
    ".mp4",
    ".mov",
    ".avi",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".log",
    ".zip",
}
FORBIDDEN_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    "runs",
    "outputs",
    "results",
    "work",
    "data",
    "datasets",
    "weights",
}


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate workflow keys rather than silently accepting the last one."""


def unique_mapping(loader: yaml.Loader, node: yaml.MappingNode) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise yaml.YAMLError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def main() -> int:
    errors = []
    for workflow in sorted(Path(".github/workflows").glob("*.yml")):
        try:
            yaml.load(workflow.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        except yaml.YAMLError as exc:
            errors.append(f"Invalid workflow {workflow}: {exc}")
    files = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    for name in sorted(set(files) - {""}):
        path = PurePosixPath(name)
        forbidden = (
            any(part in FORBIDDEN_DIRECTORIES or part.endswith(".egg-info") for part in path.parts)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or (path.name.startswith(".env") and path.name != ".env.example")
            or path.name == ".DS_Store"
        )
        if forbidden:
            errors.append(f"Forbidden repository file: {name}")
        elif path.suffix.lower() in TEXT_SUFFIXES and SECRET.search(
            Path(name).read_text(encoding="utf-8")
        ):
            # Report only the filename; never echo a matched credential.
            errors.append(f"Possible credential in: {name}")
    for error in errors:
        print(error)
    if not errors:
        print("Source-only repository hygiene passed.")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
