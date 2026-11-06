"""Compile a checked-in SDK example without calling any model provider."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from articraft.agent.workspace import LocalWorkspace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("examples/hinged_box/main.py"))
    parser.add_argument("--output", type=Path, default=Path("outputs/examples"))
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    workspace = LocalWorkspace(output_dir=args.output.resolve(), physics_enabled=True)
    run = workspace.create_run("example-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    shutil.copyfile(source, run / "workspace/main.py")
    payload = workspace.compile_path(run)
    print(json.dumps({"run": str(run), "compile": payload}, indent=2))
    return 0 if payload["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
