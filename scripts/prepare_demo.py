"""Generate local synthetic development inputs; nothing is downloaded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from articraft.usdcraft_benchmark import build_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    print(json.dumps(build_benchmark(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
