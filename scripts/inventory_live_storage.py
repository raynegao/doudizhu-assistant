from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.reporting.live_storage import inventory_live_storage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory live evidence without deleting files.")
    parser.add_argument("--root", type=Path, default=Path("data/live_game"))
    args = parser.parse_args(argv)
    print(json.dumps(inventory_live_storage(args.root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
