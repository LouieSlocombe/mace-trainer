"""Command-line entry point for reference-data campaigns."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import load_config
from .runner import campaign_plan, run_campaign


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a campaign, print its plan, or run the configured dynamics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("config", type=Path, help="campaign JSON file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without ORCA",
    )
    args = parser.parse_args(argv)
    try:
        campaign = load_config(args.config)
        if args.dry_run:
            print(json.dumps(campaign_plan(campaign), indent=2, allow_nan=False))
        else:
            run_campaign(campaign)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"mace-trainer: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
